"""PatchCore"""
import logging
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from common import *
from sampler import *
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, confusion_matrix, \
    classification_report, roc_curve
import matplotlib.pyplot as plt
from backbones import load
LOGGER = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=ApproximateGreedyCoresetSampler(device=device,percentage=0.1),
        nn_method=FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def  _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]


        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]

        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)

        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
                input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image




#Load Training Data:


train_root_dir = os.path.abspath("")
print("Dataset Path:", train_root_dir)
train_data = CustomDataset(root_dir=train_root_dir, transform=transform)
print(f"Length of train data : {len(train_data)}" )
train_loader = DataLoader(train_data, batch_size=4, shuffle=False)


# Load Validation Data
# Define paths for normal and zflower validation images
val_normal_root_dir = os.path.abspath(
    "")
val_zflower_root_dir = os.path.abspath(
    "")
# Load normal validation images
val_normal_data = CustomDataset(root_dir=val_normal_root_dir, transform=transform)
print(f"Length of validation normal data: {len(val_normal_data)}")
# Load zflower validation images
val_zflower_data = CustomDataset(root_dir=val_zflower_root_dir, transform=transform)
print(f"Length of validation zflower data: {len(val_zflower_data)}")
# Create data loaders for normal and zflower images
val_normal_loader = DataLoader(val_normal_data, batch_size=4, shuffle=False)
val_zflower_loader = DataLoader(val_zflower_data, batch_size=4, shuffle=False)

# Load Testing Data
# Define paths for normal and zflower testing images
test_normal_root_dir = os.path.abspath(
    "")


test_zflower_root_dir = os.path.abspath(
    "")

# Load normal testing images
test_normal_data = CustomDataset(root_dir=test_normal_root_dir, transform=transform)
print(f"Length of testing normal data: {len(test_normal_data)}")
# Load zflower testing images (anomalous)
test_zflower_data = CustomDataset(root_dir=test_zflower_root_dir, transform=transform)
print(f"Length of testing zflower data: {len(test_zflower_data)}")
# Create data loaders for normal and zflower images
test_normal_loader = DataLoader(test_normal_data, batch_size=4, shuffle=False)
test_zflower_loader = DataLoader(test_zflower_data, batch_size=4, shuffle=False)



patchcore_model = PatchCore(device)
layers_to_extract_from = ["layer2","layer3"]
backbone = load("resnet50").to(device)

patchcore_model.load(
    backbone=backbone,
    layers_to_extract_from=layers_to_extract_from,
    device=device,
    input_shape=(3, 512, 512),  # Change to your dataset's input shape
    pretrain_embed_dimension=512,
    target_embed_dimension=224,
    patchsize=3,
    patchstride=1,
)

# Training phase
patchcore_model.fit(train_loader)


#Validation Phase :
def calculate_optimal_threshold(scores, labels_gt):
    """Calculate the optimal threshold based on precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels_gt, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Log information
    print(f"Optimal F1-Score: {f1_scores[optimal_idx]:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Optional: Plot Precision-Recall Curve
    import matplotlib.pyplot as plt
    plt.plot(thresholds, f1_scores[:-1], label="F1-Score")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score vs. Threshold")
    plt.legend()
    plt.show()
    return optimal_threshold


def evaluate_performance(scores, labels_gt, threshold):
    """Evaluate performance metrics based on the threshold."""
    predictions = (scores >= threshold).astype(int)
    accuracy = accuracy_score(labels_gt, predictions)
    conf_matrix = confusion_matrix(labels_gt, predictions)
    report = classification_report(labels_gt, predictions)
    auroc = roc_auc_score(labels_gt, scores)
    return accuracy, conf_matrix, report, auroc


def plot_validation_curves(scores, labels_gt):
    """Visualize Precision-Recall and ROC curves."""
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels_gt, scores)
    plt.figure()
    plt.plot(recall, precision, marker='.', label="Precision-Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels_gt, scores)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label="ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def validation_phase(model, val_normal_loader, val_zflower_loader):
    """
    Perform validation to evaluate model's anomaly detection capabilities.

    Args:
        model: The trained anomaly detection model.
        val_normal_loader: DataLoader for normal images in validation.
        val_zflower_loader: DataLoader for anomalous (zflower) images in validation.

    Returns:
        accuracy, conf_matrix, report, auroc, optimal_threshold
    """
    model.eval()  # Set the model to evaluation mode

    # Initialize lists to collect scores and ground truth
    scores = []
    labels_gt = []

    # Process normal images
    with tqdm.tqdm(val_normal_loader, desc="Processing Normal Images") as data_iterator:
        for image in data_iterator:
            try:
                if not isinstance(image, dict):  # Handle dict-based DataLoader
                    labels_gt.extend([0] * image.shape[0])
                # Get anomaly scores for normal images
                normal_scores, _ = model._predict(image)
                scores.extend(normal_scores)
            except Exception as e:
                print(f"Skipping corrupted normal image due to error: {e}")

    # Process anomalous (zflower) images
    with tqdm.tqdm(val_zflower_loader, desc="Processing Anomalous (Zflower) Images") as data_iterator:
        for image in data_iterator:
            try:
                if not isinstance(image, dict):  # Handle dict-based DataLoader
                    labels_gt.extend([1] * image.shape[0])  # Label for anomalous images
                # Get anomaly scores for anomalous images
                zflower_scores, _ = model._predict(image)
                scores.extend(zflower_scores)
            except Exception as e:
                print(f"Skipping corrupted anomalous image due to error: {e}")

    # Convert scores and ground truth to numpy arrays
    scores = np.array(scores)
    labels_gt = np.array(labels_gt)

    # Step 1: Calculate optimal threshold
    optimal_threshold = calculate_optimal_threshold(scores, labels_gt)
    print(f"Optimal Threshold: {optimal_threshold}")

    # Step 2: Evaluate performance
    accuracy, conf_matrix, report, auroc = evaluate_performance(scores, labels_gt, optimal_threshold)
    print(f"Validation Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{report}")
    print(f"Validation AUROC: {auroc}")

    # Step 3: Visualize performance
    plot_validation_curves(scores, labels_gt)

    return accuracy, conf_matrix, report, auroc, optimal_threshold

#accuracy, conf_matrix, report, auroc, optimal_threshold = validation_phase(patchcore_model, val_normal_loader, val_zflower_loader)

#Testing Phase:
def inference_phase(model, test_normal_loader, test_zflower_loader, optimal_threshold):
    """
    Perform testing to evaluate the model's generalization on unseen data.
    Args:
        model: The trained anomaly detection model.
        test_normal_loader: DataLoader for normal images in testing.
        test_zflower_loader: DataLoader for anomalous (zflower) images in testing.
        optimal_threshold: Threshold determined from the validation phase.

    Returns:
        accuracy, conf_matrix, report, auroc
    """
    model.eval()  # Set the model to evaluation mode

    # Initialize lists to collect scores, ground truth labels, and segmentation masks
    scores = []
    masks = []
    labels_gt = []

    # Process normal images
    with tqdm.tqdm(test_normal_loader, desc="Processing Normal Test Images") as data_iterator:
        for image in data_iterator:
            try:
                if not isinstance(image, dict):  # Handle dict-based DataLoader
                    labels_gt.extend([0] * image.shape[0])  # Label for normal images
                # Get anomaly scores and segmentation masks for normal images
                normal_scores, normal_masks = model._predict(image)
                scores.extend(normal_scores)
                masks.extend(normal_masks)
            except Exception as e:
                print(f"Skipping corrupted normal test image due to error: {e}")

    # Process anomalous (zflower) images
    with tqdm.tqdm(test_zflower_loader, desc="Processing Anomalous Test Images") as data_iterator:
        for image in data_iterator:
            try:
                if not isinstance(image, dict):  # Handle dict-based DataLoader
                    labels_gt.extend([1] * image.shape[0])  # Label for anomalous images
                # Get anomaly scores and segmentation masks for anomalous images
                zflower_scores, zflower_masks = model._predict(image)
                scores.extend(zflower_scores)
                masks.extend(zflower_masks)
            except Exception as e:
                print(f"Skipping corrupted anomalous test image due to error: {e}")

    # Convert scores, masks, and ground truth labels to numpy arrays
    scores = np.array(scores)
    masks = np.array(masks)
    labels_gt = np.array(labels_gt)

    # Step 1: Evaluate performance
    accuracy, conf_matrix, report, auroc = evaluate_performance(scores, labels_gt, optimal_threshold)
    print(f"Testing Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{report}")
    print(f"Testing AUROC: {auroc}")
    return accuracy, conf_matrix, report, auroc



# Run the testing phase
optimal_threshold = optimal_threshold
accuracy, conf_matrix, report, auroc = inference_phase(patchcore_model, test_normal_loader, test_zflower_loader, optimal_threshold)




# Test the model
scores, masks, labels_gt, masks_gt = patchcore_model.predict(test_zflower_loader)



def normalize_map(anomaly_map):
    anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
    return anomaly_map

def visualize_anomaly(image, masks, score, idx):
    """Visualizes the anomaly map."""
    # Convert image to HWC format and NumPy array
    image_np = image.permute(1, 2, 0).numpy()


    # Resize anomaly map to match image size
    anomaly_map = normalize_map(masks)


    # Check if the anomaly map size matches the image size
    if anomaly_map.shape != image_np.shape[:2]:
        anomaly_map = cv2.resize(anomaly_map, (image_np.shape[1], image_np.shape[0]))

    # Display original image, anomaly map, and overlay
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Anomaly Map (Score: {score:.2f})")
    plt.imshow(anomaly_map, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3,3)
    plt.title("Overlay")
    plt.imshow(image_np, alpha=0.7)
    plt.imshow(anomaly_map, cmap="jet", alpha=0.3)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"anomaly_visual_{idx}.png")
    plt.show()


# Visualize a few samples
for idx, (image, masks, score) in enumerate(zip(test_zflower_data, masks, scores)):
    if idx < 20:  # Visualize the first 20 images
        visualize_anomaly(image, masks, score, idx)


#end
