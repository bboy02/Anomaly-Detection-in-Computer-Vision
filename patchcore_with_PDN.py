import os
import shutil
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from tqdm.auto import tqdm
import torch
from matplotlib.cm import get_cmap
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import Normalize

# Set up image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )
model_path = Path('models/teacher_small.pth')
teacher = get_pdn_small(384,False)
state_dict = torch.load(model_path, map_location=device)
teacher.load_state_dict(state_dict)

teacher.eval()


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        """This class extracts the feature maps from a pretrained Resnet model."""
        super(FeatureExtractor, self).__init__()
        out_channels = 384
        self.model = get_pdn_small(out_channels)
        state_dict = torch.load(r'models/teacher_small.pth', map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, input):
        with torch.no_grad():
                self.features = self.model(input)
                patch = self.features.reshape(self.features.shape[1], -1).T  # Create a column tensor
        return patch


# Create a feature extractor using the teacher model
feature_extractor = FeatureExtractor()

# Create memory bank from 'normal' data
memory_bank = []
folder_path = Path('')
# Function to check and load valid images (JPEG/JPG/PNG) and handle RGBA conversion
def load_image(pth):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if pth.suffix.lower() not in valid_extensions:
        return None  # Skip files with invalid formats

    try:
        img = Image.open(pth)
        if img.mode == 'RGBA':  # Convert RGBA to RGB if necessary
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {pth}: {e}")
        return None  # Return None if there's an error opening the image

# Extract features for 'good' images
for pth in tqdm(folder_path.iterdir(), leave=False):
    img = load_image(pth)  # Load the image
    if img is None:  # Skip invalid or unreadable files
        print(f"Skipping file {pth} due to invalid image format.")
        continue  # Skip to the next iteration if the image is None

    with torch.no_grad():
        try:
            data = transform(img).unsqueeze(0).to(device)  # Apply the transform only if img is valid
            features = feature_extractor(data)
            print(features.shape)
            memory_bank.append(features.cpu().detach())
        except Exception as e:
            print(f"Error processing file {pth}: {e}")
            continue  # Continue processing the next file if an error occurs

memory_bank = torch.cat(memory_bank, dim=0).to(device)

print("memory bank shape")
print(memory_bank.shape)
# # Only select 10% of total patches to avoid long inference time and computation
# selected_indices = np.random.choice(len(memory_bank), size=len(memory_bank) // 10, replace=False)
# memory_bank = memory_bank[selected_indices]


# For Anomalous Images
y_score = []
y_true = []

# Evaluate on 'defective' and 'good' classes'
for cls in ['defective', 'good']:
    folder_path = Path(f'')  # Fixing the path for each class
    for pth in tqdm(folder_path.iterdir(), leave=False):
        img = load_image(pth)  # Attempt to load the image
        if img is None:  # Skip invalid or unreadable files
            print(f"Skipping invalid image: {pth}")
            continue

        try:
            # Apply transformations and compute features
            data = transform(img).unsqueeze(0).to(device)
            features = feature_extractor(data)

            # Compute distances and anomaly score
            distances = torch.cdist(features, memory_bank, p=2.0)
            dist_score, dist_score_idxs = torch.min(distances, dim=1)
            s_star = torch.max(dist_score)

            # Update scores and ground truth labels
            y_score.append(s_star.cpu().numpy())
            y_true.append(0 if cls == 'good' else 1)
        except Exception as e:
            print(f"Error processing file {pth}: {e}")
            continue





# Ensure y_true contains both 0 and 1
if len(np.unique(y_true)) > 1:
    # Evaluation Metrics
    auc_roc_score = roc_auc_score(y_true, y_score)
    print("AUC-ROC Score:", auc_roc_score)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Calculate F1 scores for different thresholds
    f1_scores = [f1_score(y_true, y_score >= threshold) for threshold in thresholds]

    # Select the best threshold based on F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f'Best Threshold = {best_threshold}')
else:
    print("Only one class present in y_true. ROC AUC score is not defined.")



# Generate confusion matrix
cm = confusion_matrix(y_true, (y_score >= best_threshold).astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['OK', 'NOK'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# Printout predictions on the test set
from IPython.display import clear_output

feature_extractor.eval()
test_path = Path('')
paths = list(test_path.glob('*.png')) + list(test_path.glob('*.JPG'))
for path in paths:
    print(paths)

    test_image = transform(load_image(path)).unsqueeze(0).to(device)

    with torch.no_grad():
            features = feature_extractor(test_image)
            distances = torch.cdist(features, memory_bank, p=2.0)
            print("features")
            print(features.shape)
            print("memory bank")
            print(memory_bank.shape)
            print(torch.min(memory_bank), torch.max(memory_bank))
            print("Distance ")
            print(distances.shape)
            print(torch.min(distances), torch.max(distances))
            dist_score, dist_score_idxs = torch.min(distances, dim=1)
            print(f"dist  score shape {dist_score}")
            s_star = torch.max(dist_score)
            print(f"s_star is {s_star}")
            segm_map = dist_score.view(1, 1, 48,48)

            segm_map = torch.nn.functional.interpolate(segm_map, size=(224, 224), mode='bilinear').cpu().squeeze().numpy()
            y_score_image = s_star.cpu().numpy()
            y_pred_image = 1 * (y_score_image >= best_threshold)
            class_label = ['OK', 'NOK']

            plt.figure(figsize=(15, 5))

            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
            # plt.title(f'Fault Type: {fault_type}')

            # Anomaly Score Heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(segm_map, cmap='jet')
            plt.title(f'Anomaly Score: {y_score_image / best_threshold:.4f} || {class_label[y_pred_image]}')

            # Segmentation Map
            plt.subplot(1, 3, 3)
            plt.imshow((segm_map > best_threshold), cmap='gray')
            plt.title('Segmentation Map')

            plt.show()

            time.sleep(0.05)
            clear_output(wait=True)
    # with torch.no_grad():
    #     features = feature_extractor(test_image)
    #     distances = torch.cdist(features, memory_bank, p=2.0)
    #     print("features")
    #     print(features.shape)
    #     print("memory bank")
    #     print(memory_bank.shape)
    #     print(torch.min(memory_bank), torch.max(memory_bank))
    #     print("Distance ")
    #     print(distances.shape)
    #     print(torch.min(distances), torch.max(distances))
    #     dist_score, dist_score_idxs = torch.min(distances, dim=1)
    #     print(f"dist  score shape {dist_score}")
    #     s_star = torch.max(dist_score)
    #     print(f"s_star is {s_star}")
    #     segm_map = dist_score.view(1, 1, 48, 48)
    #
    #     segm_map = torch.nn.functional.interpolate(segm_map, size=(224, 224), mode='bilinear').cpu().squeeze().numpy()
    #     y_score_image = s_star.cpu().numpy()
    #     y_pred_image = 1 * (y_score_image >= best_threshold)
    #     class_label = ['OK', 'NOK']
    #
    #     # Normalize segmentation map to [0, 1] for colormap
    #     segm_map_normalized = Normalize(vmin=0, vmax=best_threshold)(segm_map)
    #
    #     # Colormap overlay
    #     colormap = get_cmap('jet')
    #     segm_colored = colormap(segm_map_normalized)[..., :3]  # Remove alpha channel from colormap
    #     overlay = 0.6 * test_image.squeeze().permute(1, 2, 0).cpu().numpy() + 0.4 * segm_colored  # Blend images
    #
    #     plt.figure(figsize=(20, 5))
    #
    #     # Original Image
    #     plt.subplot(1, 4, 1)
    #     plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
    #     plt.title('Original Image')
    #
    #     # Anomaly Score Heatmap
    #     plt.subplot(1, 4, 2)
    #     plt.imshow(segm_map, cmap='jet')
    #     plt.title(f'Anomaly Score: {y_score_image / best_threshold:.4f} || {class_label[y_pred_image]}')

        # # Segmentation Map
        # plt.subplot(1, 4, 3)
        # plt.imshow(segm_map > best_threshold, cmap='gray')
        # plt.title('Segmentation Map')

        # Overlay Image
        # plt.subplot(1, 4, 3)
        # plt.imshow(np.clip(overlay, 0, 1))  # Clip values to ensure valid image range
        # plt.title('Overlay')
        #
        # plt.show()
        #
        # time.sleep(0.05)
        # clear_output(wait=True)