# Deep-Learning-Diseases-Classification-Thermal-Image
# Thermal Images Classification

## Project Overview
This project aims to classify thermal images of leaves into different categories: 
- **Blast**
- **BLB (Bacterial Leaf Blight)**
- **Healthy**
- **Hispa**
- **Leaf Folder**
- **Leaf Spot**

The dataset used for this classification is sourced from Kaggle:
[Thermal Images of Diseased and Healthy Leaves - Paddy](https://www.kaggle.com/sujaradha/thermal-images-diseased-healthy-leaves-paddy).

## Dependencies
The following libraries are required to run the notebook:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random
```

## Data Preprocessing and Augmentation
The dataset undergoes preprocessing using the following transformations:
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
- **Resizing**: Images are resized to 299x299 pixels.
- **Random Horizontal Flip**: Helps in reducing overfitting.
- **Random Rotation**: Rotates images randomly by up to 10 degrees.
- **Normalization**: Applies ImageNet mean and standard deviation normalization.

## Model Training
The project utilizes a deep learning model from the `torchvision.models` module. The training pipeline includes:
1. **Dataset Loading**: Images are loaded using PyTorch's `DataLoader`.
2. **Model Selection**: A pre-trained model (such as InceptionV3, ResNet, or others) is fine-tuned for classification.
3. **Loss Function & Optimizer**: Uses Cross-Entropy Loss and an optimizer like Adam or SGD.
4. **Evaluation Metrics**: Classification performance is measured using precision, recall, F1-score, and confusion matrix.

## How to Run the Notebook
1. Download the dataset from Kaggle.
2. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib tqdm scikit-learn pillow
   ```
3. Open the notebook (`Thermal_Images.ipynb`) in Jupyter Notebook or Google Colab.
4. Ensure the dataset is correctly placed in the expected directory.
5. Run the notebook cells sequentially to train and evaluate the model.

## Results and Analysis

### Training Performance
| Model        | Final Training Accuracy (%) | Final Training Loss |
|-------------|-----------------------------|----------------------|
| ResNet50    | 98.43%                      | 0.0383               |
| ResNet101   | 98.69%                      | 0.0356               |
| EfficientNetV2 | 99.74%                   | 0.0185               |
| ViT         | 100.00%                     | 0.0033               |
| GoogleNet   | 99.48%                      | 0.0394               |
| VGG16       | 97.38%                      | 0.0518               |
| AlexNet     | 96.85%                      | 0.0772               |
| Inception v3 | 99.74%                      | 0.0260               |

✅ **ViT achieved 100% accuracy but took the longest training time.**
✅ **EfficientNetV2 had the lowest loss, indicating strong convergence.**

### Validation & Testing Performance
| Model        | Validation Accuracy (%) | Test Accuracy (%) |
|-------------|-------------------------|-------------------|
| ResNet50    | 90.55%                  | 94.53%            |
| ResNet101   | 91.34%                  | 94.53%            |
| EfficientNetV2 | 89.76%               | 96.09%            |
| ViT         | 90.55%                  | 95.31%            |
| GoogleNet   | 93.70%                  | 92.97%            |
| VGG16       | 96.06%                  | 92.19%            |
| AlexNet     | 92.13%                  | 91.41%            |
| Inception v3 | 90.55%                  | 95.31%            |

✅ **VGG16 had the highest validation accuracy (96.06%) but slightly lower test accuracy.**
✅ **EfficientNetV2 performed best on unseen data (96.09% test accuracy).**

## Future Improvements
- Experimenting with different CNN architectures like EfficientNet, ViT, etc.
- Using more advanced augmentation techniques.
- Increasing the dataset size to improve generalization.

## Acknowledgments
Dataset: [Kaggle - Thermal Images of Diseased and Healthy Leaves - Paddy](https://www.kaggle.com/sujaradha/thermal-images-diseased-healthy-leaves-paddy).

