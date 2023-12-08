# Colonoscopy Image Classification Using CCT
## Project Overview
This project aims to classify colonoscopy images into two categories: benign and malignant. The classification is crucial in medical diagnostics to identify potential adenomas, serrated adenomas, and hyperplastic conditions in colonoscopy images. We utilize a Compact Convolutional Transformer (CCT) model for this task, leveraging its efficient design for image recognition.

## Dataset
The dataset was obtained from the [UAH Colonoscopy Dataset](https://www.depeca.uah.es/colonoscopy_dataset/) . The original dataset includes 80 adenomas, 30 serrated adenomas, and 42 hyperplastic images. These images were manually downloaded using [BeautifulSoup](https://pypi.org/project/beautifulsoup4/) for web scraping. The original images had dimensions of 768x576 pixels and contained extraneous elements like letters, menu windows, dark margins, and medical tools.

### Preprocessing Steps
- **Image Editing:** Using GIMP, each image was cropped to 480x360 pixels to remove unwanted content.
- **Data Augmentation:** To balance the dataset, the 42 hyperplastic images (classified as benign) were flipped vertically and horizontally, resulting in 126 images. To achieve a balanced dataset, 16 flipped images were manually removed, resulting in 110 benign images.
- **Data Organization:** The images were categorized into two folders: 'benign' (hyperplastic images) and 'malignant' (adenomas and serrated adenomas), leading to a balanced dataset of 110 images in each category.
### CCT Model
We employed the CCT model from the [SHI-Labs Compact Transformers repository](https://github.com/SHI-Labs/Compact-Transformers). This model is known for its efficiency in processing images, making it a suitable choice for our task.

### Model Configuration
- **Image size:** 256x256 pixels.
- **Embedding dimension:** 768.
- **Number of convolution layers:** 1.
- **Number of transformer layers:** 7.
- **Number of heads:** 6.
- **Model trained to classify two classes:** benign and malignant.
### Training Process
The model was trained using PyTorch. A custom training loop was implemented with the following key features:

- **Loss Function:** CrossEntropyLoss.
- **Optimizer:** Adam optimizer with a learning rate of 0.001.
- **Learning Rate Scheduler:** StepLR scheduler with a step size of 5 and gamma of 0.1.
- **Early Stopping:** Implemented to prevent overfitting, with a patience of 5 epochs.
- ## Fine-Tuning Changes
- **Image Size Reduction:** Original images were reduced from 480x360 to 256x256 for computational efficiency.
- **Data Augmentation Removal:** Certain augmentations (RandomPerspective and RandomGrayscale) were removed after observing that they did not significantly contribute to the model's performance.
- **Balanced Dataset:** Class weights were initially used but later removed after balancing the dataset through augmentation and manual editing.
### Results and Metrics
The model's performance was evaluated using metrics such as accuracy, precision, recall, and F1 score. Here's a summary of the performance over epochs:

| Epoch    | Train Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score | Confusion Matrix   |
| -------- | ---------- | --------------- | -------- | --------- | ------ | -------- | ------------------ |
| 1/20     | 63.7728    | 0.7045          | 50.00%   | 0.54      | 0.30   | 0.39     | [[15  6] [16  7]]  |
| 2/20     | 60.3948    | 0.7121          | 45.45%   | 0.47      | 0.39   | 0.43     | [[11 10] [14  9]]  |
| 3/20     | 60.4134    | 0.7145          | 47.73%   | 0.50      | 0.30   | 0.38     | [[14  7] [16  7]]  |
| 4/20     | 60.5688    | 0.6879          | 45.45%   | 0.46      | 0.26   | 0.33     | [[14  7] [17  6]]  |
| 5/20     | 60.2448    | 0.7120          | 45.45%   | 0.45      | 0.22   | 0.29     | [[15  6] [18  5]]  |
| 6/20     | 60.2571    | 0.6964          | 54.55%   | 0.62      | 0.35   | 0.44     | [[16  5] [15  8]]  |
| 7/20     | 61.3196    | 0.7053          | 50.00%   | 0.53      | 0.35   | 0.42     | [[14  7] [15  8]]  |
| 8/20     | 60.4732    | 0.7109          | 45.45%   | 0.46      | 0.26   | 0.33     | [[14  7] [17  6]]  |
| 9/20     | 60.5204    | 0.6983          | 47.73%   | 0.50      | 0.30   | 0.38     | [[14  7] [16  7]]  |
| Early Stopping Triggered                    |

## Interpretation of Results
The model displayed varying degrees of performance across different metrics. The highest accuracy achieved was 54.55% in the 6th epoch.
Precision and recall values fluctuated, suggesting challenges in achieving a consistent balance in the model's predictive capabilities.
F1 Score, a measure of the test's accuracy, varied across epochs, with the highest being 0.44 in the 6th epoch.

## Confusion Matrices
The confusion matrices provide valuable insights into the model's classification accuracy for each class.
For instance, the confusion matrix from Epoch 6 ([[16 5] [15 8]]) indicates an improvement in identifying true positives and true negatives compared to earlier epochs.
This matrix suggests a better balance between sensitivity (true positive rate) and specificity (true negative rate) in the 6th epoch.

## Conclusion and Future Work
This project demonstrates the potential of using CCT for medical image classification. The model achieved modest performance, with the best results observed in the 6th epoch. Future work could focus on further tuning the model architecture, experimenting with different data augmentation techniques, and possibly exploring ensemble methods to improve accuracy and robustness.

## Setup and Installation
### Requirements
- Python 3.7+
- Required Libraries: PyTorch, torchvision, scikit-learn, numpy, matplotlib, BeautifulSoup4, timm
### Running the Code
1. **Download the Jupyter Notebook (MVAI_v04_Final_Project.ipynb).**
2. **Download and unzip utils.zip from Compact Transformers (or use the included utils.zip).**
3. **Download and unzip the dataset files (benign.zip and malignant.zip). Place them in a folder named 'Colonoscopy Images 3' (or your chosen name).**
4. **Install required libraries.**
5. **Run the Jupyter Notebook and follow the instructions within.**
