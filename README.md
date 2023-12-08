# Colonoscopy-Image-Classification
## Colonoscopy Images Classification using a Compact Convolutional Transformers (CCT)

Colonoscopy Image Classification Using CCT
Project Overview
This project aims to classify colonoscopy images into two categories: benign and malignant. The classification is crucial in medical diagnostics to identify potential adenomas, serrated adenomas, and hyperplastic conditions in colonoscopy images. We utilize a Compact Convolutional Transformer (CCT) model for this task, leveraging its efficient design for image recognition.

Dataset
The dataset was obtained from the UAH Colonoscopy Dataset. The original dataset includes 80 adenomas, 30 serrated adenomas, and 42 hyperplastic images. These images were manually downloaded using BeautifulSoup for web scraping. The original images had dimensions of 768x576 pixels and contained extraneous elements like letters, menu windows, dark margins, and medical tools.

Preprocessing Steps
Image Editing: Using GIMP, each image was cropped to 480x360 pixels to remove unwanted content.
Data Augmentation: To balance the dataset, the 42 hyperplastic images (classified as benign) were flipped vertically and horizontally, resulting in 126 images. To achieve a balanced dataset, 16 flipped images were manually removed, resulting in 110 benign images.
Data Organization: The images were categorized into two folders: 'benign' (hyperplastic images) and 'malignant' (adenomas and serrated adenomas), leading to a balanced dataset of 110 images in each category.
CCT Model
We employed the CCT model from the SHI-Labs Compact Transformers repository. This model is known for its efficiency in processing images, making it a suitable choice for our task.

Model Configuration
Image size: 256x256 pixels.
Embedding dimension: 768.
Number of convolution layers: 1.
Number of transformer layers: 7.
Number of heads: 6.
Model trained to classify two classes: benign and malignant.
Training Process
The model was trained using PyTorch. A custom training loop was implemented with the following key features:

Loss Function: CrossEntropyLoss.
Optimizer: Adam optimizer with a learning rate of 0.001.
Learning Rate Scheduler: StepLR scheduler with a step size of 5 and gamma of 0.1.
Early Stopping: Implemented to prevent overfitting, with a patience of 5 epochs.
Fine-Tuning Changes
Image Size Reduction: Original images were reduced from 480x360 to 256x256 for computational efficiency.
Data Augmentation Removal: Certain augmentations (RandomPerspective and RandomGrayscale) were removed after observing that they did not significantly contribute to the model's performance.
Balanced Dataset: Class weights were initially used but later removed after balancing the dataset through augmentation and manual editing.
Results and Metrics
The model was evaluated using metrics such as accuracy, precision, recall, and F1 score. Here's a summary of the performance over epochs:

Epoch 1: Accuracy: 47.73%, Precision: 0.48, Recall: 0.45, F1 Score: 0.47
Epoch 2: Accuracy: 47.73%, Precision: 0.48, Recall: 0.50, F1 Score: 0.49
Epoch 3: Accuracy: 50.00%, Precision: 0.50, Recall: 0.45, F1 Score: 0.48
Epoch 4: Accuracy: 50.00%, Precision: 0.50, Recall: 0.50, F1 Score: 0.50
Epoch 5: Accuracy: 54.55%, Precision: 0.55, Recall: 0.55, F1 Score: 0.55
Epoch 6: Accuracy: 61.36%, Precision: 0.61, Recall: 0.64, F1 Score: 0.62
Epoch 7: Accuracy: 40.91%, Precision: 0.39, Recall: 0.32, F1 Score: 0.35
Epoch 8: Accuracy: 50.00%, Precision: 0.50, Recall: 0.45, F1 Score: 0.48
Epoch 9: Accuracy: 50.00%, Precision: 0.50, Recall: 0.45, F1 Score: 0.48
Epoch 10: Accuracy: 50.00%, Precision: 0.50, Recall: 0.45, F1 Score: 0.48
Epoch 11: Accuracy: 52.27%, Precision: 0.52, Recall: 0.50, F1 Score: 0.51
Early stopping was triggered after the 11th epoch.
Interpretation of Results
The model showed a gradual improvement in accuracy and F1 score over the initial epochs, peaking at epoch 6. The precision and recall metrics fluctuated, reflecting the model's varying ability to correctly identify positive cases across epochs. Early stopping was triggered after the 11th epoch, suggesting that the model had begun to overfit or ceased to make significant learning progress.

Confusion Matrices
The confusion matrices for each epoch provided insights into the model's performance regarding false positives and false negatives. Here are a few notable confusion matrices:

Epoch 1: [[11 11] [12 10]]
Epoch 5: [[12 10] [10 12]]
Epoch 6: [[13 9] [ 8 14]] (Best performance)
Epoch 11: [[12 10] [11 11]]
The confusion matrix from Epoch 6 indicates the best balance between true positives and true negatives.

Conclusion and Future Work
This project demonstrates the potential of using CCT for medical image classification. The model achieved modest performance, with the best results observed in the 6th epoch. Future work could focus on further tuning the model architecture, experimenting with different data augmentation techniques, and possibly exploring ensemble methods to improve accuracy and robustness.

Potential improvements include more extensive data preprocessing, exploring more sophisticated data augmentation techniques, and utilizing a larger and more diverse dataset for training.


