# Leaf Disease Detection using Image Classification

This repository contains Python code for a deep learning model that classifies leaf diseases in rice plants. The model utilizes a Convolutional Neural Network (CNN) architecture to analyze images of diseased leaves and predict the type of disease (Bacterial Leaf Blight, Brown Spot, Leaf Smut).

**Dataset:**

* The dataset consists of images of rice leaves affected by different diseases.
* The images are divided into three categories: 
    * Bacterial Leaf Blight
    * Brown Spot
    * Leaf Smut
* The dataset is further split into training, validation, and testing sets.

**Model Architecture:**

* The model is a Sequential CNN with the following architecture:
    * Convolutional layers with ReLU activation and max-pooling for feature extraction.
    * Dropout layers to prevent overfitting.
    * Fully connected layers for classification.
    * Softmax activation in the output layer for multi-class classification.

**Training:**

* The model is trained using the Adam optimizer and categorical crossentropy loss.
* Data augmentation techniques (e.g., rescaling) are applied to the training data to improve model robustness.
* The model is trained for a specified number of epochs.

**Evaluation:**

* The model's performance is evaluated on the validation set during training.
* Training and validation accuracy/loss are plotted to monitor model performance.

**Dependencies:**

* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* scikit-learn (optional)

**Usage:**

1. **Install dependencies:**
   ```bash
   pip install tensorflow keras numpy matplotlib seaborn scikit-learn
