# MNIST Digit Classification using CNN

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** for classifying handwritten digits from the **MNIST dataset**. The CNN achieves high accuracy in recognizing digits and evaluating model performance using confusion matrices and classification metrics.

## Dataset
- The **MNIST dataset** consists of 70,000 grayscale images of handwritten digits (0-9), each of size **28x28 pixels**.
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

## Project Workflow
### 1. **Data Preprocessing**
   - Load the dataset and normalize pixel values to the range **[0,1]**.
   - Convert labels to **one-hot encoding**.
   - Reshape input images for CNN compatibility **(28,28,1)**.

### 2. **Building the CNN Model**
   - **Convolutional Layers (Conv2D)**: Extracts spatial features.
   - **MaxPooling Layers**: Reduces dimensionality and prevents overfitting.
   - **Flatten Layer**: Converts 2D feature maps into a 1D vector.
   - **Dense Layers**: Fully connected layers for classification.
   - **Dropout Layer**: Reduces overfitting by randomly deactivating neurons.

### 3. **Model Training**
   - **Loss Function**: Categorical Crossentropy
   - **Optimizer**: Adam Optimizer
   - **Batch Size**: 256
   - **Epochs**: 30
   - **Validation Data**: Used to monitor model performance.

### 4. **Performance Evaluation**
   - Achieved **99.49% training accuracy** and **99.73% validation accuracy**.
   - Computed **confusion matrix** to analyze misclassified digits.
   - Visualized training loss and validation loss.

### 5. **Model Saving & Prediction**
   - Saved the trained model as **network_for_mnist.h5**.
   - Performed digit classification on test images and plotted results.

## Installation & Dependencies
To run this project, install the required Python libraries:

```bash
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn
```

## Usage
Clone the repository and run the Jupyter Notebook:

```bash
git clone https://github.com/snouskap/mnist-cnn.git
cd pattern-skeleton
jupyter notebook
```

Then, open **pattern-skeleton.ipynb** in Jupyter Notebook and execute the cells sequentially.

## Results & Findings
- The CNN model achieved **99.73% validation accuracy**.
- The model successfully classified most handwritten digits with **minimal misclassification**.
- **Confusion matrix** visualization showed strong performance across all digit classes.

## Future Work
- Experiment with **data augmentation** to further improve accuracy.
- Try deeper CNN architectures like **ResNet or EfficientNet**.
- Implement **transfer learning** to improve generalization.


