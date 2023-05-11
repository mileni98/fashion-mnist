# Classification of Garment Images

This project was conducted as a part of my Bachelor's thesis in Data Science. The project aimed to explore the effectiveness of various machine learning algorithms (KNN, CNN, MLP, and SVM) for image classification tasks, using the Fashion-MNIST dataset.

The project used cross-validation to determine the optimal parameters for each algorithm and applied Principal Component Analysis (PCA) for dimensionality reduction. Standardization of features was also done to ensure consistency.

The results of this study demonstrate the potential of machine learning algorithms in real-world applications, particularly in image classification tasks. This project provides a comprehensive overview of the performance of different machine learning algorithms on the Fashion-MNIST dataset.

## Dataset

The Fashion-MNIST dataset contains a set of 70,000 grayscale images of size 28x28 pixels, which are divided into 10 different classes. 


## Algorithms Used

This project utilizes the following four machine learning algorithms for image classification:

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Multilayer Perceptron (MLP)
4. Convolutional Neural Network (CNN)


## Workflow of Each Algorithm File

The workflow for each algorithm file in this project is as follows:

- Cross-validation hyperparameter tuning is performed on 30% of the original training dataset to determine the best parameters.
- The model is trained using these parameters on 30% of the original training dataset and tested on the test dataset.

- The model is then trained using cross-validation on 100% of the original training dataset and tested on the test dataset.

- Cross-validation hyperparameter tuning is performed on 100% of the dataset with Principal Component Analysis (PCA) reduction to determine new best parameters.
- The model is trained using these parameters on 100% of the PCA-reduced dataset and tested on the test dataset.


## Conclusion


## Running the project

To use this project, follow these steps:

1. Create a "datasets" directory in the root folder.
2. Donwload "fashion-mnist_test.csv" and "fashion-mnist_train.csv" datasets from this link:
    https://www.kaggle.com/datasets/zalando-research/fashionmnist
3. Put the downloaded files into the "datasets" folder.
4. Make sure all the required dependencies are installed.
5. Run additonal_metrics.py and datasets_analysis.py files first.
6. Then run any of the algorithm files.


## Dependencies

This project has the following depencencies:

- pandas
- numpy
- matplotlib
- scikit-learn
- TensorFlow
- Keras
- Additional_metrics