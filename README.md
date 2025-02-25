
# Brain Tumor MRI Classification Using TensorFlow CNN

## Project Description
This project focuses on the classification of brain tumor MRI images using Convolutional Neural Networks (CNNs) in TensorFlow. Due to the small size of the dataset, Transfer Learning is employed with a pre-trained EfficientNetB0 model to enhance the performance. The objective is to build a robust deep-learning model that can accurately distinguish between different categories of brain tumors and non-tumorous images.

This model has potential applications in healthcare, where it could assist in the early detection of brain tumors, thus speeding up diagnosis and aiding in clinical decision-making.

### Dataset Categories:
1. Glioma Tumor
2. No Tumor
3. Meningioma Tumor
4. Pituitary Tumor

## Built With
- **Python**: Core language used throughout the project.
- **TensorFlow & Keras**: Deep learning frameworks for constructing and training CNN models.
- **Google Colab**: Cloud-based platform for executing the notebook with GPU acceleration.
- **OpenCV**: Image preprocessing, including resizing MRI images and data augmentation.
- **EfficientNetB0**: Pre-trained model on the ImageNet dataset used for Transfer Learning.
- **Scikit-learn**: For splitting datasets and evaluating model performance.
- **Matplotlib & Seaborn**: For visualizing data and plotting loss/accuracy graphs.
- **Pandas & Numpy**: For managing and manipulating data, including loading and organizing datasets.

## Model Overview
This project employs a Convolutional Neural Network (CNN) built on top of TensorFlow. Due to the small dataset size, Transfer Learning is employed with the EfficientNetB0 model, which is pre-trained on the ImageNet dataset. The model is trained to classify MRI scans into four categories, and its performance is optimized using techniques like early stopping and learning rate reduction.

### Image Preprocessing
MRI images are resized to 150x150 pixels and augmented using techniques such as rotation and zooming. OpenCV is used for handling this preprocessing to ensure that the model generalizes well across different images.

### Transfer Learning
EfficientNetB0, pre-trained on ImageNet (1000 classes), is fine-tuned for this specific task. The pre-trained weights help in faster convergence and better performance on the MRI dataset.

### Performance Metrics
The model is evaluated using:
- **Accuracy**: Overall correctness of the model’s predictions.
- **Confusion Matrix**: Helps visualize true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Includes precision, recall, F1-score, and support for each class.

## Results
The final trained model achieves high accuracy (98%) and robust classification of brain tumors. The use of Transfer Learning with EfficientNetB0 has significantly improved model performance and training time as we didn't have enough data to train CNN from the beginning.

## Conclusion
This project demonstrates the efficacy of using CNNs with Transfer Learning for brain tumor MRI classification. By leveraging a pre-trained model (EfficientNetB0) and optimizing the training process, the model can achieve high accuracy and support healthcare professionals in diagnosing brain tumors faster.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
