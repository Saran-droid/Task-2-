# Task-2-
**Plant Disease Classification using ResNet-50**
**Introduction**
This project aims to classify plant diseases from images of leaves using deep learning techniques. We leverage the power of a pretrained ResNet-50 model to perform transfer learning and achieve high accuracy in identifying different plant diseases. The notebook walks through the entire process, from data loading and preprocessing to model training, evaluation, and visualization of results.
![image](https://github.com/user-attachments/assets/9aec93ac-372a-4e56-8c37-5248aaf931c5)

**Data Loading and Preprocessing**
**Data Description**
The dataset used in this project consists of images of plant leaves, each labeled with a specific disease. The dataset is split into training and validation sets to facilitate model training and evaluation.

**Model Architecture**
ResNet-50
ResNet-50 is a powerful convolutional neural network architecture known for its deep layers and residual connections. In this project, we use a pretrained ResNet-50 model, which has been trained on the ImageNet dataset. This allows us to leverage the learned features and adapt them to our specific task of plant disease classification.

**Custom Layers**
We add custom fully connected layers on top of the ResNet-50 base model. These layers are tailored to our classification task, adjusting the number of output neurons to match the number of disease classes in our dataset.

**Model Compilation**
To prepare the model for training, we compile it using the categorical_crossentropy loss function, suitable for multi-class classification problems. We use the Adam optimizer, which is well-regarded for its efficiency and effectiveness in training deep learning models.

**Model Training**
Training Process
The training process involves fitting the model to the training data and validating it on the validation set. We specify parameters such as the number of epochs and batch size. Throughout the training process, we monitor the training and validation accuracy and loss to ensure the model is learning effectively.

**Early Stopping**
To prevent overfitting, we employ an early stopping mechanism. This stops the training process if the validation loss does not improve for a specified number of epochs, ensuring the model maintains good generalization performance.

**Model Evaluation**
**Performance Metrics**
After training, we evaluate the model's performance on the validation set. Key metrics such as accuracy and loss provide insights into how well the model has learned to classify plant diseases. Additionally, we may look at the confusion matrix to understand the model's performance on individual classes.

**Visualizing Results**
Visualization plays a crucial role in understanding the model's performance. We plot the training and validation accuracy and loss over the epochs, providing a clear picture of the model's learning curve. We also visualize some example predictions on test images, showcasing the model's ability to correctly classify plant diseases.
![image](https://github.com/user-attachments/assets/04afbd94-8e73-4669-8029-393c807dc8d9)![image](https://github.com/user-attachments/assets/2120c175-42a8-4583-9101-c1f7a2b947ac)


**Conclusion**
This project demonstrates the process of building and training a deep learning model for plant disease classification using ResNet-50. By leveraging transfer learning, we can achieve high accuracy with relatively little training data. This approach can be extended to other image classification tasks, highlighting the versatility and power of deep learning models.

![image](https://github.com/user-attachments/assets/40d52b0e-2209-497a-a79d-3b2c94ec4cc0)
