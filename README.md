# Facial Emotion Recognition (FER) using Convolution Neural Networks
![Python](https://img.shields.io/badge/python-v3.11.4+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

![ezgif com-crop](https://github.com/ACM40960/project-22200226/assets/114998243/8e0d0e6a-a864-4616-9b60-b7a16abd7937)


## Introduction
In this project, we build a Convolutional Neural Network (CNN) model with the ability to categorize seven different emotional states in human faces: Anger, Disgust, Fear, Happy, Sad, Surprise, and Neutral. Furthermore, we put this model into practical use by integrating it with a webcam for real-time application.

The benefits of this project reach different areas. For example, in healthcare, it can make it easier for people with Alexithymia to understand how others feel and to express their own emotions *(People with Alexithymia have difficulties recognizing and communicating their own emotions, and they also struggle to recognize and respond to emotions in others)*. It also has uses in public safety, education, jobs, and more.

## Getting Started - Instructions
Get ready for fun! Follow instructions for real-time facial emotion recognition through your webcam, all done using Python in Jupyter Notebook (Anaconda). Let's go! 🚀😃

*Note - You can find a tutorial for downloading Python with Anaconda at this [link](https://docs.anaconda.com/free/anaconda/install/).*

### Files Structure:
- [fer_main.ipynb](fer_main.ipynb) - Main project file about the dataset, train the CNN, model evaluation, and analysis
- [fer_webcam.ipynb](fer_webcam.ipynb) - Uses the pre-trained model to predict emotions via webcam
- [haarcascade_frontalface_default](haarcascade_frontalface_default.xml) - Face detection algorithm
  (we obtain this from this [repository](https://github.com/opencv/opencv/tree/master/data/haarcascades))
- [model_final.json](model_final.json) - Neural network architecture
- [weights_final.h5](weights_final.h5) - Trained model weights
- [requirements.txt](requirements.txt) - Version of each dependency
- CNN Visualization folder - Source code for generating visual representation of CNN architecture (created using LaTeX format)
- [gitattributes](gitattributes) - Source code to upload large file more than 25mb to Github (to be ignored)
- [Literature Review](literature_review.pdf) - Synopsis of Facial Emotion Recognition project
- [Project Poster](Poster%20-%20FER%20using%20CNN.pdf) - Project Poster

## Prerequisites
Install these prerequisites before proceeding:
```
pip3 install numpy
pip3 install pandas
pip3 install seaborn
pip3 install keras
pip3 install matplotlib
pip3 install plotly
pip3 install scikit-learn
pip3 install tensorflow
pip3 install opencv-python
```


### Method 1 : Using the built model 

No need to train from scratch! Just use [fer_webcam.ipynb](fer_webcam.ipynb) with pre-trained [model_final.json](model_final.json) and [weights_final.h5](weights_final.h5) to predict facial emotions real time on your webcam. Customize it for your needs! 🤩

### Method 2 : Start from scratch
Let's get started with building the model from scratch! Follow these steps:

1. Clone the repository using the command:  
```
https://github.com/ACM40960/project-22200226.git
```

2. Download and extract [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).

This dataset contains 35,887 facial image extracts where emotion labels are linked to the corresponding pixel values in each image. It covers 7 distinct emotions/classes: Angry (0), Disgust (1), Fear (2), Happy (3), Sad (4), Surprise (5), and Neutral (6). The dataset is split into sections for training, validation, and testing purposes. All images are in grayscale and have dimensions of 48 x 48 pixels.

3. Run [fer_main.ipynb](fer_main.ipynb) and modify to your needs!


## Our Work
### Methodology


![image](https://github.com/ACM40960/project-22200226/assets/114998243/cb2da4f9-88c0-458c-bde0-8d6dffbc0103)


For the real-time facial expression recognition, we employed the **Haar Cascade classifier**, a feature-based object detection algorithm, implemented through **OpenCV**. This approach allowed us to detect faces in live video streams from the laptop's webcam. Subsequently, the selected CNN model was applied to recognize facial expressions in real-time, providing instantaneous emotion detection.

### Analysis & Findings
The classes in the dataset show imbalance,  where 'Happy' is predominant and 'Disgust' is minority. 
<img width="734" alt="Class Distribution" src="https://github.com/ACM40960/project-22200226/assets/114998243/d14b09d5-e6cc-4934-a508-219b02799d34">

Below are sample images from the dataset in each class:\
<img width="497" alt="sample_images" src="https://github.com/ACM40960/project-22200226/assets/114998243/4da6df45-28cb-43bc-a9d5-57af433bcb86">


### CNN Build Model and Model Summary
> :rocket: **Alert!** Buckle up, because the training process for our model takes around *6.37 hours*! :hourglass_flowing_sand: (We use a 1.4 GHz Quad-Core Intel Core i5 processor)

Three blocks with 2 convolutional layers, BatchNormalization, MaxPooling, and Dropout (0.4-0.6), followed by a 128-unit FC layer and a softmax layer. Convolutional layers in each block use 64, 128, and 256 filters of size 3x3. MaxPooling layers have 2x2 kernels. Training involves a batch size of 32 for 100 epochs. RELU activation and HeNormal kernel initializer are used. Callbacks include Early Stopping & ReduceLRonPlateau. The optimizer is Nadam, and the loss function is Categorical Cross-Entropy. 

<img width="667" alt="cnn visualization" src="https://github.com/ACM40960/project-22200226/assets/114998243/33c4698a-be55-4e8f-ae1a-425f94f514d0">



<img width="283" alt="final_model" src="https://github.com/ACM40960/project-22200226/assets/114998243/218d2086-3230-49f5-8d00-f8f02e73a3ea">


### Model Evaluation
1. **Training vs Validation Loss & Accuracy** - Gradually improving, and it stops at epoch 52 because of the Early Stopping callback. By that point, the training and validation accuracy are around 70%, and the training and loss accuracy are about 0.9.


<img width="806" alt="final_curve" src="https://github.com/ACM40960/project-22200226/assets/114998243/b03538fc-1c4a-4c94-bbbb-672c5966b0ef">



2. **Normalized Confusion Matrix** - Model Evaluation on the Test Set\
Disgust images frequently predicted as Anger. Notably, Happy demonstrated exceptional classification performance, with 801 accurate predictions across all images, the highest among all emotion categories.


<img width="660" alt="final_confusion_matrix" src="https://github.com/ACM40960/project-22200226/assets/114998243/b328fd8a-9e48-4eff-97eb-c26db0652a4d">




3. **Classification Report** - Model Evaluation on the Test Set
   

| Classes       | Precision | Sensitivity (Recall) | Specificity | F1-Score | Accuracy |
| ------------- | --------- | -------------------- | ----------- | -------- | -------- |
| 0 - Angry     | 0.604     | 0.603                | 0.937       | 0.603    | 0.892    |
| 1 - Disgust   | 0.719     | 0.418                | 0.997       | 0.529    | 0.989    |
| 2 - Fear      | 0.563     | 0.405                | 0.946       | 0.471    | 0.866    |
| 3 - Happy     | 0.877     | 0.911                | 0.959       | 0.894    | 0.947    |
| 4 - Sad       | 0.529     | 0.598                | 0.894       | 0.561    | 0.845    |
| 5 - Surprise  | 0.778     | 0.776                | 0.971       | 0.777    | 0.948    |
| 6 - Neutral   | 0.635     | 0.698                | 0.915       | 0.665    | 0.877    |


> **Overall Accuracy = 68.24%**

4. **One-VS-Rest Multiclass ROC** - Model Evaluation on the Test Set

<img width="480" alt="final_roc_auc" src="https://github.com/ACM40960/project-22200226/assets/114998243/93b2d5d3-e616-4c85-9bd1-170b7d203fa3">

*ROC curve misleads due to highly imbalance dataset.*

5.  **One-VS-Rest Multiclass PR** - Model Evaluation on the Test Set
   
<img width="454" alt="final_pr_curve" src="https://github.com/ACM40960/project-22200226/assets/114998243/da40a072-174a-41e7-8f13-7c401e54fef5">


### Conclusion
The model's performance on the test set achieves an overall accuracy of approximately 68.24%. Given the highly imbalanced class, evaluating the model through metrics such as F1-score and Precision-Recall (PR) curve become more appropriate as they focus on the positive class. Notably, we can see that negative emotions such as "Angry", "Disgust", "Fear", and "Sad" have lower F1-score and PR curve value than positive or neutral emotions with "Fear" having the lowest score. Looking at the images in the dataset again, it's tough for even people to tell the difference between "Fear" and other emotions like "Anger" or being "Sad" or to distinguish between "Disgust" and "Anger". This is also true in real life – detecting the negative emotion is challenging. 

### Future Work
Exploring transfer learning methods with pre-trained models, facial landmark alignment, additional data augmentation, addressing class imbalance and expanding the dataset to include more varied examples could improve the model's classification capabilities. Additionally, to investigate human performance in emotion detection and compare it with the outcomes of the CNN model.

*Additional Information: This dataset was used for a Kaggle Challenge. The top-performing solution reached 71.16% accuracy, while our model achieved 68.24% accuracy, placing us in 4th position.*

## Acknowledgments
The facial emotion recognition algorithm was adapted from the following sources:
* [mayurmadnani](https://github.com/mayurmadnani/fer.git)
* [greatsharma](https://github.com/greatsharma/Facial_Emotion_Recognition.git)

## Authors
If you have any questions with this project, feel free to reach out to us at:

- [Clementine Surya - 22200226](https://github.com/ACM40960/project-22200226.git) - [clementine.surya@ucdconnect.ie](mailto:clementine.surya@ucdconnect.ie)

- [Liu Ye - 22200868](https://github.com/ACM40960/project-YeLiu.git) - [ye.liu1@ucdconnect.ie](mailto:ye.liu1@ucdconnect.ie)




