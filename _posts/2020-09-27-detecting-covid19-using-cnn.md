---
layout: post
title:  "Domain Expertise: What deep learning needs for better COVID-19 detection"
date:   2020-09-27 08:00:00 +0800
categories: DATA
tags: deep-learning supervised-learning classification convolutional-neural-network COVID-19
---

By now, you've probably seen a few, if not many, [articles](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=detecting+covid-19+using+neural+network&btnG=) on how deep learning could help detect COVID-19. In particular, convolutional neural networks (CNNs) have been studied as a faster and cheaper alternative to the gold-standard PCR test by just analyzing the patient's computed tomography (CT) scan. It's not surprising since CNN is excellent at image recognition; many places have CT scanners rather than COVID-19 testing kits (at least initially).

Despite its success in image recognition tasks such as the ImageNet challenge, can CNN really help doctors detect COVID-19? If it can, how accurately can it do so? It's well known that CT scans are sensitive but not [specific](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7227176/) to COVID-19. That is, COVID-19 almost always produces abnormal lung patterns visible from CT scans. However, other pneumonia can create the same abnormal patterns. Can the powerful and sometimes magical CNN tackle this ambiguity issue?

We had a chance to answer these questions ourselves (with my colleague [Yuchen](https://www.linkedin.com/in/yuchen-shi-2830ba158/?originalSubdomain=sg) and advisor [A/P Chen](https://www.eng.nus.edu.sg/isem/staff/chen-nan/)). I'll walk you through a COVID-19 classifier that we've built as an entry to 2020 INFORMS QSR Data [Challenge](https://connect.informs.org/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey=f404f7b8-fcd6-75d5-f7a7-d262eab132e7). If you are not familiar with CNN or would like a refresher on the key features of CNNs, I highly recommend reading [Convolutional Neural Network: How is it different from the other networks?](https://yangxiaozhou.github.io/data/2020/09/24/intro-to-cnn.html) first. Also, if you'd like to get hands-on, you can get all the code and data from my Github [repo](https://github.com/YangXiaozhou/CNN-COVID-19-classification-using-chest-CT-scan). 

## Key takeaways
1. Transfer learning using pre-trained CNN can achieve a really strong baseline performance on COVID-19 classification (85% accuracy).
2. However, domain expertise-informed feature engineering and adaptation are required to elevate the CNN (or other ML methods) to a medically convincing level. 

------------------------------------------------------------------------------
# What's the challenge?

COVID-19 pandemic has changed lives around the world. This is the current situation as of 2020/09/26, according to [WHO](https://covid19.who.int/). ![current situation]({{ '/' | relative_url }}assets/cnn-covid-19/covid-19-pandemic.png) 

CT scans have been used to screen and diagnose COVID-19, especially in areas where swab test resources are severely lacking. The goal of this data challenge is to diagnose COVID-19 using chest CT scans. Therefore, we need to build a **classification model** that can classify patients to COVID or NonCOVID based on their chest CT scans, **as accurately as possible**.

## What's provided?
Relatively even number of COVID and NonCOVID images are provided to train the model. While meta-information of these images is also provided, they will not be provided during testing. The competition also requires that the model's training with provided data must take less than one hour. 

- Training data set
    - 251 COVID-19 CT images
    - 292 non-COVID-19 CT images
- Meta-information
    - e.g., patient information, severity, image caption
    
All challenge data are taken from a public [data set](https://github.com/UCSD-AI4H/COVID-CT).

# Model performance
Let's first take a look at the result, shall we? 

The trained model is evaluated with an independent set of test data. Here you can see the confusion matrix. The overall accuracy is about 85% with slightly better sensitivity than specificity, i.e., true positive rate > true negative rate. 
![confusion]({{ '/' | relative_url }}assets/cnn-covid-19/confusion_matrix.png) 


# Implementation
Here are some of the provided NonCOVID and COVID CT scans. It's important to note that the challenge is to distinguish between COVID and NonCOVID CT scans, rather than COVID and Normal scans. In fact, there may be some NonCOVID CT scans that belong to other pneumonia patients. 
![first_look]({{ '/' | relative_url }}assets/cnn-covid-19/first_look.png) 

## Train-validation split
We reserve 20% of the data for validation. Since some consecutive images come from the same patient, they tend to be similar to each other.  That is, many of our data are **not independent**. To prevent data leakage (information of training data spills over to validation data), we keep the original image sequence and hold out the last 20% as the validation set. 

After the splitting, we have two pairs of data:
1. `X_train`, `y_train`
2. `X_val`, `y_val`

X is a list of CT scans, and y is a list of binary labels (0 for NonCOVID, 1 for COVID).

## Data augmentation
Data augmentation is a common way to include more random variations in the training data. It helps to prevent overfitting. For image-related learning problems, augmentation typically means applying **random** geometric (e.g., crop, flip, rotate, etc.) and appearance transformation (e.g., contrast, edge filter, Gaussian blur, etc.). Here we use `tf.keras.Sequential` to create a pipeline in which the input image is randomly transformed through the following operations:
1. Random horizontal and vertical flip
2. Rotation by a random degree in the range of $[-5\%, 5\%]*2\pi$
3. Random zoom in height by $5\%$
4. Random translation by $5\%$
5. Random contrast adjustment by $5\%$

This is how they look after the augmentation. 
![augmented_scans]({{ '/' | relative_url }}assets/cnn-covid-19/augmented_scans.png) 

## Using pre-trained CNN as the backbone
We do not build a CNN from scratch. For an image-related problem with only a modest number of training images, it is recommended to use a pre-trained model as the backbone and do [transfer learning](https://cs231n.github.io/transfer-learning/) on that. The chosen model is [EfficientNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0). It belongs to the family of models called [EfficientNets](https://arxiv.org/abs/1905.11946) proposed by researchers from Google. EfficientNets are among the current state-of-the-art CNNs for computer vision tasks. They
1. need a considerably lower number of parameters,
2. achieved very high accuracies on ImageNet,
3. transferred well to other image classification tasks. 

Here's a performance [comparison](https://arxiv.org/abs/1905.11946) between EfficientNets and other well-known models:![EfficientNet]({{ '/' | relative_url }}assets/cnn-covid-19/efficientnets.png)

EfficientNets, and other well-known pre-trained models, can be easily loaded from `tf.keras.applications`. We first import the pre-trained EfficientNetB0 and use it as our model backbone. We remove the original output layer of EfficientNetB0 since it was trained for 1000-class classification. Also, we freeze the model's weights so that they won't be updated during the initial training.
```
# Create a base model from the pre-trained EfficientNetB0
base_model = keras.applications.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False)
base_model.trainable = False
```

## Wrap our model around it

With EfficientNet imported, we can use it to our problem by wrapping our classification model around it. You can think of the EfficientNetB0 as a trained feature extractor. The final model has:
1. An input layer 
2. **EfficientNetB0 base model**
3. An average pooling layer: Pool the information by average operation
4. A dropout layer: Set a percentage of inputs to zero
5. A classification layer: Output the probability of NonCOVID

We can also use `tf.keras.utils.plot_model` to visualize our model. ![model]({{ '/' | relative_url }}assets/cnn-covid-19/model.png)

We can see that:
1. The `?` in input and output shape is a reserved place for the number of samples, which the model does not know yet.
2. EfficientNetB0 sits right after the input layer.
3. The last (classification) layer has an output of dimension 1: The probability for NonCOVID.

## Training our model
**Public data pre-training**: To help EfficientNetB0 adapt to COVID vs NonCOVID image classification, we've actually trained our model on another public CT scan data [set](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset). The hope is that training the model on CT scans would allow it to learn features specific to our COVID-19 classification task. We will not go into the public data training part, but the procedure is essentially the same as what I will show below. 

**Transfer learning workflow**: We use a typical transfer-learning workflow:
1. Phase 1 (Feature extraction): Fix EfficientNetB0's weights, only update the last classification layer's weights.
2. Phase 2 (Fine tuning): Allow some of EfficientNetB0' weights to update as well.

You can read more about the workflow [here](https://www.tensorflow.org/guide/keras/transfer_learning#the_typical_transfer-learning_workflow).

**Key configurations**: We use the following metrics and loss function:
1. **Metrics**: to evaluate model performance
    - Binary accuracy
    - False and true positives
    - False and true negatives
2. **Loss function**: to guide gradient search
    - Binary cross-entropy

We use the ```Adam``` optimizer, the learning rates is set to ```[1e-3, 1e-4]``` and the number of training epochs is set to ```[10, 30]``` for the two phases, respectively. The two-phase training is iterated for two times. 

**Training history**: Let's visualize the training history: ![training_history]({{ '/' | relative_url }}assets/cnn-covid-19/training_history.png)

Here you can see that after we've allowed some layers of EfficientNets to update (after Epoch 10), we obtain a significant improvement in classification accuracy. The final training and validation accuracy is around 98% and 82%. 

# How does it perform on test data?
We can obtain a set of test data from the same data repo that contains 105 NonCOVID and 98 COVID images. Let's see how the trained model performs on them. Here's a result breakdown using ```sklearn.metrics.classification_report```.
```
              precision    recall  f1-score   support

       COVID       0.85      0.83      0.84        98
    NonCOVID       0.84      0.87      0.85       105

    accuracy                           0.85       203
   macro avg       0.85      0.85      0.85       203
weighted avg       0.85      0.85      0.85       203
```
And the ROC curve: ![roc_curve]({{ '/' | relative_url }}assets/cnn-covid-19/roc_curve.png)

## What are correctly and incorrectly classified CT scans?

We can dive into the classification result and see which ones are identified correctly and identified incorrectly. **Potential patterns** found could be leveraged to help further improve the model.  

**Could you identify some patterns?**
![true_positives]({{ '/' | relative_url }}assets/cnn-covid-19/true_positives.png)
![true_negatives]({{ '/' | relative_url }}assets/cnn-covid-19/true_negatives.png)

And here are the incorrect ones:
![false_positives]({{ '/' | relative_url }}assets/cnn-covid-19/false_positives.png)
![false_negatives]({{ '/' | relative_url }}assets/cnn-covid-19/false_negatives.png)

We can probably make several observations here:
1. True positives have obvious abnormal patterns, and the lung structures are well-preserved.
2. Many of the true negatives have complete black lungs (no abnormal pattern).
3. The lung boundaries of many false positives are not clear. 

The point is, to a non-medical person like me, many of the COVID and NonCOVID images look the same. The ambiguity is even more severe when some images have unclear lung boundaries. It seems like our CNN is also having trouble distinguishing those images. 


# Where do we go from here?

From the above results, we can see that a pre-trained CNN can be adapted to achieve a really strong baseline performance. However, there are clear limitations to what a deep learning model (or any other model) alone can achieve. In this case, computer vision researchers and medical experts need to collaborate in a meaningful way so that the end model is both computationally capable and medically reliable. 

There are several directions for which we could make the model better:
1. **Lung segmentation**: Process each image and retain only the lung area of the CT scan, for example, see [here](https://pubs.rsna.org/doi/full/10.1148/rg.2015140232).
2. More sophisticated **transfer learning** design: For example, see multi-task [learning](https://ruder.io/multi-task/) or supervised domain [adaptation](https://en.wikipedia.org/wiki/Domain_adaptation#The_different_types_of_domain_adaptation). 
3. **Ensemble** model: This seems like a common belief, especially among Kaggle users, that building an ensemble [model](https://scikit-learn.org/stable/modules/ensemble.html) would almost always give you an extra few percent accuracy increases. 

*That's it for our CNN COVID-19 CT scan classification! Thank you!* 👏👏👏
