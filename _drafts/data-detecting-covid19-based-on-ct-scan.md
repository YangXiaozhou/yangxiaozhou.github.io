---
layout: post
title:  "INFORMS Data Challenge: Can we detect COVID-19 using deep learning?"
date:   2020-09-25 08:01:00 +0800
categories: DATA
tags: deep-learning supervised-learning classification
---

In this article, I'll walk you through the COVID-19 convolutional neural network (CNN) classifier that we've built as an entry to the 2020 INFORMS QSR Data [Challenge](https://connect.informs.org/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey=f404f7b8-fcd6-75d5-f7a7-d262eab132e7). This article focuses on the **implementation** of an end-to-end CNN image classifier using `tensorflow.keras`. If you are not familiar with CNN, or would like a refresher on the key features of CNNs, I highly recommend reading [Introduction to convolutional neural network](http://yangxiaozhou.github.io/data/2020-09-24-intro-to-cnn.html) first. 

* TOC
{:toc}

------------------------------------------------------------------------------

The COVID-19 pandemic has changed lives around the world. This is the current situation as of 2020/09/24 according to [WHO](https://covid19.who.int/). 
![current situation]({{ '/' | relative_url }}assets/cnn-covid-19/covid-19-pandemic.png) 


# What's the challenge?

**Computed tomography (CT) scans** have been used for screening and diagnosing COVID-19, especially in areas where swab test resources are severely lacking. The goal of the data challenge is to diagnose COVID-19 using the chest CT scans.

Therefore, the challenge is to come up with a **classification model** that classify patients to COVID or NonCOVID based on their chest CT scans, **as accurately as possible**.

### What's provided?

- Training data set
    - 251 COVID-19 CT images
    - 292 non-COVID-19 CT images
- Meta-information
    - e.g., patient information, severity, image caption
    
All data are taken from a [public data set](https://github.com/UCSD-AI4H/COVID-CT).


# Proprocessing

Let's have a look at some NonCOVID and COVID CT scans. It's important to note that the challenge is to distinguish between COVID and NonCOVID CT scans, rather than COVID and Normal scans. In fact, it is obvious that the NonCOVID scans show different degree of abnormal patterns. This is because the similar patterns can develop in COVID-19 patients as well as other pneumonia patients. 
![first_look]({{ '/' | relative_url }}assets/cnn-covid-19/first_look.png) 

### Set up data for training
We reserve 20% of the data for validation. Since some consecutive images come from the same patient, they tend to be similar to each other.  That is, many of our data are **not independent**. To prevent data leakage (information of training data spills over to validation data), we keep the original image sequence and hold out the last 20% as the validation set. 

After the splitting, we have two pairs of data:
1. `X_train`, `y_train`
2. `X_val`, `y_val`

X is a list of CT scans, and y is a list of binary labels (0 for NonCOVID, 1 for COVID).

### Create `tf.data.Dataset` object

Data science workflow unique to `tensorflow` is the usage of `Dataset` object. The `tf.data.Dataset` API supports writing descriptive and efficient input pipelines. Essentially, it is a tensorflow data structure that greatly simplifies some essential operations, for example, preprocessing, shuffling and training. You can read more about it [here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle).

We create two `tf.data.Dataset` objects, one for training and the other for validation. We also define a wrapper `resize_and_shuffle` function where we 
1. create a `Dataset` object from a (`X, y`) pair,
2. resize each image to a standard size,
3. shuffle and split the data into batches for CNN training later.

# Classifier via transfer learning

# Where do we go from here?

