---
layout: post
title:  "INFORMS Data Challenge: Can we detect COVID-19 using deep learning?"
date:   2020-09-24 08:01:00 +0800
categories: DATA
tags: deep-learning supervised-learning classification
---

In this article, I'll walk you through the COVID-19 convolutional neural network (CNN) classifier that we've built as an entry to the 2020 INFORMS QSR Data [Challenge](https://connect.informs.org/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey=f404f7b8-fcd6-75d5-f7a7-d262eab132e7). This article focuses on the **implementation** of an end-to-end CNN image classifier using `tensorflow.keras`. If you are not familiar with CNN, or would like a refresher on the key features of CNNs, I highly recommend reading [Introduction to convolutional neural network](http://yangxiaozhou.github.io/data/2020-09-24-intro-to-cnn.html) first. 

* TOC
{:toc}

------------------------------------------------------------------------------

# COVID-19 classifier using CNN

The COVID-19 pandemic has changed lives around the world. This is the current situation as of 2020/09/24 according to [WHO](https://covid19.who.int/). 
![current situation]({{ '/' | relative_url }}assets/cnn-covid-19/covid-19-pandemic.png) 


## What's the challenge?

**Computed tomography (CT) scans** have been used for screening and diagnosing COVID-19, especially in areas where swab test resources are severely lacking. The goal of the data challenge is to diagnose COVID-19 using the chest CT scans.

Therefore, the challenge is to come up with a **classification model** that classify patients to COVID or NonCOVID based on their chest CT scans, **as accurately as possible**.

## What's provided?

- Training data set
    - 251 COVID-19 CT images
    - 292 non-COVID-19 CT images
- Meta-information
    - e.g., patient information, severity, image caption
    
All of data are taken from a [public data set](https://github.com/UCSD-AI4H/COVID-CT).

## Classifier via transfer learning
### Pre-processing
### Transfer learning

# Where do we go from here?

