---
layout: post
title:  "Statistical learning knowledge repo: Foundation"
date:   2020-09-11 08:00:00 +0800
categories: DATA
tags: statistical-learning machine-learning data-science study-notes
---

This is a collection of my study notes on various foundational topics of statistical learning. It is intended as a knowledge repository for myself and a work in progress that I will periodically update. 

# Principal Component Analysis



# Ridge regularization


#### On the relationship between OLS, Ridge regression, and PCA
Simple yet elegant relaionships between OLS estimates, ridge estimates and PCA can be found through the lens of spectral decomposition. We see these relationships through Exercise 8.8.1 of MA[^MA]:

**Set-up** Given the following regression model:

$$
\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\mu \mathbf{1}+\mathbf{u}, \quad \mathbf{u} \sim N_{\mathrm{n}}\left(\mathbf{0}, \sigma^{2} \mathbf{I}\right),
$$

consider the columns of $\mathbf{X}$ have been standardized to have mean 0 and variance 1. Then the ridge estimate of $\mathbf{\beta}$ is $\mathbf{\beta}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$ where for given $\mathbf{X}$, $\lambda \ge 0$ is a small fixed ridge regularization parameter. Note that when $\lambda = 0$, it is just the OLS estimate. 

Also consider the spectral decomposition of the var-cov matrix $\mathbf{X}^T \mathbf{X} = \mathbf{G} \mathbf{L} \mathbf{G}^T$. Let $\mathbf{W} = \mathbf{X}\mathbf{G}$ be the principal component transformation of the original data matrix. 





# References
[^MA]: Mardia, K. V., Kent, J. T., & Bibby, J. M. *Multivariate Analysis*. 1979. Probability and mathematical statistics. Academic Press Inc.


