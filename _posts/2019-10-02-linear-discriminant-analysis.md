---
layout: post
title:  Linear discriminant analysis (LDA)
date:   2019-10-02 08:00:00 +0800
categories: STATISTICS
tags: LDA supervised-learning classification
---

LDA is used as a tool for classification, dimension reduction, and data visualization. It has been around for quite some time now. Despite its simplicity, LDA often produces stable, effective, and interpretable classification results. Therefore, when tackling a classification problem, LDA is often the first and benchmarking method before other more complicated and flexible methods are employed. 

Two prominent examples of using LDA (and it's variants) include:
- *Bankruptcy prediction*: Edward Altman's [1968 model](https://en.wikipedia.org/wiki/Altman_Z-score) predicts the probability of company bankruptcy using trained LDA coefficients. The accuracy is said to be between 80% and 90%, evaluated over 31 years of data.
- *Facial recognition*: While features learnt from Principal Components Analysis (PCA) operations are called Eigenfaces, features learnt from LDA operations are called [Fisherfaces](http://www.scholarpedia.org/article/Fisherfaces), named after the great statistician, Sir Ronald Fisher. The connection will be explained later. 

This article starts with introducing the classic LDA and its reduced-rank version. Then we summarize the merits and disadvantages of LDA. The second article following this generalizes LDA to handle more complex problems. 

--------------------------------------------------------------------------
### Classification by discriminant analysis
Consider a generic classification problem: A random variable $X$ comes from one of $K$ classes, $G = 1, \dots, K$, with density $f_k(\mathbf{x})$ on $\mathbb{R}^p$. A discriminant rule divides the space into $K$ disjoint regions $\mathbb{R}_1, \dots, \mathbb{R}_K$. Classification by discriminant analysis simply means that we allocate $\mathbf{x }$ to $\Pi\_{j}$ if $\mathbf{x} \in \mathbb{R}_j$. We can follow two allocation rules:

- *Maximum likelihood rule*: If we assume that each class could occur with equal probability, then allocate $\mathbf{x }$ to $\Pi_{j}$ if $j = \arg\max_i f_i(\mathbf{x})$ .
- *Bayesian rule*: If we know the class prior probabilities, $\pi_1, \dots, \pi_K$, then allocate $\mathbf{x }$ to $\Pi_{j}$ if $j = \arg\max_i \pi_i f_i(\mathbf{x}) $.

### LDA & QDA
If we assume data comes from multivariate Gaussian distribution, i.e. $X \sim N(\mathbf{\mu}, \mathbf{\Sigma})$, explicit forms of the above allocation rules can be obtained. Following the Bayesian rule, we classify $\mathbf{x}$ to $\Pi_{j}$ if $j = \arg\max_i \delta_i(\mathbf{x})$ where 

$$
\begin{align}
    \delta_i(\mathbf{x}) = \log f_i(\mathbf{x}) + \log \pi_i
\end{align}
$$ 

is called the discriminant function. Note the use of log-likelihood here.  The decision boundary separating any two classes, $k$ and $\ell$, is the set of $\mathbf{x}$ where two discriminant functions have the same value, i.e. $$\{\mathbf{x}: \delta_k(\mathbf{x}) = \delta_{\ell}(\mathbf{x})\}$$. 

LDA arises in the case where we assume equal covariance among $K$ classes, i.e. $\mathbf{\Sigma}_1 = \mathbf{\Sigma}_2 = \dots = \mathbf{\Sigma}_K$. Then we can obtain the following discriminant function:

$$
\begin{align}
    \delta_{k}(\mathbf{x}) = \mathbf{x}^{T} \mathbf{\Sigma}^{-1} \mathbf{\mu}_{k}-\frac{1}{2} \mathbf{\mu}_{k}^{T} \mathbf{\Sigma}^{-1} \mathbf{\mu}_{k}+\log \pi_{k} \,.
    \label{eqn_lda}
\end{align}
$$

This is a linear function in $\mathbf{x}$. Thus, the decision boundary between any pair of classes is also a linear function in $\mathbf{x}$. This is the reason that this classification procedure is called linear discriminant analysis. Without the equal covariance assumption, the quadratic term in the likelihood does not cancel out, hence the resulting discriminant function is a quadractic function in $\mathbf{x}$:
$$
\begin{align}
    \delta_{k}(\mathbf{x}) = 
    - \frac{1}{2} \log|\mathbf{\Sigma}_k| 
    - \frac{1}{2} (\mathbf{x} - \mathbf{\mu}_{k})^{T} \mathbf{\Sigma}_k^{-1} (\mathbf{x} - \mathbf{\mu}_{k}) + \log \pi_{k} \,.
    \label{eqn_qda}
\end{align}
$$

Similarly, the decision boundary is quadratic in $\mathbf{x}$. This is known as quadratic discriminant analysis (QDA).

#### Number of parameters
In real problems, population parameters are usually unknown and estimated from training data as $\hat{\pi}_k, \hat{\mathbf{\mu}}_k, \hat{\mathbf{\Sigma}}_k$. While QDA accommodates more flexible decision boundaries compared to LDA, the number of parameters needed to be estimated also increase faster than that of LDA. From (\ref{eqn_lda}), $p+1$ parameters (nonlinear transformation of the original distribution parameters) are needed to construct the discriminant function. For a problem with $K$ classes, we would only need $K-1$ such discriminant functions by arbitrarily choosing one class to be the base class, i.e. 

$$
\delta_{k}'(\mathbf{x}) = \delta_{k}(\mathbf{x}) - \delta_{K}(\mathbf{x})\,,
$$

$k = 1, \dots, K-1$. Hence, the total number of estimated parameters for LDA is $$(K-1)(p+1)$$. On the other hand, for each QDA discriminant function (\ref{eqn_qda}), mean vector, covariance matrix, and class prior need to be estimated:
- Mean: $p$
- Covariance: $p(p+1)/2$
- Class prior: 1
 
Hence the total number of estimated parameters for QDA is $$(K-1)\{p(p+3)/2+1\}$$. *Therefore, the number of parameters estimated in LDA increases linearly with $p$ while that of QDA increases quadratically with $p$.* We would expect QDA to have worse performance than LDA when the dimension $p$ is large. 




### Reduced-rank LDA
### Fisher's LDA
### Summary of LDA

--------------------------------------------------------------------------
### Flexible discriminant analysis (FDA)
### Penalized discriminant analysis (PDA)
### Mixture discriminant analysis (MDA)

--------------------------------------------------------------------------
## Conlcusion

