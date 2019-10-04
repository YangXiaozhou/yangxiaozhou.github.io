---
layout: post
title:  Linear discriminant analysis - LDA
date:   2019-10-2 08:00:00 +0800
categories: STATISTICS
tags: LDA supervised-learning classification
---

Linear discriminant analysis (LDA) is used as a tool for classification, dimension reduction, and data visualization. It has been around for quite some time now. Despite its simplicity, LDA often produces stable, effective, and interpretable classification results. Therefore, when tackling a classification problem, LDA is often the first and benchmarking method before other more complicated and flexible methods are employed. 

Two prominent examples of using LDA (and it's variants) include:
- *Bankruptcy prediction*: Edward Altman's [1968 model](https://en.wikipedia.org/wiki/Altman_Z-score) predicts the probability of company bankruptcy using trained LDA coefficients. The accuracy is said to be between 80% and 90%, evaluated over 31 years of data.
- *Facial recognition*: While features learnt from Principal Components Analysis (PCA) operations are called Eigenfaces, features learnt from LDA operations are called [Fisherfaces](http://www.scholarpedia.org/article/Fisherfaces), named after the great statistician, Sir Ronald Fisher. The connection will be explained later. 

This article starts with introducing the classic LDA and its reduced-rank version. Then we summarize the merits and disadvantages of LDA. The second article following this generalizes LDA to handle more complex problems. 

### Classification by discriminant analysis
Consider a generic classification problem: A random variable $X$ comes from one of $K$ classes, $G = 1, \dots, K$, with density $f_k(\mathbf{x})$ on $\mathbb{R}^p$. A discriminant rule divides the space into $K$ disjoint regions $\mathbb{R}_1, \dots, \mathbb{R}_K$. Classification by discriminant analysis simply means that we allocate $\mathbf{x }$ to $\Pi\_{j}$ if $\mathbf{x} \in \mathbb{R}_j$. We can follow two allocation rules:

- *Maximum likelihood rule*: If we assume that each class could occur with equal probability, then allocate $\mathbf{x }$ to $\Pi_{j}$ if $j = \arg\max_i f_i(\mathbf{x})$ .
- *Bayesian rule*: If we know the class prior probabilities, $\pi_1, \dots, \pi_K$, then allocate $\mathbf{x }$ to $\Pi_{j}$ if $j = \arg\max_i \pi_i f_i(\mathbf{x}) $.

#### Linear and quadratic discriminant analysis
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
 
The total number of estimated parameters for QDA is $$(K-1)\{p(p+3)/2+1\}$$. *Therefore, the number of parameters estimated in LDA increases linearly with $p$ while that of QDA increases quadratically with $p$.* We would expect QDA to have worse performance than LDA when the dimension $p$ is large. 

#### Compromise between LDA & QDA
We can find a compromise between LDA and QDA by regularizing the individual class covariance matrices. That is, individual covariance matrix shrinks toward a common pooled covariance matrix through a penalty parameter $\alpha$:

$$
\hat{\mathbf{\Sigma}}_k (\alpha) = \alpha \hat{\mathbf{\Sigma}}_k + (1-\alpha) \hat{\mathbf{\Sigma}} \,.
$$

The pooled covariance matrix can also be regularized toward an identity matrix through a penalty parameter $\beta$:

$$
\hat{\mathbf{\Sigma}} (\beta) = \beta \hat{\mathbf{\Sigma}} + (1-\beta) \mathbf{I} \,.
$$

#### Computation for LDA
We can see from (\ref{eqn_lda}) and (\ref{eqn_qda}) that computations of discriminant functions can be simplified if we diagonalize the covariance matrices first. That is, data are transformed to have an identity covariance matrices. In the case of LDA, here's how we proceed witht the computation:

1. Perform eigen-decompostion on the pooled covariance matrix: 
$$
\hat{\mathbf{\Sigma}} = \mathbf{U}\mathbf{D}\mathbf{U}^{T} \,.
$$
2. Sphere the data:
$$
\mathbf{X}^{*} \leftarrow \mathbf{D}^{-\frac{1}{2}} \mathbf{U}^{T} \mathbf{X} \,.
$$
3. Obtain class centroids in the transformed space: $$\hat{\mu}_1, \dots, \hat{\mu}_{K}$$.
4. Classify $\mathbf{x}$ according to $\delta_{k}(\mathbf{x}^{*})$:

$$
\begin{align}
\delta_{k}(\mathbf{x}^{*})=\mathbf{x^{*}}^{T} \hat{\mu}_{k}-\frac{1}{2} \hat{\mu}_{k}^{T} \hat{\mu}_{k}+\log \hat{\pi}_{k} \,.
\label{eqn_lda_sphered}
\end{align}
$$

Step 2 spheres the data to produce an identity covariance matrix in the transformed space. Step 4 is obtained by following (\ref{eqn_lda}). Let's take a two class example to see what LDA is actually doing. Suppose there are two classes, $k$ and $\ell$. We classify $\mathbf{x}$ to class $k$ if $$\delta_{k}(\mathbf{x}^{*}) - \delta_{\ell}(\mathbf{x}^{*}) > 0$$. Following the four steps outlined above, we write

$$
\begin{align*}
\delta_{k}(\mathbf{x}^{*}) - \delta_{\ell}(\mathbf{x}^{*}) &= 
\mathbf{x^{*}}^{T} \hat{\mu}_{k}-\frac{1}{2} \hat{\mu}_{k}^{T} \hat{\mu}_{k}+\log \hat{\pi}_{k}
- \mathbf{x^{*}}^{T} \hat{\mu}_{\ell} + \frac{1}{2} \hat{\mu}_{\ell}^{T} \hat{\mu}_{\ell} - \log \hat{\pi}_{k} \\
&= \mathbf{x^{*}}^{T} (\hat{\mu}_{k} - \hat{\mu}_{\ell}) - \frac{1}{2} (\hat{\mu}_{k}^{T}\hat{\mu}_{k} - \hat{\mu}_{\ell}^{T} \hat{\mu}_{\ell}) + \log \hat{\pi}_{k}/\hat{\pi}_{\ell} \\
&= \mathbf{x^{*}}^{T} (\hat{\mu}_{k} - \hat{\mu}_{\ell}) - \frac{1}{2} (\hat{\mu}_{k} + \hat{\mu}_{\ell})^{T}(\hat{\mu}_{k} - \hat{\mu}_{\ell}) + \log \hat{\pi}_{k}/\hat{\pi}_{\ell} \\
&> 0 \,.
\end{align*}
$$

That is, we classify $\mathbf{x}$ to class $k$ if

$$
\mathbf{x^{*}}^{T} (\hat{\mu}_{k} - \hat{\mu}_{\ell}) > \frac{1}{2} (\hat{\mu}_{k} + \hat{\mu}_{\ell})^{T}(\hat{\mu}_{k} - \hat{\mu}_{\ell}) - \log \hat{\pi}_{k}/\hat{\pi}_{\ell} \,.
$$

The derived allocation rule reveals the working of LDA. The left-hand side of the equation is the length of orthorgonal projection of $$\mathbf{x^{*}}$$ onto the line segment joining the two class centroids. The right-hand side is the location of the centre of the segment corrected by class prior probabilities. *Essentially, LDA classifies the data to the closest class centroid.* Two observations can be made here.
1. The decision point deviates from the middle point when the class prior probabilities are not the same, i.e. the boundary is pushed toward the class with a smaller prior probability.
2. Data are projected onto the space spanned by class centroids, e.g. $$\hat{\mu}_{k} - \hat{\mu}_{\ell}$$. Distance comparisons are then done in that space. 


### Reduced-rank LDA
What I've just described is the idea of classification by discriminant analysis with certain distribution assumptions on the data. LDA is also popular for its ability to find a small number of meaningful dimensions, thus allowing us to visualize high-dimensional problems in a few dimensions. 

![lda_vs_pca](/assets/2019-10-02/lda_vs_pca.pdf)

### Fisher's LDA

### Summary of LDA

## Conlcusion

