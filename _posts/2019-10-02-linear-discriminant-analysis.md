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
- *Facial recognition*: While features learnt from Principal Component Analysis (PCA) operations are called Eigenfaces, features learnt from LDA operations are called [Fisherfaces](http://www.scholarpedia.org/article/Fisherfaces), named after the great statistician, Sir Ronald Fisher. The connection will be explained later. 

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

In situations where the number of input variables greatly exceed the number of samples, covariance matrix can be poorly estimated. Shrinkage can hopefully improve the estimation and classification accuracy.  
![lda_shrinkage]({{ '/' | relative_url }}assets/2019-10-02/lda_shrinkage.pdf)
<details>
<summary>Here's the script taken from <a href="https://scikit-learn.org/stable/auto_examples/classification/plot_lda.html">scikit-learn</a> to generate the above plot.</summary>
<div markdown="1">
``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
%matplotlib inline

n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


def generate_data(n_samples, n_features):
    """Generate random blob-ish data with noisy features.

    This returns an array of input data with shape `(n_samples, n_features)`
    and an array of `n_samples` target labels.

    Only one feature contains discriminative information, the other features
    contain only noise.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # add non-discriminative features
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y

acc_clf1, acc_clf2 = [], []
n_features_range = range(1, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2 = 0, 0
    for _ in range(n_averages):
        X, y = generate_data(n_train, n_features)

        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5).fit(X, y)
        clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)

        X, y = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)

    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)

features_samples_ratio = np.array(n_features_range) / n_train

with plt.style.context('seaborn-talk'):
    plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
             label="LDA with shrinkage", color='navy')
    plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
             label="LDA", color='gold')

    plt.xlabel('n_features / n_samples')
    plt.ylabel('Classification accuracy')
    plt.legend(prop={'size': 18})
    plt.tight_layout()
```
</div>
</details>

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
What I've just described is classification by LDA. LDA is also popular for its ability to find a small number of meaningful dimensions, allowing us to visualize high-dimensional problems. What do we mean by meaningful and how does LDA find these dimensions? We will anwser these questions shortly. First, take a look at the below plot. For a [wine classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine) problem with 3 different types of wines and 13 input variables, the plot visualizes the data in two discriminant coordinates found by LDA. In this 2-dimensional space, the classes can be well-separated. In comparison, the classes are not as clearly separated using the first 2 principal components found by PCA. 

![lda_vs_pca]({{ '/' | relative_url }}assets/2019-10-02/lda_vs_pca.pdf)
<details>
<summary>Here's the script to generate the above plot.</summary>
<div markdown="1">
``` python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
%matplotlib inline

wine = datasets.load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

X_r_lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y).transform(X)
X_r_pca = PCA(n_components=2).fit(X).transform(X)

with plt.style.context('seaborn-talk'):
    fig, axes = plt.subplots(1,2,figsize=[15,6])
    colors = ['navy', 'turquoise', 'darkorange']
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        axes[0].scatter(X_r_lda[y == i, 0], X_r_lda[y == i, 1], alpha=.8, label=target_name, color=color)
        axes[1].scatter(X_r_pca[y == i, 0], X_r_pca[y == i, 1], alpha=.8, label=target_name, color=color)
    axes[0].title.set_text('LDA for Wine dataset')
    axes[1].title.set_text('PCA for Wine dataset')
    axes[0].set_xlabel('Discriminant Coordinate 1')
    axes[0].set_ylabel('Discriminant Coordinate 2')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
```
</div>
</details>


#### Inherent dimension reduction
In the above wine example, a 13-dimensional problem is visualized in a 2d space. Why is this possible? This is possible because there's inherent dimension reduction in LDA. We have observed from the previous section that LDA makes distance comparison in the space spanned by different class centroids. Two distinct points lie on a 1d line; three distinct points lie on a 2d plane. Similarly, $K$ class centroids lie on a hyperplane with dimension at most $(K-1)$. In particular, the subspace spanned by the centroids is

$$
H_{K-1}=\mu_{1} \oplus \operatorname{span}\left\{\mu_{i}-\mu_{1}, 2 \leq i \leq K\right\} \,.
$$ 

When making distance comparisons, distances orthorgonal to this subspace would add no information since they contribute equally for each class. Hence, by restricting distance comparisons to this subspace only would not lose any information useful for LDA classification. That means, we can safely transform our task from a $p$-dimensional problem to a $(K-1)$-dimensional problem by an orthogonal projection of the data onto this subspace. When $p \gg K$, this is a considerable drop in the number of dimensions. What if we want to reduce the dimension further from $p$ tp $L$ where $K \gg L$? We can construct an $L$-dimensional subspace, $H_L$, from $H_{K-1}$ and this subspace is optimal, in some sense, to LDA classification. 

#### Optimal subspace and computation
Fisher proposes that the subspace $H_L$ is optimal when the class centroids of sphered data have maximum separation in this subspace in terms of variance. Following this defition, optimal subspace coordinates are simply found by doing PCA on sphered class centroids. The computation steps are summarized below:
1. Find class centroid matrix, $\mathbf{M}_{(K\times p)}$, and pooled var-cov, $$\mathbf{W}_{(p\times p)}$$, where

    $$
    \mathbf{W} = \sum_{k=1}^{K} \sum_{g_i = k} (\mathbf{x}_i - \hat{\mu}_k)(\mathbf{x}_i - \hat{\mu}_k)^T \,.
    $$

2. Sphere the centroids: $\mathbf{M}^* = \mathbf{M} \mathbf{W}^{-\frac{1}{2}}$, using eigen-decomposition of $\mathbf{W}$.
3. Compute $$\mathbf{B}^* = \operatorname{cov}(\mathbf{M}^*)$$, the between-class covariance of sphered class centroids by

    $$
    \mathbf{B}^* = \sum_{k=1}^{K} (\hat{\mathbf{\mu}}^*_k - \hat{\mathbf{\mu}}^*)(\hat{\mathbf{\mu}}^*_k - \hat{\mathbf{\mu}}^*)^T \,.
    $$

4. Obtain $L$ eigenvectors $$(\mathbf{v}^*_\ell)$$ in $$\mathbf{V}^*$$ of 
$$\mathbf{B}^* = \mathbf{V}^* \mathbf{D_B} \mathbf{V^*}^T$$ cooresponding to the $L$ largest eigenvalues. These define the coordinates of the optimal subspace.
5. Obtain $L$ new (discriminant) variables $Z_\ell = (\mathbf{W}^{-\frac{1}{2}} \mathbf{v}^*_\ell)^T X$, for $\ell = 1, \dots, L$.

Through this procedure, we reduce our data dimension from $$\mathbf{X}_{(N \times p)}$$ to $$\mathbf{Z}_{(N \times L)}$$. Discriminant coordinate 1 and 2 in the previous wine plot are found by setting $L = 2$. Repeating LDA procedures for classification using the new data $\mathbf{Z}$ is called the reduced-rank LDA. 

### Fisher's LDA
Fisher derived the computation steps according to his optimality definition in a different way[^3]. His steps of performing the reduced-rank LDA would later be known as the Fisher's LDA.

### Summary of LDA

## Conlcusion


#### Footnotes
[^3]: Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of eugenics, 7(2), 179-188.
