---
layout: post
title:  "Roadmap of statistical learning"
date:   2020-01-10 08:00:00 +0800
categories: DATA
tags: statistical_learning
---

A (work-in-progress) roadmap for statistical learning concepts and tools. 

**Regression**
1. Regularised Linear Regression
    - Ridge regression
    - Lasso regression
    - Logistic (ridge/lasso) regression

Next: however, linear relationship might be too restrictive in many cases, we wish to allow nonlinear relationship between response and predictors. 

2. Basis Expansion Method
    - Approximate mean function in a piecewise way
        - kth order spline for kth order piece, i.e. cubic spline means each segment is approximated by a cubic function formed by basis functions
        - Positive part remain operator (x - r)+ to ensure continuity at knots

3. Additive Model (semi-parametric method, building on top of basis expansion method)
    - More flexible than linear but retains the interpretability
        - Partially linear additive model: summation of linear variables and functions of variable
            - Nonlinear variables can be assumed to be represented by spline basis
            - What if a variable influences the relationship between Y and X?
                - Varying coefficient regression model: variable coefficients are functions of a variable(Z)
                - Such function can be assumed to have good estimation by spline method
    - Generalised: extend the method beyond linear link function, e.g. logistic
    - If we approximate each additive variable by piecewise linear model: multivariate adaptive regression spline (MARS)

4. Local Averaging Method
    - To estimate the regression surface, we can use local averaging method by defining two things
        1. What is “local”? i.e. how do we partition the space/find out the regression surface?
        2. How to compute “average”? i.e. simple average, weighted average?
    - Point 1 above
        - Binary recursive method: Regression and classification tree (CART)
        - Neighbourhood method: identifies a neighbour through some metric (kNN)
    - Point 2 above
        - Simple average: CART, kNN
        - Weighted average: weighted kNN, see point 7 below

But this kind of method produces non-smooth m(x) estimation since the weight used is indicator function.

5. Kernel Smoothing
    - Replace the indicator function by a kernel function (symmetric pdf) 
    - Nadaraya-Watson (NW) estimator: least square estimator weighted by kernel function
    - The value of h defines the size of the neighbourhood
    - Hence h determines the trade-off between model complexity and stability
    - But y need not be locally constant (regression tree, kNN, KS), we can assume y is locally linear or polynomial
        - LLKS
        - LPKS
6. Next: But the MSE of KS (nonparametric) methods increases with p (CoD). We can do dimension reduction, besides assuming a structure for m(x).
7. Dimension-reduction based method
    1. Ridge function = 1: Single-index model: one projection direction, in terms of unknown link function
        - More flexible than linear regression, but also more interpretable than PPR
    2. Ridge function > 1: Projection pursuit regression: one projection direction for each ridge function
        - Approximate target by non-linear function of the linear combination of input
8. Machine Learning
    - Without assuming the model of the data, learning a function that maps features (X) to predictions (Y), assume a space of the function (linear space, non-linear space), assume a convex loss/risk function (square error function).
    - Representer theorem provides the theoretical foundation for functions from RKHS to approximate the relationship by assuming a certain kernel, which is defined by the kind of kernel. 
    - Support Vector Machine: learning the location of the support vectors and the value of alpha for the kernel of support vectors, i.e. linear kernel, gaussian kernel, rbf kernel.
    - How is it different (performance, computation and etc.) for low and high dimensional case when we use non-linear kernel SVM?  
    - Classification
        - Fisher’s linear discriminant: linear combination of dimensions of X -> find a linear hyperplane that maximally separates the linear combination and minimises the variance.

**Classification**
1. Linear methods
    - Linear regression
    - Linear discriminant analysis
2. Non-linear methods
    - SVM
    - Discriminant analysis
        1. QDA ( assume unequal variance)
        2. Flexible discriminant analysis
        3. Mixture discriminant analysis

**Bayesian Inference**
1. Inference for dynamic systems/state-space models(SSMs)
    - Finite SSM: Baum-Petrie filter is the optimal filter with O(K^2) complexity.
    - Linear-Gaussian SSM: Kalman filter is the optimal filter, propagate mean and variance for inference.
    - Non-linear dynamics and/or non-Gaussian noise:
        1. Extended Kalman filter
        2. Unscented Kalman filter
        3. Sequential Monte Carlo (Particle filters) methods can be used since the distribution information is preserved beyond mean and covariance:
            1. Importance sampling to tackle difficult-to-sample posterior distribution problem
            2. Recursive formulation to tackle online inference complexity problem
            3. Resampling to ensure long-term stability of the particle method (mitigates sample impoverishment)
        4. Particle filtering
            1. 
        5. Particle smoothing

