---
layout: post
title:  "Expectation-maximization algorithm, explained"
date:   2020-10-01 08:00:00 +0800
categories: DATA
tags: expectation-maximization statistical-learning clustering inference
---

the Expectation-maximization algorithm (EM, for short)

**Write about the problem that the readers care**

**Create instability and offer solution**

If you are in the data science/ML "bubble", you probably have came across EM at some time and wondered: What is EM and why do I need to know it? 

My take on EM, what it is, how it works, how it's related to other techniques, and how it might be improved. 

1. Motivating examples
2. What is EM?
3. EM in action: Does it really work?
4. Intuition and theory: Why does it work?
5. So...is it perfect?
6. Uncertainty: Going beyond point estimate😱

## Motivating examples: Why do we care?

Maybe you already know why you want to use EM, or maybe you don't. Either way, let me use two motivating examples to set the stage for EM. These are quite lengthy, I know, but they perfectly highlight the common feature of the problems that EM is best at solving: the presence of **missing information**. 

### Unsupervised learning: Solving Gaussian mixture model for clustering

Suppose you have a data set with n number of data points. It could be a group of customers visiting your website (customer profiling) or an image with different objects in it (image segmentation). Clustering is the task of finding out k number of natural groups for your data when you don't know (or don't specify) the real grouping. This is an unsupervised learning problem because no ground-truth labels are used. 

Such clustering problem can be tackled by several types of algorithms, e.g., combinatorial type such as k-means or hierarchical type such as Ward’s hierarchical clustering. However, if you believe that your data could be better modeled as a mixture of normal distributions, then you would go for Gaussian mixture model (**GMM**), another popular clustering approach. 

The underlying idea of GMM is this, you assume that behind your data, there's a data generating mechanism. This mechanism first choses one of the k normal distributions (with a certain probability) and then delivers a sample from that distribution. Therefore, once you have estimated the parameters of each normal distribution, you could easily cluster each data point by selecting the one that gives the highest likelihood. 

<p>
  <img width="1024" alt="ClusterAnalysis Mouse" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/ClusterAnalysis_Mouse.svg/1024px-ClusterAnalysis_Mouse.svg.png">
</p>
<i>An example of mixture of Gaussian data and clustering using k-means and GMM (solved by EM). [Source](https://commons.wikimedia.org/wiki/File:ClusterAnalysis_Mouse.svg)</i>

However, estimating the parameters is not a simple task since we do not know which distribution generated which points (**missing information**). EM is an algorithm that can help us solve exactly this problem. This is why EM is the underlying algorithm for solving GMMs in scikit-learn's [implementation](https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture). 

### Population genetics: Estimating moth allele frequencies to observe natural selection

Have you heard the phrase "industrial melanism" before? It's a term coined by biologists in the 19th century to describe the phenomenon that animals change their skin color due to the heavy industrialization in the cities. In particular, they observed that previously rare dark peppered moth started to dominate the population in industrialized coal-fueled cities. Scientists at the time were surprised and fascinated by this observation. Subsequent research suggests that the industrialized cities tend to have darker tree barks which disguise darker moths better than the light ones. 

<p align="center">
  <img src="{{'/'|relative_url}}assets/intro-to-EM/dark_light_moth.png" alt="pepper_moths" style="zoom: 75%;">
</p><i>Dark (top) and light (bottom) peppered moth. Image by Jerzy Strzelecki via Wikimedia Commons</i>


As a result, dark moths survive the predation better and pass on their genes, giving rise to a predominantly dark peppered moth population.  To prove their natural selection theory, scientists first need to estimate the percentage of black-producing and light-producing genes/alleles present in the moth population. The gene responsible for the moth's color has three types of alleles: C, I and T. Genotypes **C**C, **C**I, and **C**T produce dark peppered moth (*Carbonaria*); **T**T produces light peppered moth (*Typica*); **I**I and **I**T produce moths with intermediate color (*Insularia*). 

Here's a hand-drawn graph that shows the **observed** and **missing** information. 

<p align="center">
  <img src="{{'/'|relative_url}}assets/intro-to-EM/moth_relationship.jpg" alt="moth_relationship" style="zoom: 100%;">
</p><i>Relationship between peppered moth alleles, genotypes, and phenotypes. We observed phenotypes, but wish to estimate percentges of alleles in the population. Image by author</i>


We wish to know the percentages of C, I, and T in the population. However, we can only observe the number of *Carbonaria*, *Typica*, and *Insularia* moths by capturing them, but not the genotypes (**missing information**). The fact that we do not observe the genotypes and multiple genotypes produce the same subspecies make the calculation of the allele frequencies difficult. This is where EM comes in to play. With EM, we can easily estimate the allele frequencies and provide concrete evidence for the microevolution that happens on a human time scale due to environmental  pollution. 

How does EM tackle the GMM problem and the peppered moth problem in the presence of missing information? We will illustrate these in the later section. But first, let's see what EM is really about. 

## What is EM?

At this point, you must be thinking (I hope): All these examples are wonderful, but what is really EM? Let's dive into it. 

EM algorithm is an iterative optimization method that finds the maximum likelihood estimate (MLE) of parameters in problems where hidden/missing/latent variables are present. It was first introduced in its full generality by Dempster, Laird, and Rubin in their famous paper[^Dempster] (currently 62k citations). Since then, it has been widely used for its easy implementation, numerical stability, and strong empirical performance.

Let's set up the EM for a general problem and introduce some notations. Suppose that $Y$ are our observed variables, $X$ are hidden variables, and we say that $(X, Y)$ is the complete data. We also denote any unknown parameter of interest as $\theta \in \Theta$. The objective of most parameter estimation problems is to find the most probable $\theta$ given our model and data, i.e.,

$$
\begin{equation}
\theta = \arg\max_{\theta \in \Theta} p_\theta(\mathbf{y}) \,,
\end{equation}
$$

where  $p_\theta(\mathbf{y})$ is the incomplete-data likelihood. Using the law of total probability, we can also express the incomplete-data likelihood as


$$
p_\theta(\mathbf{y}) = \int p_\theta(\mathbf{x}, \mathbf{y}) d\mathbf{x} \,,
$$


where $p_\theta(\mathbf{x}, \mathbf{y})$ is known as the complete-data likelihood. 

What's with all these complete- and incomplete-data likelihoods? In many problems, the maximization of the incomplete-data likelihood $p_\theta(\mathbf{y})$ is difficult because of the missing information. On the other hand, it’s often easier to work with complete-data likelihood. EM algorithm is designed to take advantage of this obsercarion. It iterates between an expectation step (E-step) and a maximization step (M-step) to find the MLEs. Assuming $\theta^{(n)}$ is the estimate obtained at the $n$th iteration, the algorithm proceeds as follows:

- **E-step**: define 
  $Q(\theta | \theta^{(n)})$ as the conditional expectation of the complete-data log-likelihood w.r.t. the hidden variables, given observed data and current parameter estimate, i.e.,
  $$
  \begin{align}
  Q(\theta | \theta^{(n)}) = \mathbb{E}_{X|\mathbf{y}, \theta^{(n)}}\left[\ln p_\theta(\mathbf{x}, \mathbf{y})\right] \,.
  \end{align}
  $$

- **M-step**: find a new $\theta$ that maximizes the above expectation and set it to $\theta^{(n+1)}$, i.e.,
  $$
  \begin{align}
  \theta^{(n+1)} = \arg\max_{\theta \in \Theta} Q(\theta | \theta^{(n)}) \,.
  \end{align}
  $$
  

The above definitions might seem hard-to-grasp at first. Some intuitive explanation might help:

- **E-step**: This step is asking, given our observed data $\mathbf{y}$ and current parameter estimate $\theta^{(n)}$, what are the probabilities of different $X$? Also, under these probable $X$, what are the corresponding log-likelihoods? 
- **M-step**: Here we ask, under these probable $X$, what is the value of $\theta$ that gives us the maximum expected log-likelihood?

The algorithm iterates between the two steps until a stopping criterion is reached, e.g., when either the Q function or the parameter estimate converged. 

## Where is EM in the big picture?

Connections of EM to other techniques

1. Clustering: k-means
2. Hidden Markov model inference: Baum-Welch algorithm
3. Bayesian inference: Gibbs sampling

## EM in action: Does it really work?

#### Solving GMM for clustering

#### Estimating allele frequencies

Peppered moth [game](https://askabiologist.asu.edu/peppered-moths-game/play.html)

## Why does it work?

#### Intuitive explanation

<details>
    <summary><b>Mathematical proof:</b></summary>
  Here we show why the above iterative scheme can find the maximum likelihood estimate of the parameter. Let $\ell(\theta) = \ln p_\theta(\mathbf{y})$, thus we have


$$
\ell(\theta) - \ell(\theta^{(n)}) = \ln p_\theta(\mathbf{y}) - \ln p_{\theta^{(n)}}(\mathbf{y}) \,.
$$


We wish to compute an updated $\theta$ such that the above relationship holds above zero. Using $p_\theta(\mathbf{y}) = \int p_\theta(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\mathbf{x}$, we have


$$
\begin{align*} \ell(\theta) - \ell(\theta^{(n)}) &= \ln \int p_\theta(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\mathbf{x} - \ln p_{\theta^{(n)}}(\mathbf{y}) \\
&= \ln \int p_\theta(\mathbf{x}, \mathbf{y}) \frac{p_{\theta^{(n)}}(\mathbf{x} | \mathbf{y})}{p_{\theta^{(n)}}(\mathbf{x} | \mathbf{y})} \, \mathrm{d}\mathbf{x} 

- \ln p_{\theta^{(n)}}(\mathbf{y}) \\
  &= \ln \mathbb{E}_{\mathbf{X} | \mathbf{y} , \theta^{(n)}}\left[\frac{p_\theta(\mathbf{x}, \mathbf{y})}{p_{\theta^{(n)}}(\mathbf{x} | \mathbf{y})}\right] - \ln p_{\theta^{(n)}}(\mathbf{y}) \\
  &\ge \mathbb{E}_{\mathbf{X} | \mathbf{y} , \theta^{(n)}}\left[\ln \frac{p_\theta(\mathbf{x}, \mathbf{y})}{p_{\theta^{(n)}}(\mathbf{x} | \mathbf{y})}\right] - \ln p_{\theta^{(n)}}(\mathbf{y}) \\
  &= \mathbb{E}_{\mathbf{X} | \mathbf{y} , \theta^{(n)}}\left[\ln \frac{p_\theta(\mathbf{x}, \mathbf{y})}{p_{\theta^{(n)}}(\mathbf{x} | \mathbf{y} ) p_{\theta^{(n)}}(\mathbf{y})}\right] \\
  &:= \Delta(\theta | \theta^{(n)}) \,.
  \end{align*}
$$


The inequality step follows by Jensen's inequality and the fact that $\ln(\cdot)$ is concave on $[0, \infty]$. The second last step follows since $p_{\theta^{(n)}}(\mathbf{y})$ does not depend on $\mathbf{X}$. Therefore, we have 
$$
\ell(\theta) \ge \ell(\theta^{(n)}) + \Delta(\theta|\theta^{(n)}) \,.
$$
Define 


$$
\ell(\theta | \theta^{(n)}) := \ell(\theta^{(n)}) + \Delta(\theta|\theta^{(n)}) \,,
$$
 

then 
$\ell(\theta) \ge \ell(\theta|\theta^{(n)})$. 
That is, 
$\ell(\theta|\theta^{(n)})$ 
is upper-bounded by $\ell(\theta)$ for all $\theta \in \Theta$. The equality holds when $\theta = \theta^{(n)}$ since


$$
\begin{align*}
\ell(\theta^{(n)}|\theta^{(n)}) &= \ell(\theta^{(n)}) + \Delta(\theta^{(n)}|\theta^{(n)}) \\
&= \ell(\theta^{(n)}) + \mathbb{E}_{\mathbf{X} | \mathbf{y} , \theta^{(n)}}\left[\ln \frac{p_{\theta^{(n)}}(\mathbf{x}, \mathbf{y})}{p_{\theta^{(n)}}(\mathbf{x} | \mathbf{y} ) p_{\theta^{(n)}}(\mathbf{y})}\right] \\
&= \ell(\theta^{(n)}) + \mathbb{E}_{\mathbf{X} | \mathbf{y} , \theta^{(n)}}\left[\ln \frac{p_{\theta^{(n)}}(\mathbf{x}, \mathbf{y})}{p_{\theta^{(n)}}(\mathbf{x}, \mathbf{y})}\right] \\
&= \ell(\theta^{(n)}) \,.
\end{align*}
$$


Therefore, when computing an updated $\theta$, any increase in 
$\ell(\theta|\theta^{(n)})$ leads to an increase in $\ell(\theta)$ by at least 
$\Delta(\theta|\theta^{(n)})$. The observation is that, by selecting the $\theta$ that maximizes 
$\Delta(\theta|\theta^{(n)})$, we can achieve the largest increase in $\ell(\theta)$. Formally, we have 


$$
\begin{align*}
\theta^{(n+1)} &= \arg\max_{\theta\in\Theta} \ell(\theta | \theta^{(n)}) \\
& = \arg\max_{\theta\in\Theta} 
\left\lbrace
\ell(\theta^{(n)}) + \mathbb{E}_{\mathbf{X} | \mathbf{y} , \theta^{(n)}} 
\left[
\ln \frac{p_\theta(\mathbf{x}, \mathbf{y})}{p_{\theta^{(n)}}(\mathbf{x} | \mathbf{y}) p_{\theta^{(n)}}(\mathbf{y})}
\right]
\right\rbrace\\
& = \underbrace{\arg\max_{\theta\in\Theta}}_{\text{Maximization}} \underbrace{\mathbb{E}_{\mathbf{X} | \mathbf{y} , \theta^{(n)}}}_{\text{Expectation}}[\ln p_\theta(\mathbf{x}, \mathbf{y})] \\
& = \arg\max_{\theta\in\Theta} Q(\theta | \theta^{(n)}) \,,
\end{align*}
$$


where the second last step follows by dropping terms constant with respect to $\theta$. Thus, the E-step and M-step are made apparent in the formulation. Also, by maximizing 
  $\ell(\theta | \theta^{(n)})$ instead of $\ell(\theta)$, we have made use of the information of hidden variables $\mathbf{X}$ in the complete-data likelihood. 

</details>

## So...is it perfect?

#### Improving E-step
#### Improving M-step

## Uncertainty: Going beyond point estimate 😱



----------------
## References
[^Dempster]: Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, *39*(1), 1-22.