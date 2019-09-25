---
layout: post
mathjax: true
title:  "Setting up my blog"
date:   2019-09-25 08:00:00 +0800
categories: programming
tags: 
---

What better way to start this blog than writing a post about how I actually set it up? The process is not as straightforward as I thought it would be.

### 1. Set up Jekyll 


### 2. Host it on GitHub


### 3. Tags & Categories


### 4. MathJax

1. Create a `mathjax.html` file
2. Include it in `head.html`

#### Tips
- To use the normal dollar sign instead of the MathJax command (escape), put `<span class="tex2jax_ignore">...</span>` around the text you don't want MathJax to process.


#### Math Rendering Showcase
Inline math using `\$...\$`: $\mathbf{x}+\mathbf{y}$.

Displayed math using `\$\$...\$\$` on a new paragrah: 

$$
\mathbf{x}+\mathbf{y}
$$

Automatic numbering and referencing using <span class="tex2jax_ignore">`\ref{label}`</span>:
In (\ref{eq:sample}), we find the value of an interesting integral:
\begin{align}
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
  \label{eq:sample}
\end{align}

Multiline equations using `\begin{align*}`:
\begin{align\*}
  \nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \newline
  \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho
\end{align\*}






