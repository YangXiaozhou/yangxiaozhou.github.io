---
layout: post
mathjax: true
title:  "Set up my blog"
date:   2019-09-25 08:00:00 +0800
categories: cs
tags: 
---

What would be a better way to start this blog than writing a post about how I actually set it up? Because after three days of reading through all the docs/blogs/stackoverflows/github issues, I finally realised that the process is not as straightforward as I thought it would be. Luckily, I got it to work, for now. 

My aim is simple, to set up a blog for myself where I can post stuff about my life. The blog needs to be free, elegant, easy to navigate and support math. My current set up, Jekyll + Hyde + Github, matches with that. Since there are many resources online about setting up a blog using Jekyll and serve it with GitHub, I am going to skip all the standard procedures by refering to the official documents. Instead, this post specifically documents:
- the sequence of setting up different parts,
- adding support for Tags, Categories and their corresponding pages,
- adding MathJax to support $\LaTeX$-like math.

Finally, I am using a macbook, so the steps will be described assuming the system is macOS. When in doubt, just google the relevant steps for other OSs. 

## 1. Set up Jekyll 
Jekyll is the package that is generating all your webiste pages. First thing you want to do is to make sure that [Jekyll](https://jekyllrb.com) is installed and ready to run. 

Follow the official [instructions](https://jekyllrb.com/docs/). If you successfully made a new site, good! But if you ran into a **file permission error**, here's what you need to do:

1. Run the following lines to set GEM_HOME to your user directory.
    ```bash
    echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
    echo 'export GEM_HOME=$HOME/gems' >> ~/.bashrc
    echo 'export PATH=$HOME/gems/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
    ```
2. Now you can proceed with the following line and the rest of the steps.
    ```bash
    gem install bundler jekyll
    ```

The problem is caused by the macOS Mojave update. The above solution is provided in one of the GitHub [issues](https://github.com/jekyll/jekyll/issues/7274). Make sure Jekyll can run normally before proceeding to step 2. 

## 2. Set up GitHub repo
Jekyll generates web pages locally, we need a GitHub repository to host our pages so that they can be accessed on the internet. For this part, setup can be done by following GitHub's official [instructions](https://pages.github.com). In the end, you should have a repo on GitHub called *username*.github.io, and the corresponding local folder on your computer. In my case, the name of my repo is yangxiaozhou.github.io. 

By the end of Step 1 and 2, we have set up the local engine for generating web pages and the GitHub repo for hosting and publishing your pages. Now we proceed to the actual website construction.

## 3. Use a website template
[Hyde](http://hyde.getpoole.com) is a Jekyll website theme built on [Poole](https://github.com/poole/poole). They provide the template and the theme for the website. There are many themes for Jekyll, but I decided to use Hyde because I like the elegant design and it's easy to customize. 

To get Hyde, you just download [their repo](https://github.com/poole/hyde) and move all the files into the local folder that you have just created in Step 2. Remember to clear any existing files in that folder before moving in Hyde files. From here, you just have to edit parts of those files to make the website yours (or use it as it is). I changed the following two lines in `_config.yml` since redcarpet and pygments are not supported anymore. Other variables can also be changed such as name, github account and etc. 
```yaml
markdown: kramdown
highlighter:      rouge
```
At this point, it would be a good idea to learn some basics of [Jekyll](https://jekyllrb.com/docs/), e.g. what is a front matter, what is a page, how to creat a layout and etc. After learning these, you can go ahead and customize your website as you'd like. 

One problem that I run into is that pages look fine in local serve, but when I publish them to the web, pages other than the home page have suddenly lost all their style elements. After searching though internet, I realize that this has to do with the `url` and `baseurl` usage. 

Solution:
- change all the `site.baseurl`
instances in `head.html` and `sidebar.html` to `'/' | relative_url` so that the correct files can be located. 



## 4. Add tags & categories


## 5. Add MathJax

1. Create a `mathjax.html` file
2. Include it in `head.html`

#### Tips
- To use the normal dollar sign instead of the MathJax command (escape), put `<span class="tex2jax_ignore">...</span>` around the text you don't want MathJax to process.
- Check currently supported [Tex/LaTeX commands by MathJax](https://docs.mathjax.org/en/latest/input/tex/macros/index.html).


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



