---
layout: post
mathjax: true
title:  "How to set up my blog using Jekyll, Hyde and GitHub"
date:   2019-09-25 08:00:00 +0800
categories: cs
tags: 
---

What would be a better way to start this blog than writing a post about how I set it up? Because after three days of sifting through all the documents, blogs, StackOverflow answers and GitHub issues, I finally realized that the process is not as straightforward as I thought it would be. Anyway, I got it to work (for now). 

My aim is simple, to set up a blog for myself where I can post stuff about my life. The blog needs to be free, elegant, intuitive and support math. My current set up, Jekyll + Hyde + Github + MathJax, matches with that. Since there are many resources online about setting up a blog using Jekyll and serve it with GitHub, I am going to skip all the standard procedures by referring to the official documents. Instead, this post specifically documents:
- the sequence of setting up different parts,
- adding support for Tags, Categories and their corresponding pages,
- adding MathJax to support $\LaTeX$-like math.

I am using a macbook, so the steps will be described assuming the system is macOS. When in doubt, just google the relevant steps for other OSs. 

## 1. Set up Jekyll 
Jekyll is the package that is generating all your website pages. First thing you want to do is to make sure that [Jekyll](https://jekyllrb.com) is installed and ready to run. 

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

To get Hyde, just download [their repo](https://github.com/poole/hyde) and move all the files into the local folder that you have just created in Step 2. Remember to clear any existing files in that folder before moving in Hyde files. From here, you just have to edit parts of those files to make the website yours (or use it as it is). I changed the following two lines in `_config.yml` since redcarpet and pygments are not supported anymore. Other variables can also be changed such as name, GitHub account, etc. 
```yaml
markdown: kramdown
highlighter: rouge
```
At this point, it would be a good idea to learn some basics of [Jekyll](https://jekyllrb.com/docs/), e.g. what is a front matter, what is a page, how to create a layout, etc. After learning these, you can go ahead and customize the website as you'd like. 

One problem that I run into is that pages look fine in local serve, but when I publish them to the web, all pages other than the home page have suddenly lost all their style elements. After searching through the internet, I realize that this has to do with the `url` and `baseurl` usage. If you also have this problem, consider doing the following:
- change all the {% raw  %}`{{ site.baseurl }}`{% endraw  %}
instances in `head.html` and `sidebar.html` to {% raw  %}`{{ '/' | relative_url }}`{% endraw  %} so that the correct files can be located. 


## 4. Add tags & categories
I want to add tags and categories to my posts and also a dedicated page where posts can be arranged according to [tags](https://yangxiaozhou.github.io/tag/supervised-learning)/[categories](https://yangxiaozhou.github.io/categories/). This should be easy since tags and categories are default front matter variables that you can define in Jekyll. For example, tags and categories of my LDA post are defined like this:
```yaml
---
layout: post
mathjax: true
title:  "Linear discriminant analysis (LDA) and beyond"
date:   2019-09-26 08:00:00 +0800
categories: statistics
tags: supervised-learning classification
---
```

For **categories**, I created one page where posts of different categories are collected and is accessible through the sidebar link. To do this, just create a `category.html` in the root folder:
{% raw  %}
```html
---
layout: page
permalink: /categories/
title: Categories
---

<div id="archives">
{% for category in site.categories %}
  <div class="archive-group">
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name | slugize }}"></div>
    <p></p>

    <h3 class="category-head">{{ category_name }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% for post in site.categories[category_name] %}
    <article class="archive-item">
      <h4><a href="{{ site.baseurl }}{{ post.url }}">{{post.title}}</a></h4>
    </article>
    {% endfor %}
  </div>
{% endfor %}
</div>
```
{% endraw  %}

For **tags**, I did two things:
1. Show the tags of a post at the end of the content.
2. For every tag, create a page where posts are collected, i.e. [classification](https://yangxiaozhou.github.io/tag/classification), [supervised-learning](https://yangxiaozhou.github.io/tag/supervised-learning). 

To do 1, include the following lines after the `content` section in your `post.html`:
{% raw %}
```html
<span class="post-tags">
    {% for tag in page.tags %}
      {% capture tag_name %}{{ tag }}{% endcapture %}
      <a class="no-underline" href="/tag/{{ tag_name }}"><code class="highligher-rouge"><nobr>{{ tag_name }}</nobr></code>&nbsp;</a>    
    {% endfor %}
</span>
```
{% endraw %}
To do 2, Long Qian has written a very clear [post](https://longqian.me/2017/02/09/github-jekyll-tag/) about it, please refer to that.

## 5. Add MathJax
The last piece to my website is to add the support of $\LaTeX$-like math. This is done through MathJax. There are two steps to achieve it:

1. Create a `mathjax.html` file and put it in `_includes/`:
    {% raw %}
    ```html
    <script type="text/javascript" async
      src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
      MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        displayMath: [['$$','$$']],
        processEscapes: true,
        processEnvironments: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        TeX: { equationNumbers: { autoNumber: "AMS" },
             extensions: ["AMSmath.js", "AMSsymbols.js"] }
      }
      });
      MathJax.Hub.Queue(function() {
        // Fix <code> tags after MathJax finishes running. This is a
        // hack to overcome a shortcoming of Markdown. Discussion at
        // https://github.com/mojombo/jekyll/issues/199
        var all = MathJax.Hub.getAllJax(), i;
        for(i = 0; i < all.length; i += 1) {
            all[i].SourceElement().parentNode.className += ' has-jax';
        }
      });

      MathJax.Hub.Config({
      // Autonumbering by mathjax
      TeX: { equationNumbers: { autoNumber: "AMS" } }
      });
    </script>
    ```
    {% endraw %}
2. Put the following line before `</head>` in your `head.html`:
    {% raw %}
    ```html
    {% include mathjax.html %}
    ```
    {% endraw %}
    to enbale MathJax on the page. 

That's it for MathJax. 


#### Tips
- To use the normal dollar sign instead of the MathJax command (escape), put `<span class="tex2jax_ignore">...</span>` around the text you don't want MathJax to process.
- Check out currently supported Tex/LaTeX commands by MathJax [here](https://docs.mathjax.org/en/latest/input/tex/macros/index.html).


#### Math Rendering Showcase
- Inline math using `\$...\$`: $\mathbf{x}+\mathbf{y}$.
- Displayed math using `\$\$...\$\$` on a new paragraph: 

$$
\mathbf{x}+\mathbf{y} \,.
$$

- Automatic numbering and referencing using <span class="tex2jax_ignore">`\ref{label}`</span>:
In (\ref{eq:sample}), we find the value of an interesting integral:
\begin{align}
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15} \, .
  \label{eq:sample}
\end{align}

- Multiline equations using `\begin{align*}`:
\begin{align\*}
  \nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \,,\newline
  \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \,.
\end{align\*} 


That's it for now. Happy blogging. 

Additional resources:
- Set up [categories & tags](https://blog.webjeda.com/jekyll-categories/)
- Set up [Disqus comments & Google Analytics](http://joshualande.com/jekyll-github-pages-poole)
- MathJax basic [tutorial](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)



