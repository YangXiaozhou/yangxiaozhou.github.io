---
layout: post
title:  "From Linear Discriminant Analysis to Penalized Discriminant Analysis"
date:   2019-09-21 08:00:00 +0800
categories: jekyll update
---
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Bayesian rule with class priors $\pi_1, \dots, \pi_K$:
$
\text{allocate } \mathbf{x } \text{ to } \Pi_{j} \text{ if } j = \arg\max_i \pi_i f_i(\mathbf{x}) \,.
$

Jekyll also offers powerful support for code snippets:

{% highlight python %}
def import_data(file_name):
    """
    import simulation data from csv file
    :return: sim_data: Dataframe
    """
    sim_data = pd.read_csv(file_name)
    # remove results from first dummy simulation
    sim_data = sim_data.iloc[sim_data[sim_data.t == 0].index[1]:, :]
    sim_data.reset_index(drop=True, inplace=True)
    sim_data.columns = sim_data.columns.str.replace(' ', '')
    print(sim_data.shape)
    print(sim_data.head(1))

    return sim_data
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
