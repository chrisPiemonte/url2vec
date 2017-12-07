
# ![alt url2vec](res/img/link-logo.png "url2vec") Url2vec

### Abstract

In this thesis a new methodology for clustering Web pages is discussed, using Random Walks between pages, together with their textual content, to learn vector representations for nodes in the web graph. 
Url2vec is implemented to extract clusters of pages of the same semantic type. Unlike the clustering algorithms proposed in literature, Url2Vec does not consider a website as a collection of text documents independent from each other, but tries to combine information about the content of the pages and the structure of the website.

The experimental results produced proved to be discreet and encouraged to follow the studies in this direction to identify new ways to improve the results achieved in terms of quality.

### Setup

I suggest to setup a virtual environment using [miniconda](http://conda.pydata.org/miniconda.html)

1. Create an environment with python 2.7:
<pre>conda create --name url2vec python=2.7</pre>
2. Install requirements:
<pre>pip install -r ./requirements.txt</pre>
3. To check the examples:
<pre>jupyter-notebook ./notebooks</pre>
