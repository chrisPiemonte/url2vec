__author__ = 'chris'

import os
import igraph as ig
from nltk.stem.snowball import SnowballStemmer


""" Lambdas """
get_url_index = lambda url_code: int(url_code.replace("u_", ""))


""" Functions """
# returns the url map -> {code: url}
def get_urlmap(filename, sep=","):
    return dict( [s.strip() for s in line.split(sep)[::-1]] for line in open(filename, "r"))


# returns generator of sequences
def get_sequences(filename, sep=" -1 ", min_len=1):
    for line in open(filename, "r"):
        sequence = line.split(sep)[:-1]
        if len(sequence) >= min_len:
            yield sequence


# returns a color
def get_color(n):
    #        1.orange,   2.white,   3.yellow, 4.lgt-blue, 5.green,  6.blue,    7.fuchsia, 8.violet
    colors = ["#FF8F00", "#FFFFFF", "#FFFF00", "#00E5FF", "#76FF03", "#2979FF", "#F50057", "#9C27B0"]
    color = ""
    if n < 0:
        color = "#009688" # noise
    elif n < len(colors):
        color = colors[n]
    else:
        color = "#" + format(n**5, '06X')
    return color


# returns the web graph (igraph's graph object)
def create_graph(sequences, seqMap, is_directed=False):
    graph = ig.Graph(directed=is_directed)
    graph.add_vertices(len(seqMap))  # adding nodes

    if type(seqMap) is dict:
        for key in seqMap:
            i = get_url_index(key)
            graph.vs[i]["name"] = seqMap[key]  # adding longurl
            graph.vs[i]["color"] = "#2979FF"

    elif type(seqMap) is list:
        for el in seqMap:
            i = int(el[1].replace("u_", ""))
            graph.vs[i]["name"] = el[0]
            graph.vs[i]["community"] = el[2]
            graph.vs[i]["color"] = get_color(int(el[2]))

    for seq in sequences:
        for i in range(len(seq)-1):
            source = get_url_index(seq[i])
            target = get_url_index(seq[i+1])
            if graph.get_eid(source, target, directed=is_directed, error=False) == -1:
                graph.add_edge(source, target)  # adding unique edges
    return graph


def graph_from_sequences(sequences, urlmap, is_directed=False, default_color="#2979FF"):
    webgraph = ig.Graph(directed=is_directed)
    codes = [int(x) for x in urlmap_nocostraint.keys()]
    m = max(codes)
    missing_num = [ str(i) for i in range(m) if str(i) not in urlmap_nocostraint]
    webgraph.add_vertices(len(urlmap)+len(missing_num))  # adding nodes

    for key in urlmap:
        i = int(key)
        webgraph.vs[i]["name"] = urlmap[key]  # adding longurl
        webgraph.vs[i]["color"] = default_color

    for seq in sequences:
        for i in range(len(seq)-1):
            source = int(key.replace(seq[i]))
            target = int(key.replace(seq[i-1]))
            if webgraph.get_eid(source, target, directed=is_directed, error=False) == -1:
                webgraph.add_edge(source, target)  # adding unique edges

        return webgraph


def graph_from_file(filename, urlmap, sep="	", is_directed=False, default_color="#2979FF"):
    webgraph = ig.Graph(directed=is_directed)
    codes = [int(x) for x in urlmap.keys()]
    m = max(codes)
    missing_num = [ str(i) for i in range(m) if str(i) not in urlmap]
    webgraph.add_vertices(len(urlmap)+len(missing_num))  # adding nodes
    for i in range(len(missing_num)):
        webgraph.vs[int(missing_num[i])]["name"] = "missing"
        webgraph.vs[int(missing_num[i])]["color"] = "000000"

    for key in urlmap:
        i = int(key)
        webgraph.vs[i]["name"] = urlmap[key]  # adding longurl
        webgraph.vs[i]["color"] = default_color

    for line in open(filename, "r"):
        edge = tuple( [int(n) for n in line.split(sep)] )
        if webgraph.get_eid(edge[0], edge[1], directed=is_directed, error=False) == -1:
            webgraph.add_edge(edge[0], edge[1])

    return webgraph


""" Document Clustering """

tokenize          = lambda text: text.split(" ")
stem              = lambda token, stemmer=SnowballStemmer("english"): stemmer.stem(token)
tokenize_and_stem = lambda text, stemmer=SnowballStemmer("english"): [stem(token, stemmer) for token in tokenize(text)]


# returns the content map -> {code: content} content is a string
def get_content_map(filename, sep="	"):
    return dict( [s.strip() for s in line.split(sep)] for line in open(filename) )


# returns a map of -> {code, tokenized_content} tokenized_content is a list
def to_tokens_map(content_map):
    return {key: tokenize(content_map[key]) for key in content_map} # very pythonic indeed


# returns a map of ->  {code, stemmed_content} stemmed_content is a list
def to_stems_map(content_map, stemmer=SnowballStemmer("english")):
    #return {key: [stem(token, stemmer) for token in tokenize(content_map[key])] for key in content_map}
    return {key: tokenize_and_stem(content_map[key], stemmer=stemmer) for key in content_map}


# merges all the lists in one vocabulary
def get_set_vocab(tokens_map):
    return set([token for key in tokens_map for token in tokens_map[key]])


# class for retrieving the real membership from a manually pre-generated file
class GroundTruth:
    filepath = os.path.abspath(os.path.dirname(__file__)) + "/../../dataset/ground_truth/urlToMembership.txt"

    # constructor, needs the file path
    def __init__(self, fpath=filepath, sep=","):
        self.ground_truth = dict( [s.strip() for s in line.split(sep)] for line in open(fpath) )

    # returns the real cluster membership of a URL
    def get_groundtruth(self, url, print_missing=False):
        ret = "-1"
        original_url = url
        if url.startswith("https"):
            url = url.replace("https", "http")
        if not url.endswith("/"):
            url += "/"
        if url.startswith("http://www."):
            url = url.replace("http://www.", "http://")
        try:
            ret = self.ground_truth[url]
        except KeyError:
            if print_missing:
                print("Url not found -", url, "-", original_url)
        return ret


    def get_labelset(self):
        return set(self.ground_truth.values())
