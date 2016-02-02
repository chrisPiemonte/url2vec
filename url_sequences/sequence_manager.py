import igraph as ig
import plotly.plotly as py
from plotly.graph_objs import *
import nltk
from nltk.stem.snowball import SnowballStemmer

to_dict       = lambda line, sep: get_key_value(split_line(line, sep))
get_key_value = lambda line_list: {line_list[1].rstrip(): line_list[0]}


# -----------------------------------------------------------------------------------------------------------


""" Lambdas """
clean         = lambda string: string.strip()
split_line    = lambda line, sep: line.split(sep)
take_part     = lambda num_part, line, sep: clean(split_line(line, sep)[num_part])
to_tuple      = lambda line, sep: tuple(rem_back_n(split_line(line, sep)))
rem_back_n    = lambda coll: coll[-1].rstrip() if len(coll) > 0 else coll
remove_last   = lambda coll: coll[:-1] if len(coll) > 0 else coll
get_url_index = lambda url_code: int(url_code.replace("u_", ""))


""" Functions """
# returns the url map -> {code: url}
def get_urlmap(filename, sep=","):
    return {take_part(1, line, sep): take_part(0, line, sep) for line in open(filename, "r")}


# returns the membership map -> {code: membership} manually clusterized
def get_membership_map(filename, sep=" , "):
    return {take_part(1, line, sep): take_part(2, line, sep) for line in open(filename, "r")}


# returns the tuple list -> (url, code, membership) manually clusterized
def get_sequences_with_membership(filename, sep=" , "):
    return [(to_tuple(line, sep)) for line in open(filename, "r")]

# returns generator of sequences
def get_sequences(filename, sep=" -1 ", min_len=1):
    for line in open(filename, "r"):
        sequence = remove_last(split_line(line, sep))
        if len(sequence) >= min_len:
            yield sequence

#
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


""" Document Clustering """

tokenize          = lambda text: text.split(" ")
stem              = lambda token, stemmer=SnowballStemmer("english"): stemmer.stem(token)
tokenize_and_stem = lambda text, stemmer=SnowballStemmer("english"): [stem(token, stemmer) for token in tokenize(text)]

# returns the content map -> {code: content} content is a string
def get_content_map(filename, sep="	"):
    return {take_part(0, line, sep): take_part(1, line, sep) for line in open(filename)}

# returns a map of -> {code, tokenized_content} tokenized_content is a list
def to_tokens_map(content_map):
    return {key: tokenize(content_map[key]) for key in content_map} # very pythonic indeed

# returns a map of ->  {code, stemmed_content} stemmed_content is a list
def to_stems_map(content_map, stemmer=SnowballStemmer("english")):
    #return {key: [stem(token, stemmer) for token in tokenize(content_map[key])] for key in content_map}
    return {key: tokenize_and_stem(content_map[key]) for key in content_map}

# merges all the lists in one vocabulary
def get_total_vocab(tokens_map):
    return [token for key in tokens_map for token in tokens_map[key]]

# -----------------------------------------------------------------------------------------------------------

class RealMembership:
    filepath = "/home/chris/workspace/jupyter-notebook/url2vec/dataset/manual-membership/urlToMembership.txt"
    
    def __init__(self, fpath=filepath, sep=","):
        utm_map = {}
        for line in open(fpath, "r"):
            kv = line.split(sep)
            utm_map[kv[0]] = kv[1].strip()
            print(kv)
        self.url_membership_map = utm_map
        
    def get_membership(self, url):
        ret = "-1"
        if url.startswith("https"):
            url = url.replace("https", "http")
        if not url.endswith("/"):
            url += "/"
        try:
            ret = self.url_membership_map[url]
        except KeyError:
            print("--url not found--")
        return ret

    
