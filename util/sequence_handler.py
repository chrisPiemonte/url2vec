__author__ = 'chris'

import igraph as ig
import plotly.plotly as py
from plotly.graph_objs import *
from url_sequences.sequence_manager import *


rem_back_n = lambda coll: coll[-1].rstrip() if len(coll) > 0 else coll
rem_last = lambda coll: coll[:-1] if len(coll) > 0 else coll
split_line = lambda line, sep: line.split(sep)
get_key_value = lambda line_list: {line_list[1].rstrip(): line_list[0]}
get_url_index = lambda url_code: url_code.replace("u_", "")

to_dict = lambda line, sep: get_key_value(split_line(line, sep))
to_tuple =  lambda line, sep: tuple(rem_back_n(split_line(line, sep)))



def get_sequence_map(filename):
    url_dict = {}
    for line in open(filename, "r"):
        tup = line.split(" , ")
        url_dict[tup[1].rstrip()] = tup[0]
    return url_dict


# same with lambda
def get_seq_map(filename):
    url_dict = {}
    [url_dict.update(to_dict(line, " , ")) for line in open(filename, "r")]
    return url_dict


def get_sequence_tuple_list(filename):
    url_list = []
    for line in open(filename, "r"):
        elements = line.split(" , ")
        elements[-1] = elements[-1].rstrip()
        url_list.append(tuple(elements))
    return url_list


# same with lambda
def get_seq_tuple_list(filename):
    url_list = []
    [url_list.append(to_tuple(line, " , ")) for line in open(filename, "r")]
    return url_list


def get_sequences(filename, min_len=1):
    for line in open(filename, "r"):
        sequence = line.split(" -1 ")
        sequence.pop(len(sequence)-1)
        if len(sequence) >= min_len:
            yield sequence


# same with lambda
def get_seq(filename, min_len=1):
    for line in open(filename, "r"):
        sequence = rem_last(split_line(line, " -1 "))
        if len(sequence) >= min_len:
            yield sequence


# useless         
def get_labels(sequenceMap):
    labels = [None] * len(sequenceMap)
    for key in sequenceMap:
        i = int(key.replace("u_", ""))
        labels[i] = sequenceMap[key]
    return labels


def create_graph(sequences, seqMap):
    graph = ig.Graph(directed=False)
    graph.add_vertices(len(seqMap))  # adding nodes
    
    if type(seqMap) is dict:
        for key in seqMap:
            i = int(key.replace("u_", ""))
            graph.vs[i]["name"] = seqMap[key]  # adding labels
            graph.vs[i]["color"] = "#2979FF"
    
    elif type(seqMap) is list:
        for el in seqMap:
            i = int(el[1].replace("u_", ""))
            graph.vs[i]["name"] = el[0]
            graph.vs[i]["community"] = el[2]
            graph.vs[i]["color"] = get_color(int(el[2]))
        
    for seq in sequences:
        for i in range(len(seq)-1):
            source = int(seq[i].replace("u_", ""))
            target = int(seq[i+1].replace("u_", ""))
            if graph.get_eid(source, target, directed=True, error=False) == -1:
                graph.add_edge(source, target)  # adding unique edges
    return graph