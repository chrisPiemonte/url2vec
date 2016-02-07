__author__ = 'chris'

import igraph as ig
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.graph_objs as go



def graph_plot(graph, graph_name):
    py.sign_in('chrispolo', '89nned6csl')
    layt = graph.layout('kk', dim=3)
    
    Xn = [layt[k][0] for k in range(len(graph.vs))] # x-coordinates of nodes
    Yn = [layt[k][1] for k in range(len(graph.vs))] # y-coordinates of nodes
    Zn = [layt[k][2] for k in range(len(graph.vs))] # z-coordinates of nodes
    Xe = []
    
    Ye = []
    Ze = []
    
    for e in graph.es:
        Xe += [layt[e.source][0], layt[e.target] [0], None] # x-coordinates of edge ends
        Ye += [layt[e.source][1], layt[e.target] [1], None] # y-coordinates of edge ends
        Ze += [layt[e.source][2], layt[e.target] [2], None] # z-coordinates of edge ends
    
    trace1 = Scatter3d(
        x = Xe,
        y = Ye,
        z = Ze,
        mode = 'lines',
        line = Line(
            color = 'rgb(125,125,125)',
            width = 0.5
        ),
        hoverinfo = 'none'
    )
    
    trace2 = Scatter3d(
        x = Xn,
        y = Yn,
        z = Zn,  
        mode = 'markers',
        name = 'actors',
        marker = Marker(
            symbol = 'dot',
            size = 6,
            color = graph.vs["color"],
            colorscale = 'Viridis',
            line = Line(
                color = 'rgb(50,50,50)',
                width = 0.5
            )
        ),
        text = graph.vs["name"],
        hoverinfo = 'text'
    )
    
    axis = dict(
        showbackground = False,
        showline = False,
        zeroline = False,
        showgrid = False,
        showticklabels = False,
        title = ''
    )
    
    layout = Layout(
        title = graph_name,
        width = 1000,
        height = 1000,
        showlegend = False,
        scene = Scene(
            xaxis = XAxis(axis),
            yaxis = YAxis(axis),
            zaxis = ZAxis(axis),
        ),
        margin = Margin(
            t = 100
        ),
        hovermode = 'closest',
        annotations = Annotations([
                Annotation(
                    showarrow = False,
                    text = "Data source: <a href='#'>[1]</a>",
                    xref = 'paper',
                    yref = 'paper',
                    x = 0,
                    y = 0.1,
                    xanchor = 'left',
                    yanchor = 'bottom',
                    font = Font(
                        size = 14
                    )
                )
        ]),
    )
    
    data = Data([trace1, trace2])
    fig = Figure(data=data, layout=layout)
    return fig
    # py.iplot(fig, filename=graph_name)
    

def scatter_plot(two_dim_vecs, word_labels=None, colors="#FFFF00"):
    py.sign_in('chrispolo', '89nned6csl')
    
    x_coord = two_dim_vecs[:, 0]
    y_coord = two_dim_vecs[:, 1]
    
    trace = go.Scattergl(
        x = x_coord, #
        y = y_coord, #
        mode = 'markers',
        text = word_labels, #
        marker = dict(
            color = colors,
            line = dict(width = 1)
        )
    )

    data = [trace]
    return data

