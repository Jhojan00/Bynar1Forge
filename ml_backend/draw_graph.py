import networkx as nx
import plotly.graph_objects as go
from math import ceil


def get_nodes(neurons, layers):
    neurons_layers = [
        ceil((neurons - 1) / (layers - 1)) for i in range(layers - 1)
    ]  # -1 because is the last layer and neuron

    last_neuron = neurons_layers[0] / 2
    neurons_per_layer = []

    for nl in range(len(neurons_layers) - 1):
        neurons_per_layer.append(ceil(neurons_layers[nl] - last_neuron))
        neurons_layers[nl + 1] += ceil(last_neuron)

    neurons_per_layer.append(neurons_layers[-1])

    neurons_by_layer = []

    for i, npl in enumerate(reversed(neurons_per_layer)):
        neurons_by_layer.append([])
        for n in range(npl):
            neurons_by_layer[i].append(f"Neuron:{n} - Layer: {i}")

    neurons_by_layer.append(["Output"])

    return neurons_by_layer


def get_graph(neurons_by_layer):
    network_graph = nx.DiGraph()

    edges = [
        (a, b)
        for i in range(len(neurons_by_layer) - 1)
        for a in neurons_by_layer[i]
        for b in neurons_by_layer[i + 1]
    ]

    network_graph.add_edges_from(edges)

    # Asign positions
    off_set = len(neurons_by_layer[0])
    for i_l, layer in enumerate(neurons_by_layer):
        for i_n, neuron in enumerate(layer):
            x = i_l
            y = i_n + (off_set - len(layer) / 2)
            network_graph.nodes[neuron]["pos"] = (x, y)

    return network_graph


def draw_graph(network_graph):
    edge_x = []
    edge_y = []
    for edge in network_graph.edges():
        x0, y0 = network_graph.nodes[edge[0]]["pos"]
        x1, y1 = network_graph.nodes[edge[1]]["pos"]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="lightgreen"),
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in network_graph.nodes():
        x, y = network_graph.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            size=10,
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Neuronal Network preview (It may have errors):",
                font=dict(size=16),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        ),
    )

    return fig
