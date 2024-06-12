from .core import L0Linear
import networkx as nx
import numpy as np
import torch


def gated_connections_graph(model, thresh=1e-8):
    # Extract gated weights from each layer
    gated_weights = [
        module.weight * torch.clamp(torch.sigmoid(module.log_alpha) * (module.zeta - module.gamma) + module.gamma, 0, 1)
        for module in model.layers if isinstance(module, L0Linear)
    ]

    # Create a directed graph
    graph = nx.DiGraph()

    # Add input nodes
    k = 0
    for i in range(model.layers[0].in_features):
        graph.add_node(f'input_{i}', layer='input', pos=(k, i))
    k += 1

    # loop over layers
    for layer in model.layers[:-1]:
        if not isinstance(layer, L0Linear):
            continue
        for i in range(layer.out_features):
            graph.add_node(f'hidden{k}_{i}', layer=f'hidden{k}', pos=(k, i))
        k += 1

    # Add output nodes
    last_layer = model.layers[-1]
    for i in range(last_layer.out_features):
        graph.add_node(f'output_{i}', layer='output', pos=(k, i))

    # Add edges with gated weights
    for i, layer_weights in enumerate(gated_weights):
        for j in range(layer_weights.size(0)):
            for k in range(layer_weights.size(1)):
                if np.abs(layer_weights[j, k].item()) > thresh:
                    if i == 0:  # Input to hidden1
                        graph.add_edge(f'input_{k}', f'hidden{i+1}_{j}', weight=layer_weights[j, k].item())
                    elif i < len(gated_weights)-1:
                        graph.add_edge(f'hidden{i}_{k}', f'hidden{i+1}_{j}', weight=layer_weights[j, k].item())
                    elif i == len(gated_weights)-1:
                        graph.add_edge(f'hidden{i}_{k}', f'output_{j}', weight=layer_weights[j, k].item())
                    else:
                        raise ValueError(i)

    return graph


def render_sparse_nn(g, prune=True):
    layers = list(set([it.split("_")[0] for it in sorted(g.nodes)]))
    out_features = len([it for it in g.nodes if it.startswith('output')])
    in_features = len([it for it in g.nodes if it.startswith('input')])
    paths = []
    for t in range(out_features):
        target = f'output_{t}'
        for s in range(in_features):
            source = f'input_{s}'
            paths += list(nx.all_simple_paths(g, source, target))
            for h in range(len([it for it in layers if it.startswith('hidden')])):
                layer_nodes = [it for it in g.nodes if it.startswith(f'hidden{h + 1}')]
                for ln in layer_nodes:
                    paths += nx.all_simple_paths(g, ln, target)

    if prune:
        valid_nodes = []
        for p in paths:
            valid_nodes += p
        valid_nodes = sorted(set(valid_nodes))
    else:
        valid_nodes = sorted(set(g.nodes))

    height = {node: [] for node in valid_nodes}
    for n in valid_nodes:
        for p in paths:
            if n not in p:
                continue
            if p[-1].startswith('output'):
                height[n].append(float(p[-1].split('_')[-1]))

    for n in valid_nodes:
        # convert from list to mean
        height[n] = np.mean(height[n])

    for node in list(g.nodes):
        if node not in valid_nodes:
            g.remove_node(node)

    for layer_name in layers:
        if layer_name[-1].isnumeric():
            depth = int(layer_name[-1])
        elif layer_name == 'input':
            depth = 0
        elif layer_name == 'output':
            depth = len(layers)
        else:
            raise ValueError("Expected input, hiddenX, or output")
        layer_nodes = sorted([it for it in g.nodes if it.startswith(layer_name)], key=lambda x: height[x])
        for i, node in enumerate(layer_nodes):
            g.nodes[node]['pos'] = (depth, i - len(layer_nodes) / 2)

    # update positions
    pos = {node: data['pos'] for node, data in g.nodes(data=True)}

    fig = nx.draw(g, pos, with_labels=False, node_size=10, node_color='lightblue',
                  font_size=10, font_weight='bold', edge_color='gray')

    return fig
