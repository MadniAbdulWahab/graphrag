import networkx as nx
import matplotlib.pyplot as plt

from networkx import Graph

from .community_detection.ledian_community_detection import leiden
from .community_detection.quality_functions import Modularity

def draw_graph(G, pos=None, communities=None, labels=None, force_color={}, file='graph.png'):
    if communities:
        # Create a color map
        node_color = [0 for _ in G]

        for i, community in enumerate(communities):
            for node in community:
                node_color[node] = i

        for k, v in force_color.items():
            node_color[k] = v

        nx.draw(G, pos, node_color=node_color, cmap=plt.cm.rainbow, labels=labels)
    else:
        nx.draw(G, pos, labels=labels)

    # Save the graph image
    plt.savefig(file, bbox_inches='tight')

    # Show the graph
    plt.show()

def detect_communities(graph: Graph):
    """
        Followed ledian algorithm implementaion from from https://github.com/esclear/louvain-leiden
    """
    modularity = Modularity(1)
    partition = leiden(graph, modularity, weight="weight")
    return partition