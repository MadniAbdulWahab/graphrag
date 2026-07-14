import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import random

def create_graph_html():
    G = nx.read_gexf("output/graph_valid.gexf")
    
    # Create Pyvis network
    net = Network(notebook=False)

    # Add nodes with colors based on type
    for node, data in G.nodes(data=True):
        net.add_node(node, title=data.get("description", ""))

    # Add edges with weights
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(u, v, value=weight, title=data.get("description", ""))

    # Save the visualization
    net.save_graph("output/graph_valid.html")
    print(f"\nGraph saved as graph_valid.html")
    print(f"Number of nodes: {len(G.nodes)}")
    print(f"Number of edges: {len(G.edges)}")

# if name == "main":
# create_graph_html()


def create_graph_png():


    # Load the graph from the GEXF file
    G = nx.read_gexf("output/graph_valid.gexf")

    # Filter out isolated nodes (degree = 0)
    # connected_nodes = [node for node in G.nodes() if G.degree(node) > 0]
    # H = G.subgraph(connected_nodes)  # Create a subgraph with only connected nodes

    # Extract community IDs from node attributes
    # Assuming the community ID is stored in an attribute called 'community_id'
    community_map = {node: data['community_id'] for node, data in G.nodes(data=True)}

    # Get unique community IDs
    unique_communities = set(community_map.values())

    # Print the total number of nodes and communities
    print(f"Total number of nodes (connected only): {G.number_of_nodes()}")
    print(f"Total number of communities: {len(unique_communities)}")

    # Assign a random color to each community
    community_colors = {community_id: f'#{random.randint(0, 0xFFFFFF):06x}' for community_id in unique_communities}
    node_colors = [community_colors[community_map[node]] for node in G.nodes()]

    # Draw the graph with community-based coloring
    plt.figure(figsize=(25, 25))
    pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
    nx.draw(
        G, pos, with_labels=False, node_size=10, node_color=node_colors
    )

    plt.title("Graph Colored by Communities (Connected Nodes Only)")
    plt.savefig('output/figures/graph_valid.png')
    plt.show()

create_graph_png()