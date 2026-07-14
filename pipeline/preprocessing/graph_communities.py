import networkx as nx
import matplotlib.pyplot as plt

from networkx import Graph

from ..utils.community_detection.ledian_community_detection import leiden
from ..utils.community_detection.quality_functions import Modularity


def generate_graph(relationships: list, summaries: list, entities: list):
    local_graph = nx.Graph()
    
    ## Create list of entity names
    entity_names = []
    for entity in entities:
        entity_names.append(entity['name'])    
    
    # Add nodes to the graph
    for summary in summaries:
        local_graph.add_node(summary['Entity'], type=summary['Type'], description=summary['Summary'])
        

    
    # Add edges to the graph
    for relationship in relationships:
        if relationship['source'] not in entity_names or relationship['target'] not in entity_names:
            #! print(f'Skipping {relationship}')
            continue
        local_graph.add_edge(
            relationship['source'], 
            relationship['target'], 
            relationship=relationship['relationship'], 
            relationship_strength=relationship['relationship_strength']
        )
        #! print(f'Added Edge f{relationship}')
        
    
    return local_graph