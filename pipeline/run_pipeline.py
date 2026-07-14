import os
import torch
import re
import json
import networkx as nx
import community as community_louvain

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

from .preprocessing.create_chunks import create_chunks
from .preprocessing.entity_types import get_entity_types
from .preprocessing.entity_realtions import get_relations
from .preprocessing.entity_summarization import get_entity_summaries
from .preprocessing.graph_communities import generate_graph
from .preprocessing.community_summaries import generate_community_summaries
from .generation.generate_answer import generate_answer

from .utils.prompts import DEFAULT_TASK, ENTITY_TYPE_GENERATION_PROMPT, DEFAULT_TASK,ENTITY_RELATIONSHIPS_GENERATION_PROMPT, GENERATE_PERSONA_PROMPT
from .utils.load_huggingface_dataset import get_context_merged_datset
from .utils.graph_communities import draw_graph, detect_communities

# Load environment variables
load_dotenv()

def create_persona(task, model, tokenizer, device):
    prompt = GENERATE_PERSONA_PROMPT.format(sample_task=task)    
    inputs = tokenizer(prompt, return_tensors='pt').to(device=device)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=1000, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    phrase="persona description: \n"
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    match = pattern.search(text)
    #! print(f'RESPONSE: {text}')

    if match:
        # Extract the text after the phrase
        start_pos = match.end()
        remaining_text = text[start_pos:]
        
        # Find the first paragraph (split by double newline or single newline)
        first_paragraph = remaining_text.split('\n\n', 1)[0].strip()
        return first_paragraph
    else:
        return "Phrase not found."

def get_affected_communities(global_graph, affected_nodes):
    affected_communities = set()
    for node in affected_nodes:
        if 'community_id' in global_graph.nodes[node]:
            affected_communities.add(global_graph.nodes[node]['community_id'])
    return affected_communities



def run_pipeline(load = False):
    # Login to Hugging Face to use LLM models
    login(token=os.environ["HUGGINGFACE_TOKEN_LLAMA32"])
    summary_file = os.environ['COMMUNITIES_FILE']
    
    # Set the default device to trun the models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ## Load Dataset
    squad_df = get_context_merged_datset(dataset_name="rajpurkar/squad")
    
    # Initialize the tokenizer and model
    tokenizer_retrieval = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model_retrieval = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)

    tokenizer_retrieval.pad_token_id = tokenizer_retrieval.eos_token_id
    
    metadata_exists = os.path.exists(os.environ['GRAPH_FILE']) and os.path.exists(os.environ['COMMUNITIES_FILE']) and os.path.exists(os.environ['STATUS_FILE']) 
    if load and metadata_exists:
        status = load_status()
        current_iteration = status.get("iteration", 0)
        global_graph = nx.read_gexf(os.environ['GRAPH_FILE'])
        print(f'GRAPH LOADED FROM ITERATION: {current_iteration}')
    else:

        current_iteration = 0
        global_graph = nx.Graph()

    for i, row_data in tqdm(squad_df.iterrows(), total=squad_df.shape[0], desc="Running pipeline"):
        if i < current_iteration:
            continue #? Skip already processed iterations
        
        current_iteration = i + 1
        tqdm.write(f"Processing row {i} / {squad_df.shape[0]}") 
        ##* Step 0: Ready preliminary variable 
        domain = row_data['title']
        task = DEFAULT_TASK.format(domain=domain)
        
        persona = create_persona(task=task, model=model_retrieval, tokenizer=tokenizer_retrieval, device=device)
        
        ##* Step 1: Separate context into chunks for LLM context window. 
        chunks = create_chunks(row_data['context'])
        
        ##* Step 2: Create list of entities and relationships
        # tqdm.set_description(desc='Generating Entities and Relations')
        entity_relations = []
        for _, row_chunk in chunks.iterrows():
            ##* Step 2.1: Get entity types from the text
            enitity_types_prompt = ENTITY_TYPE_GENERATION_PROMPT.format(task=task, input_text=row_chunk['chunk'])
            entity_types = get_entity_types(model_retrieval, tokenizer_retrieval, device, prompt=enitity_types_prompt)
            #! print(f'ENTITY TYPES: {entity_types}\n')

            ##* Step 2.2: Get entities and relationships
            try:
                entity_types_str =   ", ".join(entity_types['entity_types'])
            except:
                entity_types_str = ", ".join(["PERSON", "COUNTRY", "CITY" "ORGANIZATION", "DATE", "EVENT", "BUILDING", "CULTUE", "HISTORICAL EVENT"])
                
            entity_relations_prompt = ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(
                entity_types=entity_types_str, 
                language='english', 
                input_text=row_chunk['chunk']
            )
            entity_relations = get_relations(model_retrieval, tokenizer_retrieval, device, prompt=entity_relations_prompt)
            #! print(f'ENTITY RELATIONS: {entity_relations}\n')

        ##* Step 3: Generate Element Summaries
        # tqdm.set_description(desc='Generating Entity Summaries')
        summaries, entities, relationships = get_entity_summaries(entity_relations, persona, model_retrieval, tokenizer_retrieval, device)
        #! print(f'SUMMARIES: {summaries}')
        
        ##* Step 4:  Generate Graph Communities
        local_graph = generate_graph(relationships, summaries, entities)
        
        new_nodes = set(local_graph.nodes) - set(global_graph.nodes)
        new_edges = set(local_graph.edges) - set(global_graph.edges)
        
        global_graph = nx.compose(local_graph, global_graph)
        
        ## We use the louvain algorithm for community generation
        affected_nodes = new_nodes | {edge[0] for edge in new_edges} | {edge[1] for edge in new_edges}
        affected_subgraph = global_graph.subgraph(affected_nodes)
        
        # Determine the highest existing community ID in the global graph
        existing_community_ids = {
            data.get('community_id', -1) for _, data in global_graph.nodes(data=True)
        }
        max_existing_community_id = max(existing_community_ids) if existing_community_ids else -1

        # Run Louvain algorithm on the affected subgraph
        partition = community_louvain.best_partition(affected_subgraph, weight='strength')

        # Offset new community IDs to avoid conflicts with existing IDs
        community_id_offset = max_existing_community_id + 1
        for node, community_id in partition.items():
            adjusted_community_id = community_id + community_id_offset
            global_graph.nodes[node]['community_id'] = adjusted_community_id  # Store adjusted community ID as node attribute

        # Keep track of the communities in a separate dictionary for reference 
        communities = {}
        for node, community_id in partition.items():
            communities.setdefault(community_id, []).append(node)
        
        ##* Step 5: Generate Community Summaries
        # Identify affected communities
        affected_communities = get_affected_communities(global_graph, affected_nodes)

        # Load existing summaries (if available) and only update affected communities
        existing_summaries = {}
        
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                existing_summaries = json.load(f)
        # tqdm.set_description(desc='Generating Commmunity summaries')
        new_summaries = generate_community_summaries(global_graph, affected_communities, model_retrieval, tokenizer_retrieval, device)

        # Merge existing summaries with new summaries
        updated_summaries = {**existing_summaries, **new_summaries}

        # Write updated summaries to JSON
        with open(summary_file, "w") as f:
            json.dump(updated_summaries, f)
        
        # Save the graph to file
        nx.write_gexf(global_graph, os.environ['GRAPH_FILE'])

        # Save the graph status
        status = {"iteration": current_iteration, "graph_file": os.environ['GRAPH_FILE']}
        save_status(status)
     
# Function to load status
def load_status():
    if os.path.exists(os.environ['STATUS_FILE']):
        with open(os.environ['STATUS_FILE'], "r") as f:
            return json.load(f)
    return {"iteration": 0, "graph_file": os.environ['GRAPH_FILE']}

# Function to save status
def save_status(status):
    with open(os.environ['STATUS_FILE'], "w") as f:
        json.dump(status, f)


    