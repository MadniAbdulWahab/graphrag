import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/utils')))
import json
import random
import logging
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
graph_file = "graph.gexf"
community_summaries_file = "community_summaries.json"
output_file = "query_results.json"

# User query
user_queries = ["When was the Doctor Who series released on DVD?"]

generation_settings = {
    "max_new_tokens": 300,       # Maximum output length
    "temperature": 0.4,          # Reduce randomness
    "top_p": 0.9,                # Control diversity
    "num_beams": 8,              # Use beam search for quality
    "repetition_penalty": 1.1,   # Penalize repetitive phrases
    "num_return_sequences": 1    # Generate only one output
}


# Step 1: Load graph data
logger.info("Loading graph data...")
graph = nx.read_gexf(graph_file)
node_descriptions = {node: data.get("description", "") for node, data in graph.nodes(data=True)}
logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

# Step 2: Load community summaries
logger.info("Loading community summaries...")
with open(community_summaries_file, "r") as f:
    community_summaries = json.load(f)
logger.info(f"Loaded {len(community_summaries)} community summaries.")

# Step 3: Load semantic similarity model
logger.info("Loading semantic similarity model...")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Define helper functions
def filter_relevant_context(query, summaries, nodes, top_n=10):
    """Filter the most relevant summaries and graph nodes based on semantic similarity."""
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)

    # Combine summaries and node descriptions
    context_texts = list(summaries.values()) + list(nodes.values())
    context_ids = list(summaries.keys()) + list(nodes.keys())

    # Compute embeddings for context
    context_embeddings = semantic_model.encode(context_texts, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(query_embedding, context_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:top_n]

    # Select top N relevant context pieces
    relevant_context = {context_ids[i]: context_texts[i] for i in top_indices}
    logger.info(f"Filtered top {top_n} relevant context.")
    return relevant_context

def prepare_chunks(context, chunk_size=300):
    """Split context into manageable chunks for generation."""
    all_chunks = []
    for text in context.values():
        tokens = text.split()
        chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        all_chunks.extend(chunks)
    logger.info(f"Prepared {len(all_chunks)} chunks.")
    return all_chunks

def extract_score_and_answer(response):
    """Extracts the score and answer from the model's response using robust logic."""
    try:
        # Regular expression to match "Score: <number>"
        score_match = re.search(r"score:\s*(\d+)", response, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 0

        # Regular expression to extract "Answer:" and its content before "Score:"
        answer_match = re.findall(r"answer:\s*(.*?)(?=\n|score:|$)", response, re.IGNORECASE)
        answer = None

        # Skip placeholder answers
        for ans in answer_match:
            ans = ans.strip()
            if "[Your answer here]" not in ans and ans not in ["", "Not found in the provided context.'"]:
                answer = ans
                break

        # Fallback if no valid answer found
        if not answer:
            answer = "No valid answer found in response."

        logger.info(f"Extracted Answer: {answer}")
        logger.info(f"Extracted Score: {score}")

        return score, answer
    except Exception as e:
        logger.warning(f"Failed to extract score and answer. Error: {e}")
        return 0, "Failed to extract answer."

def generate_intermediate_answers(chunks, query, llm_pipeline_global):
    """Generate intermediate answers for chunks using the query."""
    intermediate_answers = []
    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with query: {query}")
            input_text = (
                f"You are an expert assistant tasked with answering the query strictly based on the given context. Ignore your own knowledge or assumptions.\n"
                f"Context: {chunk}\n\n"
                f"Query: {query}\n\n"
                f"Instructions:\n"
                f"1. If the context contains sufficient information to answer the query, provide a detailed answer.\n"
                f"2. If the context does not contain the exact information, infer the most likely answer based on the given context.\n"
                f"3. If the context does not contain sufficient information to answer the query, respond with:\n"
                f"'Answer: Not found in the provided context.'\n"
                f"4. Assign a score out of 100 based on the relevance and completeness of the context.\n"
                f"Provide your response in the following format:\n"
                f"Answer: [Your answer here]\nScore: [0-100]\n"
            )

            response = llm_pipeline_global(input_text, **generation_settings)[0]['generated_text']
            logger.info(f"Model Response for Chunk {i+1}: {response}")

            # Extract score and answer
            score, answer = extract_score_and_answer(response)
            intermediate_answers.append({"answer": answer, "score": score})
        except Exception as e:
            logger.warning(f"Failed to process Chunk {i+1}: {chunk}. Error: {e}")
            intermediate_answers.append({"answer": "No valid answer generated.", "score": 0})
    logger.info("Completed generating intermediate answers.")
    return intermediate_answers

def generate_global_answer(intermediate_answers, llm_pipeline_global, token_limit=512):
    """Generate a global answer based on intermediate answers."""
    valid_answers = [ans for ans in intermediate_answers if ans['score'] > 0]
    valid_answers.sort(key=lambda x: x['score'], reverse=True)

    if not valid_answers:
        return "No valid answers were generated."

    global_context = ""
    for ans in valid_answers:
        if len(global_context.split()) + len(ans['answer'].split()) <= token_limit:
            global_context += ans['answer'] + " "
        else:
            break

    if not global_context.strip():
        return "Global context is empty."

    logger.info(f"Global context after processing valid answers:\n{global_context.strip()}")

    final_input = (
        f"You are an expert assistant tasked with generating a single concise answer to the query strictly based on the given context.\n"
        f"Context: {global_context.strip()}\n"
        f"Query: {user_query}\n"
        f"Instructions:\n"
        f"1. Ignore your own knowledge, assumption, or external information.\n"
        f"2. Provide a short, single-sentence direct answer without any explanations or commentary.\n"
        f"Answer: [Your answer here]\n"
    )

    try:
        final_answer = llm_pipeline_global(final_input, **generation_settings)[0]['generated_text']
        logger.info(f"Generated global answer:\n{final_answer}")
        matches = re.findall(r"Answer:\s*(.*)", final_answer, re.IGNORECASE)
        for match in matches:
            answer = match.strip()
            if "[Your answer here]" not in answer and answer:
                return answer
        return "No valid answer found."
    except Exception as e:
        logger.error(f"Failed to generate a global answer. Error: {e}")
        return "Failed to generate a global answer."

# Step 5: Main loop for query processing
#llm_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device_map="auto", torch_dtype="auto")
llm_pipeline_global = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device_map="auto", torch_dtype="auto")

results = []
for query_index, user_query in enumerate(user_queries, start=1):
    logger.info(f"Processing query {query_index}/{len(user_queries)}: {user_query}")
    relevant_context = filter_relevant_context(user_query, community_summaries, node_descriptions)
    chunks = prepare_chunks(relevant_context)
    intermediate_answers = generate_intermediate_answers(chunks, user_query, llm_pipeline_global )
    global_answer = generate_global_answer(intermediate_answers, llm_pipeline_global )
    results.append({
        "query": user_query,
        "scores": [answer["score"] for answer in intermediate_answers],
        "global_answer": global_answer
    })

# Save results to JSON
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
logger.info(f"Results saved to {output_file}.")
