import re
import sys
import os

# If needed for your local environment:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/utils')))

import json
import random
import logging

import torch
torch.cuda.empty_cache()

from transformers import BitsAndBytesConfig, pipeline
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, util

# ----- NEW IMPORTS FOR HYBRID SEARCH -----
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
# -----------------------------------------

from load_huggingface_dataset import get_context_merged_datset

quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Log in to Hugging Face if needed
login(token=os.environ["HUGGINGFACE_TOKEN_LLAMA32"])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
graph_file = "graph.gexf"
community_summaries_file = "community_summaries.json"
output_file = "query_results6.json"

# Step 1: Load question data
merged_df = get_context_merged_datset()
questions = merged_df['question']
user_queries = [question for context in questions[:8] for question in context]
# For testing a single query, uncomment:
#user_queries = ["When was the Doctor Who series released on DVD?"]

generation_settings = {
    "max_new_tokens": 200,
    "temperature": 0.3,
    "top_p": 0.9,
    "num_beams": 8,
    "repetition_penalty": 1.1,
    "num_return_sequences": 1
}

# Step 2: Load community summaries
logger.info("Loading community summaries...")
with open(community_summaries_file, "r") as f:
    community_summaries = json.load(f)
logger.info(f"Loaded {len(community_summaries)} community summaries.")

# Convert dict to lists for indexing
summary_keys = list(community_summaries.keys())      # e.g. ["0", "1", "241", "811", ...]
summary_texts = list(community_summaries.values())   # The actual text

logger.info("Preparing to build FAISS and BM25 indexes...")

# ---------------------------------------------------------------------
#  BUILD A SINGLE FAISS + BM25 INDEX FOR ALL SUMMARIES (ONCE)
# ---------------------------------------------------------------------

# 1) Build or load a sentence-transformer model
#    Using all-mpnet-base-v2 (handles synonyms better)
semantic_model_name = "all-MiniLM-L6-v2"
logger.info(f"Loading sentence-transformer model: {semantic_model_name}")
semantic_model = SentenceTransformer(semantic_model_name)

# 2) Encode all summaries into embeddings (float32 for FAISS)
logger.info("Creating embeddings for all summaries...")
all_summary_embeddings = semantic_model.encode(summary_texts, convert_to_tensor=False)
all_summary_embeddings = np.array(all_summary_embeddings, dtype='float32')
embedding_dim = all_summary_embeddings.shape[1]
logger.info(f"All summary embeddings shape: {all_summary_embeddings.shape}")

# 3) Build FAISS index with inner-product (for similarity)
logger.info("Building FAISS index...")
faiss_index = faiss.IndexFlatIP(embedding_dim)
faiss_index.add(all_summary_embeddings)
logger.info(f"FAISS index built with {faiss_index.ntotal} documents.")

# 4) Build BM25 index
logger.info("Building BM25 index...")
tokenized_texts = [text.split() for text in summary_texts]  # simplistic tokenization
bm25_index = BM25Okapi(tokenized_texts)
logger.info("BM25 index built.")

# ---------------------------------------------------------------------
#  FUNCTION: Weighted Hybrid to Get Top-N Summaries
# ---------------------------------------------------------------------
def filter_relevant_summaries_hybrid(summaries_dict, query, top_n=15, alpha=0.5, large_k=200):
    """
    Returns top-n relevant summaries from the entire corpus using Weighted Hybrid of BM25 + FAISS.
      - summaries_dict: original dictionary {key: text}
      - query: user query string
      - top_n: how many final results to return
      - alpha: weight for BM25 vs. embeddings (0.0 -> purely embeddings, 1.0 -> purely BM25)
      - large_k: how many candidates to retrieve from each index before combining
    """
    # 1) BM25 search
    query_tokens = query.split()
    bm25_scores = bm25_index.get_scores(query_tokens)  # shape: (num_docs,)
    # Get top large_k doc indices for BM25
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:large_k]

    # 2) FAISS search
    query_emb = semantic_model.encode(query, convert_to_tensor=False)
    query_emb = np.array([query_emb], dtype='float32')  # shape: (1, dim)
    distances, faiss_indices = faiss_index.search(query_emb, large_k)
    faiss_indices = faiss_indices[0]    # shape: (large_k,)
    faiss_scores = distances[0]         # shape: (large_k,)

    # 3) Combine candidates
    candidates = set(bm25_top_indices).union(set(faiss_indices))
    candidates = list(candidates)

    # 4) Gather candidate BM25 + Embedding scores
    candidate_bm25_scores = [bm25_scores[idx] for idx in candidates]

    candidate_embs = all_summary_embeddings[candidates]
    dot_scores = (candidate_embs @ query_emb.T).flatten()  # shape: (len(candidates),)

    # 5) Normalize scores for weighted combination (min-max normalization)
    bm25_min, bm25_max = min(candidate_bm25_scores), max(candidate_bm25_scores)
    bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1e-9
    norm_bm25 = [(s - bm25_min) / bm25_range for s in candidate_bm25_scores]

    dot_min, dot_max = min(dot_scores), max(dot_scores)
    dot_range = dot_max - dot_min if dot_max != dot_min else 1e-9
    norm_dot = [(d - dot_min) / dot_range for d in dot_scores]

    # 6) Weighted Hybrid: combined_score = alpha*BM25 + (1-alpha)*embeddings
    combined_scores = [alpha * nb + (1 - alpha) * nd for nb, nd in zip(norm_bm25, norm_dot)]

    # 7) Sort and pick top_n
    sorted_idx = np.argsort(combined_scores)[::-1]
    final_top_indices = sorted_idx[:top_n]

    # 8) Build dict {key: text} for the top_n
    result = {}
    for idx_pos in final_top_indices:
        doc_idx = candidates[idx_pos]
        key = summary_keys[doc_idx]
        text = summary_texts[doc_idx]
        result[key] = text

    return result


# ---------------------------------------------------------------------
#  CHUNK PREPARATION (same as before)
# ---------------------------------------------------------------------
def prepare_chunks(summaries, chunk_size=300):
    all_chunks = []
    for summary in summaries.values():
        tokens = summary.split()
        chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        all_chunks.extend(chunks)
    logger.info(f"Prepared {len(all_chunks)} chunks.")
    return all_chunks

# Initialize your LLM pipeline for final generation
llm_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

# Helper functions (unchanged from your code) -----------------------
def extract_score_and_answer(response):
    try:
        score_match = re.search(r"score:\s*(\d+)", response, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 0
        answer_match = re.findall(r"answer:\s*(.*?)(?=\n|score:|$)", response, re.IGNORECASE)
        answer = None
        for ans in answer_match:
            ans = ans.strip()
            if "[Your answer here]" not in ans and ans not in ["", "Not found in the provided context.'", "Not found in the provided context."]:
                answer = ans
                break
        if not answer:
            answer = "No valid answer found in response."
        return score, answer
    except Exception as e:
        logger.warning(f"Failed to extract score and answer. Error: {e}")
        return 0, "Failed to extract answer."

def generate_intermediate_answers(chunks, query):
    intermediate_answers = []
    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with query: {query}")
            input_text = (
                f"You are an expert assistant tasked with answering the query strictly based on the given context. "
                f"Ignore your own knowledge or assumptions.\n"
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
            response = llm_pipeline(input_text, **generation_settings)[0]['generated_text']
            logger.info(f"Model Response for Chunk {i+1}: {response}")
            score, answer = extract_score_and_answer(response)
            intermediate_answers.append({"answer": answer, "score": score})
        except Exception as e:
            logger.warning(f"Failed to process Chunk {i+1}: {chunk}. Error: {e}")
            intermediate_answers.append({"answer": "No valid answer generated.", "score": 0})
    logger.info("Completed generating intermediate answers.")
    return intermediate_answers

def generate_global_answer(intermediate_answers, user_query, token_limit=512):
    valid_answers = [ans for ans in intermediate_answers if ans['score'] > 0]
    valid_answers.sort(key=lambda x: x['score'], reverse=True)

    if not valid_answers:
        return "No valid answers were generated."

    global_context = ""
    for ans in valid_answers:
        # Ensure we don't exceed token_limit
        if len(global_context.split()) + len(ans['answer'].split()) <= token_limit:
            global_context += ans['answer'] + " "
        else:
            break

    if not global_context.strip():
        return "Global context is empty."

    logger.info(f"Global context:\n{global_context.strip()}")

    final_input = (
        f"You are an expert assistant tasked with generating a single concise answer to the query strictly based on the given context.\n"
        f"Context: {global_context.strip()}\n"
        f"Query: {user_query}\n"
        f"Instructions:\n"
        f"1. Ignore your own knowledge, assumption or any external information.\n"
        f"2. Provide a short, single-sentence direct answer without any explanations or commentary.\n"
        f"Format your response as follows:\n"
        f"Answer: [Your answer here]\n"
    )

    try:
        final_answer = llm_pipeline(final_input, **generation_settings)[0]['generated_text']
        logger.info(f"Global generation response:\n{final_answer}")

        matches = re.findall(r"Answer:\s*(.*)", final_answer, re.IGNORECASE)
        for match in matches:
            ans = match.strip()
            if "[Your answer here]" not in ans and "Not found in the provided context." not in ans and ans:
                return ans

        return "No valid answer found."
    except Exception as e:
        logger.warning(f"Failed to generate a global answer. Error: {e}")
        return "Failed to generate a global answer."

# ---------------------------------------------------------------------
# MAIN LOOP: For each user query, do Weighted Hybrid -> chunk -> LLM
# ---------------------------------------------------------------------
results = []
for query_index, user_query in enumerate(user_queries, start=1):
    logger.info(f"\nProcessing query {query_index}/{len(user_queries)}: {user_query}")

    # 1) Retrieve top-15 relevant summaries using Weighted Hybrid
    relevant_summaries = filter_relevant_summaries_hybrid(
        summaries_dict=community_summaries,
        query=user_query,
        top_n=15,
        alpha=0.5,       # 0.5 = equal weight to BM25 & embeddings
        large_k=200      # retrieve top-200 from each method before combining
    )

    # 2) Prepare chunks from those top summaries
    chunks = prepare_chunks(relevant_summaries)

    # 3) Get intermediate answers per chunk
    intermediate_answers = generate_intermediate_answers(chunks, user_query)
    print("Intermediate Answers Scores:")
    for i, ans in enumerate(intermediate_answers, 1):
        print(f"  Chunk {i}: Score = {ans['score']}")

    # 4) Summarize them into a single global answer
    global_answer = generate_global_answer(intermediate_answers, user_query)
    print(f"Global Answer: {global_answer}")

    # 5) Save the results for this query
    query_result = {
        "query": user_query,
        "scores": [a["score"] for a in intermediate_answers],
        "global_answer": global_answer
    }
    results.append(query_result)

# Finally, write out to JSON
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

logger.info("Completed processing all user queries.")
