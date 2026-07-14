import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/utils')))

from load_huggingface_dataset import get_context_merged_datset
import json
from transformers import BitsAndBytesConfig
import random
import numpy as np
from transformers import pipeline
import logging
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login
import torch
import faiss
torch.cuda.empty_cache()
quant_config = BitsAndBytesConfig(load_in_8bit=True)

login(token=os.environ["HUGGINGFACE_TOKEN_LLAMA32"])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the data from the files
graph_file = "graph.gexf"
community_summaries_file = "community_summaries.json"

# File to store results
output_file = "query_results3.json"
results = []

# Extract questions
merged_df = get_context_merged_datset()
questions = merged_df['question']

user_queries = [question for context in questions[:8] for question in context]
# user_queries = ["Which NASA faction came around first to support the LOR?"]

generation_settings = {
    "max_new_tokens": 300,       # Maximum output length
    "temperature": 0.4,          # Reduce randomness
    "top_p": 0.9,                # Control diversity
    "num_beams": 8,              # Use beam search for quality
    "repetition_penalty": 1.1,   # Penalize repetitive phrases
    "num_return_sequences": 1    # Generate only one output
}



# Step 1: Load and parse community summaries
logger.info("Loading community summaries...")
with open(community_summaries_file, "r") as f:
    community_summaries = json.load(f)
logger.info(f"Loaded {len(community_summaries)} community summaries.")

# Prepare the community summaries
limited_summaries = dict(list(community_summaries.items()))

# Load a semantic similarity model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Loaded semantic similarity model.")

def filter_relevant_summaries_faiss(query, index_file="faiss_index", data_file="summary_data.npz", model_name="all-MiniLM-L6-v2", top_n=15):
    """Filters the top N relevant summaries using FAISS."""
    # Ensure files exist
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file '{index_file}' not found. Please create the index first.")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Summary data file '{data_file}' not found. Please create the data file first.")

    # Load FAISS index and summary data
    index = faiss.read_index(index_file)
    data = np.load(data_file, allow_pickle=True)
    summary_keys = data["summaries"]
    summary_texts = data["texts"]

    # Encode query
    semantic_model = SentenceTransformer(model_name)
    query_embedding = semantic_model.encode([query], convert_to_numpy=True)

    # Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search top N relevant summaries
    distances, indices = index.search(query_embedding, top_n)
    relevant_summaries = {summary_keys[i]: summary_texts[i] for i in indices[0]}

    logger.info(f"Filtered top {top_n} relevant summaries using FAISS.")
    return relevant_summaries


# Step 2: Filter summaries dynamically
def prepare_chunks(summaries, chunk_size=300):
    all_chunks = []
    for summary in summaries.values():
        tokens = summary.split()
        chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        all_chunks.extend(chunks)
    logger.info(f"Prepared {len(all_chunks)} chunks.")
    return all_chunks

# Initialize a second 3B model for global answer generation
llm_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device_map="auto", torch_dtype="auto")
# llm_pipeline_global = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device_map="auto", torch_dtype="auto")

# Iterate through all user queries and process dynamically
for query_index, user_query in enumerate(user_queries, start=1):
    logger.info(f"Processing query {query_index}/{len(user_queries)}: {user_query}")
    
    relevant_summaries = filter_relevant_summaries_faiss(
        query=user_query,
        index_file="faiss_index",
        data_file="summary_data.npz"
    )


    chunks = prepare_chunks(relevant_summaries)

    # Print the chunks
    logger.info("Printing prepared chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}\n")


    def extract_score_and_answer(response):
        """
        Extracts the score and answer from the model's response using robust logic.
        """
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
                if "[Your answer here]" not in ans and ans not in ["", "Not found in the provided context.'", "Not found in the provided context."]:
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


    def generate_intermediate_answers(chunks, query):
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

                response = llm_pipeline(input_text, **generation_settings)[0]['generated_text']
                logger.info(f"Model Response for Chunk {i+1}: {response}")

                # Extract score and answer
                score, answer = extract_score_and_answer(response)
                intermediate_answers.append({"answer": answer, "score": score})
                # intermediate_answers.append({"score": score})
            except Exception as e:
                logger.warning(f"Failed to process Chunk {i+1}: {chunk}. Error: {e}")
                intermediate_answers.append({"answer": "No valid answer generated.", "score": 0})
        logger.info("Completed generating intermediate answers.")
        return intermediate_answers

    intermediate_answers = generate_intermediate_answers(chunks, user_query)

    # Display scores for all intermediate answers
    print("Scores for intermediate answers:")
    for i, answer in enumerate(intermediate_answers, 1):
        print(f"Chunk {i}: Score = {answer['score']}")

    # Step 4: Reduce to global answer
    def generate_global_answer(intermediate_answers, token_limit=512):
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
        
        # Log the global context for debugging
        logger.info(f"Global context after processing valid answers:\n{global_context.strip()}")

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
            final_answer = llm_pipeline(final_input, max_new_tokens=200, num_return_sequences=1, truncation=True)[0]['generated_text']
            logger.info(f"Generated global answer. here it is:\n{final_answer}")
            matches = re.findall(r"Answer:\s*(.*)", final_answer, re.IGNORECASE)
            for match in matches:
                answer = match.strip()
                # Validate the answer
                if (
                    "[Your answer here]" not in answer
                    and "Not found in the provided context." not in answer
                    and answer  # Ensure it's not empty
                ):
                    logger.info(f"Valid answer found: {answer}")
                    return answer

            # If no valid answer is found
            logger.warning("No valid answer found in response.")
            return "No valid answer found."
        except Exception as e:
            return "Failed to generate a global answer."

    global_answer = generate_global_answer(intermediate_answers)
    print("Global Answer:", global_answer)
    
    # Step 5: Save the scores and global answer for the current query
    query_result = {
        "query": user_query,
        "scores": [answer["score"] for answer in intermediate_answers],
        "global_answer": global_answer
    }
    results.append(query_result)
    logger.info(f"Saved results for query {query_index}.")


# Write all results to a JSON file
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
logger.info("Completed processing all user queries.")