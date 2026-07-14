import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def main():
    # ---------------------------
    # 1. Load Summaries from JSON
    # ---------------------------
    json_file = 'community_summaries.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 'data' is a dictionary like {"0": "...", "1": "...", "241": "...", "811": "...", ...}
    keys = list(data.keys())           # e.g. ["0", "1", "241", ...]
    summaries = list(data.values())    # e.g. ["NASA is a leading ...", "...", "..."]

    print(f"Total summaries loaded: {len(summaries)}")
    # Quick check that keys "241" and "811" exist:
    print("Is '241' in data?", "241" in data)
    print("Is '811' in data?", "811" in data)

    # For debugging, let's identify the index of doc 241 and 811 if they exist
    idx_241 = keys.index("241") if "241" in keys else None
    idx_811 = keys.index("811") if "811" in keys else None

    # ---------------------------
    # 2. Initialize Embedding Model
    # ---------------------------
    model_name = 'all-mpnet-base-v2'  # or another SentenceTransformer model
    print(f"\nLoading SentenceTransformer model: {model_name} ...")
    model = SentenceTransformer(model_name)

    # ---------------------------
    # 3. Compute Embeddings
    # ---------------------------
    print("Creating embeddings for all summaries...")
    summary_embeddings = model.encode(summaries, convert_to_tensor=False)
    # Convert to float32 for FAISS
    summary_embeddings = np.array(summary_embeddings, dtype='float32')

    embedding_dim = summary_embeddings.shape[1]
    print(f"Embeddings created. Shape: {summary_embeddings.shape}")

    # ---------------------------
    # 4. Build FAISS Index
    # ---------------------------
    print("\nBuilding FAISS index (Inner Product)...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(summary_embeddings)
    print(f"FAISS index built with {index.ntotal} embeddings.")

    # ---------------------------
    # 5. Initialize BM25
    # ---------------------------
    tokenized_summaries = [s.split() for s in summaries]  # simple whitespace split
    bm25 = BM25Okapi(tokenized_summaries)

    # ---------------------------
    # 6. Define and Expand Query
    # ---------------------------
    base_question = "When was the Doctor Who series released on DVD?"

    # Example synonyms for “DVD” or “release” or “Doctor Who” if you want
    synonyms_for_dvd = [
       #"DVD", "home video", "physical release", "home media"
    ]
    # You can also add synonyms for "Doctor Who" or "release" if needed

    # Create an expanded query string
    expanded_query_part = " ".join(synonyms_for_dvd)
    expanded_question = f"{base_question} {expanded_query_part}"

    print("\nBase question:", base_question)
    print("Expanded question (for BM25 & embeddings):", expanded_question)

    # ---------------------------
    # 7. BM25 Scoring (Large top_k)
    # ---------------------------
    print("\nComputing BM25 scores...")
    query_tokens = expanded_question.split()
    bm25_scores = bm25.get_scores(query_tokens)  # shape: (num_summaries,)

    # We might retrieve a large top_k from BM25
    large_k = 200
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:large_k]  # descending

    # Print debug info for doc 241, doc 811
    if idx_241 is not None:
        print(f"\nDoc 241 BM25 score = {bm25_scores[idx_241]:.4f}")
    if idx_811 is not None:
        print(f"Doc 811 BM25 score = {bm25_scores[idx_811]:.4f}")

    # ---------------------------
    # 8. FAISS Scoring (Large top_k)
    # ---------------------------
    print("\nComputing FAISS (semantic) scores...")
    question_embedding = model.encode(expanded_question, convert_to_tensor=False)
    question_embedding = np.array([question_embedding], dtype='float32')  # shape: (1, dim)

    distances, faiss_indices = index.search(question_embedding, large_k)
    faiss_indices = faiss_indices[0]       # shape: (large_k,)
    faiss_scores = distances[0]           # shape: (large_k,)

    # For debugging, check if doc 241 or 811 are in the top FAISS results
    if idx_241 is not None and idx_241 in faiss_indices:
        print(f"Doc 241 is in FAISS top {large_k} with distance={faiss_scores[list(faiss_indices).index(idx_241)]:.4f}")
    else:
        print("Doc 241 not in FAISS top", large_k)

    if idx_811 is not None and idx_811 in faiss_indices:
        print(f"Doc 811 is in FAISS top {large_k} with distance={faiss_scores[list(faiss_indices).index(idx_811)]:.4f}")
    else:
        print("Doc 811 not in FAISS top", large_k)

    # ---------------------------
    # 9. Weighted Hybrid Scoring
    # ---------------------------
    print("\nPerforming Weighted Hybrid merge...")

    # STEP A: Collect a large set of candidate indices from both methods
    candidates = set(bm25_top_indices).union(set(faiss_indices))
    candidates = list(candidates)

    # STEP B: Compute BM25 and embedding scores for these candidates
    candidate_bm25_scores = [bm25_scores[idx] for idx in candidates]
    # For embeddings, we can do a dot product with the question embedding
    candidate_embs = summary_embeddings[candidates]
    dot_scores = (candidate_embs @ question_embedding.T).flatten()  # shape: (len(candidates),)

    # STEP C: Normalize both for a rough 0-1 range (optional but helpful)
    # min-max normalization for BM25
    bm25_min, bm25_max = min(candidate_bm25_scores), max(candidate_bm25_scores)
    bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1e-9
    norm_bm25_scores = [(s - bm25_min) / bm25_range for s in candidate_bm25_scores]

    # min-max normalization for dot product
    dot_min, dot_max = min(dot_scores), max(dot_scores)
    dot_range = dot_max - dot_min if dot_max != dot_min else 1e-9
    norm_dot_scores = [(s - dot_min) / dot_range for s in dot_scores]

    # STEP D: Weighted combination
    alpha = 0.5  # half BM25, half embeddings. Adjust as needed.
    combined_scores = [
        alpha * norm_bm25_scores[i] + (1 - alpha) * norm_dot_scores[i]
        for i in range(len(candidates))
    ]

    # STEP E: Sort candidates by combined score descending
    sorted_idx = np.argsort(combined_scores)[::-1]

    # ---------------------------
    # 10. Show Final Top 10
    # ---------------------------
    final_top_k = 10
    print(f"\nFinal Weighted Hybrid top {final_top_k}:")
    for rank_i, arr_idx in enumerate(sorted_idx[:final_top_k]):
        doc_idx = candidates[arr_idx]
        print(f"Rank {rank_i+1}, Combined Score: {combined_scores[arr_idx]:.4f} (BM25={bm25_scores[doc_idx]:.1f}, Dot={dot_scores[arr_idx]:.2f})")
        print(f"  > Key: {keys[doc_idx]}")
        print(f"  > Summary: {summaries[doc_idx]}")
        print("-" * 80)

    # ---------------------------
    # 11. Debug: Where Are 241, 811 Now?
    # ---------------------------
    if idx_241 is not None:
        try:
            pos_241 = candidates.index(idx_241)
            rank_241 = np.where(sorted_idx == pos_241)[0][0]  # rank in combined
            print(f"\nDoc 241 final rank: {rank_241 + 1} out of {len(candidates)}. Combined Score={combined_scores[pos_241]:.4f}")
        except ValueError:
            print("\nDoc 241 wasn't in final candidates somehow.")

    if idx_811 is not None:
        try:
            pos_811 = candidates.index(idx_811)
            rank_811 = np.where(sorted_idx == pos_811)[0][0]  # rank in combined
            print(f"Doc 811 final rank: {rank_811 + 1} out of {len(candidates)}. Combined Score={combined_scores[pos_811]:.4f}")
        except ValueError:
            print("Doc 811 wasn't in final candidates somehow.")

if __name__ == "__main__":
    main()
