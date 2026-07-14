# GraphRAG for Open-Domain Question Answering

This is my University of Bonn lab project on graph-based retrieval-augmented generation. I built an end-to-end pipeline that converts unstructured text into a knowledge graph, groups related entities into communities, retrieves relevant community summaries, and generates grounded answers to questions.

The project uses the SQuAD dataset as its test bed. My main goal was to understand whether explicit entity relationships and community-level context can improve retrieval compared with treating every passage as an isolated document.

## Project summary

- 10,570 SQuAD validation questions consolidated into 2,067 unique contexts
- 11,684 entities and 14,340 relationships in the saved validation graph
- 6,288 saved community-summary records
- Semantic, dense, and hybrid retrieval strategies
- BLEU, BERTScore, and ROUGE evaluation
- Checkpointed graph construction for long-running GPU jobs

## Pipeline

The system follows this sequence:

```text
SQuAD contexts
    -> overlapping text chunks
    -> entity types, entities, and relationships
    -> entity summaries
    -> NetworkX knowledge graph
    -> Louvain communities
    -> community summaries
    -> semantic, dense, or hybrid retrieval
    -> chunk-level candidate answers
    -> one global answer
```

### Dataset preparation

The validation split is loaded from Hugging Face and grouped by context. This reduces duplicate processing because several SQuAD questions can refer to the same passage. Long passages are divided into overlapping chunks so that extraction remains within the model context window.

### Knowledge graph construction

I used Llama 3.2 3B Instruct for several structured extraction tasks:

1. Identify entity types relevant to the current subject.
2. Extract entities and weighted relationships.
3. Consolidate descriptions into entity summaries.
4. Produce summaries for groups of related entities.

Entities are stored as NetworkX nodes with type and description attributes. Relationships are stored as edges with a natural-language description and strength value.

### Community detection and checkpointing

Each local graph is merged into a global graph. Louvain community detection is applied to the affected portion, and summaries are regenerated only for communities changed by the new data.

Graph construction is expensive because it requires several model calls for every context. The pipeline therefore saves its graph, summaries, and current dataset position after each iteration. An interrupted run can continue from its last checkpoint.

### Query-time retrieval and generation

The query-time pipeline I implemented connects the pre-computed graph summaries to each user's question. It consists of five stages:

1. **Load community summaries.** The summaries created during graph indexing are loaded into memory and organized for retrieval.
2. **Filter relevant summaries.** The question and every summary are embedded with SentenceTransformer's `all-MiniLM-L6-v2` model. Cosine similarity is used to select the top-N summaries most closely related to the question.
3. **Prepare context chunks.** Retrieved summaries are divided into manageable chunks so that the relevant content fits within the language model's context window without silently truncating the input.
4. **Generate intermediate answers.** Each chunk is processed with a query-focused prompt that instructs the model to rely only on the supplied context. The response contains both an answer and a relevance score from 0 to 100.
5. **Reduce to a global answer.** Zero- and low-scoring candidates are discarded. The remaining answers are ranked, combined within a token budget, and passed through the model again to produce one concise final response.

This map-and-reduce design allowed the system to work with more retrieved context than could fit into a single prompt while keeping the final answer focused on the original question.

### Retrieval experiments and generation tuning

I initially evaluated FAISS for similarity search. For the shorter, structured collection of community summaries, direct SentenceTransformer similarity offered sufficient speed and retrieval quality with less indexing complexity. I later extended the experiments to compare:

- SentenceTransformer embeddings with cosine similarity
- FAISS dense-vector search
- A weighted hybrid of BM25 lexical scores and FAISS semantic scores

In the hybrid version, a broad candidate set is retrieved from both BM25 and FAISS. Their scores are normalized before being combined into a single ranking.

For answer generation, I used Llama 3.2 3B Instruct and tuned the decoding settings to favor stable, relevant responses:

- Beam search with 8 beams
- Temperature of 0.4
- Repetition penalty of 1.1

Beam search explored several candidate sequences, while the lower temperature reduced randomness and made repeated evaluations more consistent.

## Technologies

- Python, pandas, and NumPy
- PyTorch and Hugging Face Transformers
- NetworkX and Louvain community detection
- SentenceTransformers and FAISS
- BM25 through `rank-bm25`
- PyVis and Matplotlib
- BLEU, BERTScore, and ROUGE

## Repository structure

```text
graphrag/
|-- pipeline/                 Core graph-construction pipeline
|   |-- preprocessing/       Chunking, extraction, and summarization
|   |-- generation/          Answer-generation implementation
|   |-- evaluation/          Evaluation utilities
|   `-- utils/               Prompts, data loading, and graph utilities
|-- RAG/                      Retrieval and answer-generation experiments
|-- experiments/              Earlier standalone prototypes
|   |-- generation/
|   `-- similarity/
|-- output/                   Validation graph, summaries, and results
|-- scripts/                  Environment helper scripts
|-- data_exploration.ipynb    Dataset and graph exploration
|-- evaluate_results.py       Evaluation entry point
|-- prompt_result.txt         Saved prompt output from early development
`-- main.py                   Graph-pipeline entry point
```

The experimental scripts are kept separate from the core pipeline so that the development path remains visible without obscuring the main implementation.

## Saved results

The repository includes the main validation artifacts, so the graph and generated answers can be inspected without rebuilding the full dataset:

- [`output/graph_valid.gexf`](output/graph_valid.gexf): knowledge graph with 11,684 nodes and 14,340 edges
- [`output/graph_valid.html`](output/graph_valid.html): interactive PyVis visualization
- [`output/community_summaries_valid.json`](output/community_summaries_valid.json): community-level retrieval context
- [`output/query_results.json`](output/query_results.json): questions, relevance scores, and generated answers
- [`RAG/metrics_results.txt`](RAG/metrics_results.txt): recorded evaluation metrics
- [`data_exploration.ipynb`](data_exploration.ipynb): dataset and graph exploration

GitHub may not preview the larger GEXF, HTML, JSON, or notebook files. They can still be downloaded and opened locally with Gephi, a browser, or Jupyter.

## Running the project

Python 3.10 or newer and a CUDA-capable GPU are recommended. Access to the gated Llama model on Hugging Face is required.

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate it with `.venv\Scripts\activate`.

The original requirements file records the base lab environment. The complete workflow also uses:

```bash
python -m pip install torch transformers sentence-transformers networkx python-louvain python-dotenv faiss-cpu rank-bm25 evaluate bert-score rouge-score nltk pyvis matplotlib
```

Create a `.env` file in the project root:

```dotenv
HUGGINGFACE_TOKEN_LLAMA32=your_huggingface_token
GRAPH_FILE=output/graph_valid.gexf
COMMUNITIES_FILE=output/community_summaries_valid.json
STATUS_FILE=output/status_valid.json
```

Then start or resume graph construction:

```bash
python main.py
```

The standalone scripts preserve some paths and CUDA settings from the original university GPU environment. Those values may need to be adjusted when reproducing the experiments on another machine.

## Evaluation and findings

The query-time evaluation documented in my January 2025 report used BLEU, BERTScore, and ROUGE to compare generated answers with the SQuAD references:

- BLEU-1: 0.1503
- BERTScore precision: 0.1354
- BERTScore recall: 0.1004
- BERTScore F1: 0.1150
- ROUGE-1 F1: 0.1148
- ROUGE-2 F1: 0.0489
- ROUGE-L F1: 0.1103

The evaluation produced two useful findings. First, semantic filtering reduced irrelevant content, and the chunking stage processed the selected context efficiently within the model's input limit. Second, downstream answer quality depended heavily on the information preserved in the community summaries. If a relevant detail was absent or a sparse summary ranked too highly, the generation stage could not construct a complete answer.

The report also showed that the original prompts handled direct questions better than nuanced questions requiring several pieces of context. These observations gave me a clear diagnosis across the pipeline rather than treating generation quality as an isolated model problem: graph construction affected summary quality, summary quality affected retrieval, and retrieval determined what evidence was available for the final answer.

## Project outcome

The January 2025 report captured my initial query-time implementation and its first full evaluation. I subsequently took ownership of the complete pipeline and extended the final study with:

- Entity normalization across aliases, casing, and generated entity types
- Constrained structured generation in place of regex-only response parsing
- Cached extraction results and summary embeddings for repeated experiments
- Retrieval evaluation with Recall@k and Mean Reciprocal Rank
- A direct comparison with a text-only RAG baseline on the same question set

