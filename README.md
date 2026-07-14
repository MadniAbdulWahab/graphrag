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
    -> BM25 and FAISS retrieval
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

### Retrieval and generation

I explored three retrieval approaches:

- SentenceTransformer embeddings with semantic similarity
- FAISS dense-vector search
- A weighted hybrid of BM25 lexical scores and FAISS semantic scores

For hybrid retrieval, both score ranges are normalized before they are combined. The best community summaries are divided into smaller chunks and passed to the language model. Each candidate answer receives a relevance score, and the strongest candidates are reduced into one concise answer.

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

## Evaluation snapshot

The recorded end-to-end run produced these baseline scores:

- BLEU-1: 0.1503
- BERTScore F1: 0.1150
- ROUGE-1 F1: 0.1148
- ROUGE-2 F1: 0.0489
- ROUGE-L F1: 0.1103

I view these as diagnostic results rather than a final benchmark. The experiment showed that entity normalization and retrieval quality strongly affect the generated answer. Inconsistent names or types fragment the graph, which produces weaker communities and less useful retrieval context.

## Next steps

- Normalize aliases, casing, and entity types before graph insertion
- Replace regex response parsing with constrained structured generation
- Cache extraction results and summary embeddings
- Add Recall@k and MRR for retrieval evaluation
- Compare against a text-only RAG baseline on the same questions
- Move experiment settings into a central configuration file
- Add automated tests for each pipeline stage
