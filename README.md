# GraphRAG for Open-Domain Question Answering

> A University of Bonn lab project exploring how knowledge graphs and community-level retrieval can improve question answering over a large text collection.

This repository contains my implementation of a GraphRAG pipeline built on the SQuAD dataset. Instead of treating every passage as an isolated document, the pipeline extracts entities and relationships, connects them in a knowledge graph, groups related entities into communities, and uses those community summaries as retrieval context for an instruction-tuned language model.

The project covers the full experimental workflow: data preparation, prompt design, graph construction, community detection, dense and sparse retrieval, answer generation, evaluation, and graph visualization.

## Project at a glance

- **10,570** SQuAD validation questions consolidated into **2,067 unique contexts**
- **11,684 entities** and **14,340 relationships** in the saved validation graph
- **6,288 community-summary records** generated for retrieval
- Three retrieval strategies explored: semantic similarity, FAISS dense search, and weighted BM25 + FAISS retrieval
- Answer quality evaluated with BLEU, BERTScore, and ROUGE
- Long-running graph construction can resume from a saved checkpoint

## How it works

```mermaid
flowchart LR
    A[SQuAD validation set] --> B[Merge questions by context]
    B --> C[Overlapping text chunks]
    C --> D[Entity types, entities and relations]
    D --> E[Entity summaries]
    E --> F[NetworkX knowledge graph]
    F --> G[Louvain communities]
    G --> H[Community summaries]
    H --> I[BM25 + FAISS retrieval]
    I --> J[Chunk-level answers and relevance scores]
    J --> K[Single global answer]
    K --> L[BLEU, BERTScore and ROUGE]
```

### 1. Dataset preparation

The pipeline loads the SQuAD validation split from Hugging Face and groups rows that share the same context. Each context is divided into overlapping, sentence-aware chunks so that extraction remains within the model's context window while retaining information around chunk boundaries.

### 2. Knowledge graph construction

For each context, Llama 3.2 3B Instruct is prompted to:

1. infer entity types appropriate for the current domain;
2. extract entities and weighted relationships in a structured format;
3. consolidate descriptions into an entity-level summary; and
4. create a domain-specific persona to guide summarization.

Entities become graph nodes with type and description attributes. Relationships become weighted edges carrying a natural-language explanation of the connection.

### 3. Incremental community detection

Local graphs are composed into a global NetworkX graph. Louvain community detection is applied to newly affected parts of the graph, and only affected community summaries are regenerated. The current dataset iteration, graph, and summaries are saved after every pass, allowing an interrupted GPU job to continue instead of starting over.

### 4. Retrieval and answer generation

The repository contains several retrieval experiments:

- **Semantic retrieval:** SentenceTransformer embeddings and cosine similarity
- **Dense retrieval:** FAISS indexes over community-summary embeddings
- **Hybrid retrieval:** a weighted combination of normalized BM25 and FAISS scores

The hybrid pipeline retrieves a broad candidate set from both indexes, normalizes the lexical and semantic scores, and ranks the combined results. Retrieved summaries are split into manageable chunks, answered independently, and assigned relevance scores. The highest-scoring intermediate answers are then reduced into one concise global answer.

### 5. Evaluation and visualization

Generated answers are compared with SQuAD references using BLEU, BERTScore, and ROUGE. The graph can be inspected as GEXF in tools such as Gephi, or through the included PyVis HTML visualization.

## Engineering highlights

- Designed multi-stage prompts for entity typing, relation extraction, persona generation, and hierarchical summarization
- Added parsing, fallback values, and retry logic around non-deterministic model output
- Represented graph metadata in a portable GEXF format
- Implemented resumable processing for long-running model inference jobs
- Compared lexical and semantic retrieval instead of relying on a single similarity method
- Used candidate-set fusion and min-max score normalization for hybrid ranking
- Built a map-reduce-style answer generation flow for context that does not fit in one prompt
- Kept intermediate artifacts so that graph construction, retrieval, and evaluation can be inspected independently

## Repository structure

| Path | Purpose |
| --- | --- |
| `pipeline/run_pipeline.py` | End-to-end dataset-to-graph pipeline and checkpoint handling |
| `pipeline/preprocessing/` | Chunking, entity and relation extraction, graph creation, and summaries |
| `pipeline/utils/prompts.py` | Prompt templates used throughout graph construction |
| `pipeline/utils/community_detection/` | Leiden/community quality utilities explored during development |
| `RAG/` | Semantic, FAISS, and hybrid retrieval experiments plus evaluation scripts |
| `similarity/` | Standalone hybrid-search prototype for inspecting retrieval rankings |
| `generation/` | Earlier answer-generation prototypes |
| `output/` | Saved graph, summaries, query results, checkpoints, and visualization |
| `data_exploration.ipynb` | Dataset exploration and graph-analysis notebook |

## Saved artifacts

The repository includes outputs from the validation-set experiment, so the result can be inspected without rebuilding the full graph.

| Artifact | Description |
| --- | --- |
| [`output/graph_valid.gexf`](output/graph_valid.gexf) | Knowledge graph with 11,684 nodes and 14,340 edges |
| [`output/graph_valid.html`](output/graph_valid.html) | Interactive PyVis graph visualization |
| [`output/community_summaries_valid.json`](output/community_summaries_valid.json) | Community-level retrieval context |
| [`output/query_results.json`](output/query_results.json) | Questions, chunk relevance scores, and generated answers |
| [`RAG/metrics_results.txt`](RAG/metrics_results.txt) | BLEU, BERTScore, and ROUGE results from an experimental run |
| [`data_exploration.ipynb`](data_exploration.ipynb) | Dataset statistics and graph exploration |

## Reproducing the experiment

The original runs were carried out on university GPU infrastructure. A CUDA-capable GPU is strongly recommended because graph construction performs several Llama inference calls per context. Access to the gated Llama model on Hugging Face is also required.

### 1. Create an environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The checked-in requirements file records the base lab environment. The graph, retrieval, generation, evaluation, and visualization scripts additionally use the following packages:

```bash
python -m pip install torch transformers sentence-transformers networkx python-louvain python-dotenv faiss-cpu rank-bm25 evaluate bert-score rouge-score nltk pyvis matplotlib
```

For a CUDA-enabled FAISS or PyTorch installation, use versions compatible with the local CUDA toolkit.

### 2. Configure the graph pipeline

Create a `.env` file in the project root:

```dotenv
HUGGINGFACE_TOKEN_LLAMA32=your_huggingface_token
GRAPH_FILE=output/graph_valid.gexf
COMMUNITIES_FILE=output/community_summaries_valid.json
STATUS_FILE=output/status_valid.json
```

The `.env` file is ignored by Git. File paths and CUDA device assignments in the standalone scripts under `RAG/` reflect the original lab environment and may need to be adjusted before running them elsewhere.

### 3. Build or resume the graph

```bash
python main.py
```

`main.py` starts the pipeline with checkpoint loading enabled. If the graph, summary, and status files exist, processing resumes from the saved iteration. To rebuild from scratch, call `run_pipeline(load=False)` and point the environment variables to new output files.

## Experimental results

The recorded end-to-end run produced the following generation metrics:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.1503 |
| BERTScore F1 | 0.1150 |
| ROUGE-1 F1 | 0.1148 |
| ROUGE-2 F1 | 0.0489 |
| ROUGE-L F1 | 0.1103 |

I treat these as baseline results rather than a final benchmark. The experiment showed that building a large graph is only one part of the problem: entity normalization, retrieval quality, and strict answer grounding have a major effect on downstream performance.

## What I would improve next

- Normalize aliases, casing, and entity types before inserting nodes into the global graph
- Replace regex-based response parsing with constrained structured generation
- Cache entity extraction and summary embeddings to shorten repeated experiments
- Add retrieval metrics such as Recall@k and MRR before evaluating answer generation
- Compare the graph pipeline against a text-only RAG baseline on the same question set
- Move experiment settings into a single configuration file and add automated tests for each pipeline stage

The most useful part of this project was seeing how data quality propagates through an end-to-end retrieval system. Small extraction inconsistencies create fragmented communities, which then affect retrieval and finally the generated answer. That made the project as much about careful pipeline design and evaluation as it was about language models.
