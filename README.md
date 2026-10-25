# RAG Knowledge Base Assistant

This repository contains a local Retrieval-Augmented Generation (RAG) assistant for answering questions over a document knowledge base. The current implementation uses Gemini models for query rewriting and grounded answer generation, ChromaDB for vector storage, BM25 for lexical retrieval, and a cross-encoder reranker to improve the final context passed to the generator.

The code is structured as a research / prototype pipeline rather than a production application. It includes a web interface and REST API, a document ingestion pipeline, multiple retrieval strategies, reranking, conversational state management, and evaluation scripts for retrieval, generation, latency, and scope adherence.

---

## What this project does

- Parses and chunks PDFs before indexing
- Embeds chunks with a SentenceTransformer model
- Stores them in a persistent ChromaDB collection
- Rewrites user queries before retrieval
- Retrieves relevant chunks with one of several strategies:
  - `similarity`
  - `mmr`
  - `hybrid`
- Reranks candidates with a cross-encoder
- Grounds answers in retrieved context using Gemini
- Evaluates retrieval quality and generation behavior with DeepEval
- Measures retrieval latency, RAG-pipeline latency, LLM time-to-first-streamed-chunk (TTFT), and generation latency

## Retrieval pipeline

The retrieval layer is implemented in `src/retrieval_strategies.py` and `src/retriever.py`.

### Similarity search

- Uses the `BAAI/bge-base-en-v1.5` embedding model
- Normalizes embeddings
- Queries ChromaDB for nearest neighbors
- Optionally reranks the retrieved candidates with the cross-encoder

### MMR search

- Uses LangChain Chroma `max_marginal_relevance_search`
- Balances relevance and diversity when selecting chunks
- Then reranks the result set

### Hybrid search

Combines:

- dense vector similarity
- BM25 lexical retrieval
- reciprocal rank fusion (RRF)

The fused candidates are then passed through the cross-encoder reranker.

The active strategy is selected by the `RETRIEVAL_METHOD` environment variable:

```env
RETRIEVAL_METHOD=similarity

# or:
# RETRIEVAL_METHOD=mmr
# RETRIEVAL_METHOD=hybrid
```

---

## Project structure

```text
Rag/
├── app.py
├── README.md
├── requirements.txt
├── tools.py
├── utils.py
|
├── src/
│   ├── generator.py
│   ├── query_rewriter.py
│   ├── reranker.py
│   ├── retriever.py
│   └── retrieval_strategies.py
│
├── knowledgeBase/
│   └── collection.py
│
├── operations/
│   ├── chunker.py
│   ├── embedding.py
│   └── parser.py
│
├── evals/
│   ├── eval_generator.py
│   ├── eval_latency.py
│   ├── eval_pipeline.py
│   ├── eval_retriever.py
│   ├── eval_scope.py
│   └── eval_toxicity.py
│
├── results/
│   ├── retriever_results.json
│   ├── scope_results.json
│   ├── toxicity_results.json
│   ├── prompt_v1/
│   │   └── pipeline_results.json
│   └── prompt_v2/
│       └── pipeline_results.json
│       prompt_v3/
│       └── pipeline_results.json
|
├── test_cases/
│   ├── generator_evalset.json
│   ├── pipeline_eval.json
│   ├── retriever_eval.py
│   ├── scope_evalset.json
│   └── toxicity_eval.json
│
├── vector_db/
│   ├── buildDB.py
│
│
└── chroma_db/
```

The local virtual environment and `.env` file are not included in the repository tree because they contain environment-specific files and configuration.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv rag_env
source rag_env/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv rag_env
.\rag_env\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The dependency set is currently centered on:

- `google-genai`
- `python-dotenv`
- `chromadb`
- `sentence-transformers`
- `transformers`
- `torch`
- `pymupdf`
- `langchain-chroma`
- `langchain-huggingface`
- `deepeval`

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
ENABLE_TIMING=False
RETRIEVAL_METHOD=similarity
```

Notes:

- `GEMINI_API_KEY` is required for the query rewriter and generator.
- `ENABLE_TIMING=True` enables the timing context manager in `utils.py`.
- `RETRIEVAL_METHOD` must be one of:
  - `similarity`
  - `mmr`
  - `hybrid`

---

## Build the knowledge base

The project expects a PDF knowledge base to be indexed into ChromaDB. The indexing script is in `vector_db/buildDB.py`, and it currently sets a PDF path manually:

```python
PDF_PATH = "XYZ_2023.pdf"
```

Update that value to the actual file you want to index, then run:

```bash
python vector_db/buildDB.py
```

This script:

1. loads the PDF
2. trims front/back matter if configured
3. splits the content into chunks
4. removes very short or citation-heavy chunks
5. creates embeddings
6. resets the Chroma collection
7. stores the chunks and metadata

The document ingestion pipeline itself is defined in:

- `operations/parser.py`
- `operations/chunker.py`
- `operations/embedding.py`
- `knowledgeBase/collection.py`

---

## Run the assistant

Start the command-line app:

```bash
python app.py
```

Example interaction:

```text
You: What does the knowledge base say about online learning?

Assistant: ...
```

The current app loop:

- appends the user message to the conversation history
- sends it to Gemini with a system prompt and tool schema
- calls `retrieve_documents` when the model decides retrieval is needed
- appends tool results back into the conversation
- continues until the model provides a final answer or a maximum iteration limit is reached

---

## Tooling and prompt behavior

The runtime planner in `app.py` uses a Gemini model with a tool-calling prompt. The tool schema is declared in `tools.py` and currently includes:

- `retrieve_documents(query: str)`

This is the tool that triggers the retrieval stack.

---

## Generator behavior

The answer generation logic lives in `src/generator.py`.

`generate_answer()`:

- joins the retrieved chunks into a context block
- inserts them into a strict context-grounded prompt
- calls Gemini with `temperature=0.0`
- returns the textual answer

The prompt enforces a grounded-answer policy:

- answer only from the provided context
- do not invent missing facts
- do not answer outside the knowledge-base scope
- maintain a professional tone

---

## Query rewriting

Before retrieval, `src/retriever.py` calls `rewrite_query()` from `src/query_rewriter.py`.

This step:

- rewrites the original question into a retrieval-optimized query
- keeps domain-specific terminology
- removes unnecessary filler
- tries to preserve the user's intent while improving search quality

---

## Evaluation suite

This repository includes evaluation scripts under `evals/`:

- `eval_retriever.py`: retrieval metrics using DeepEval
- `eval_generator.py`: answer quality checks against ideal context
- `eval_pipeline.py`: end-to-end RAG pipeline evaluation
- `eval_scope.py`: scope adherence evaluation
- `eval_toxicity.py`: toxicity evaluation
- `eval_latency.py`: latency benchmarking

The evaluation outputs are saved under `results/`.

---

## Baseline evaluation results

The following results were obtained from the saved evaluation runs and represent baseline measurements for the current implementation.

The evaluation threshold was **0.70**.

### Retriever(Without Reranker)

- Contextual Precision: **0.736**
- Contextual Recall: **0.858**

### Retriever(With Reranker)

- Contextual Precision: **0.775**
- Contextual Recall: **0.816**

Reranking improved Contextual Precision (0.736 → 0.775) but reduced Contextual Recall (0.858 → 0.816) — the classic cross-encoder reranking trade-off: narrowing to the most relevant top-k chunks improves ranking quality at some cost to overall coverage. "With Reranker" reflects the current production configuration. Both configurations now pass the 0.70 threshold; an earlier baseline measurement of Contextual Precision (0.664, below threshold) was taken before chunking fixes — removing index/glossary pages and citation-list blocks that were polluting the retrieved context — and no longer reflects the current implementation.

### Safety

- Scope adherence: **0.910**
- Toxicity: **0.000**

### Generator

- Answer Relevancy: **0.983**
- Faithfulness: **1.000**

### RAG pipeline — prompt_v1

- Answer Relevancy: **0.875**
- Faithfulness: **1.000**
- Correctness: **0.850**
- Completeness: **0.765**

### RAG pipeline — prompt_v2

- Answer Relevancy: **0.753**
- Faithfulness: **1.000**
- Correctness: **0.755**
- Completeness: **0.720**

### RAG pipeline — prompt_v3

- Answer Relevancy: **0.911**
- Faithfulness: **1.000**
- Correctness: **0.850**
- Completeness: **0.805**

**Prompt ablation:** Prompt V2 added explicit instructions for multi-part question coverage, relevance control, unsupported-inference prevention, scope handling, safety, and prompt-injection resistance. On the same 20-question evaluation set, V2 reduced Answer Relevancy (0.875 → 0.753), Correctness (0.850 → 0.755), and Completeness (0.765 → 0.720), while Faithfulness stayed unchanged at 1.0. The likely cause was a single instruction telling the model not to add unrelated information or repeat itself — a brevity constraint that plausibly also licensed trimming content it should have kept.

Prompt V3 tested this directly: every V2 rule was kept unchanged except that one instruction, reworded to include all relevant context detail even at greater length, only omitting genuinely unrelated information or exact restatement. V3 recovered fully and exceeded the V1 baseline: Answer Relevancy 0.911, Correctness 0.850, Completeness 0.805, Faithfulness unchanged at 1.0. This confirms the regression was caused by that specific instruction rather than by the added safety and scope constraints as a whole, and V3 is retained as the current baseline prompt, combining V2's safety/scope properties with quality that matches or exceeds V1.

Two questions in the V3 run (Gaussian mixture model responsibilities during EM; why soft voting weights confident classifiers more than hard voting) scored 0.0 on Answer Relevancy, Correctness, and Completeness while Faithfulness remained 1.0 — indicating an honest refusal rather than a hallucination, caused by the retriever failing to surface relevant content rather than a generation-quality issue. Excluding these two retrieval-gap cases, the remaining 18 questions average 0.956 / 1.000 / 0.944 / 0.894. The Gaussian mixture model gap has now appeared independently across two separate evaluation sets, suggesting a persistent retrieval issue for that specific topic rather than a one-off miss; investigating this is a planned next step, separate from prompt tuning.

The baseline evaluation shows particularly strong faithfulness and toxicity results, and retrieval precision now passes threshold following chunking fixes made since the initial baseline. The remaining weaknesses are Contextual Recall under reranking, a specific, reproducible retrieval gap on Gaussian mixture model content, and answer completeness on harder or more ambiguous questions — the latter substantially improved by prompt V3.

````

## Latency Profiling

The latency benchmark measures the RAG pipeline independently from the planner-level application loop.

Run it with:

```bash
python -m evals.eval_latency
````

### Baseline latency results

Example benchmark results:

```text
LATENCY (milliseconds)

| Run |  E2E P95 | Retrieval P95 | Generation P95 | TTFT P95 | Avg answer |
| --- | -------: | ------------: | -------------: | -------: | ---------: |
| 1   | 9,965 ms |      8,769 ms |       1,351 ms | 1,347 ms |  247 chars |
| 2   | 8,358 ms |      6,850 ms |       1,529 ms | 1,364 ms |  604 chars |
| 3   | 4,894 ms |      3,723 ms |       1,179 ms | 1,177 ms |   47 chars |
| 4   | 4,777 ms |      3,711 ms |       1,102 ms | 1,021 ms |  461 chars |

These values were obtained from a small initial benchmark and should be treated as baseline measurements rather than statistically robust latency estimates.
```

### Latency definitions

- **Retrieval latency:** Time spent in the retrieval stage measured by the benchmarked retrieval function.
- **TTFT:** Time from the start of the LLM generation request until the first non-empty streamed chunk is received.
- **Generation latency:** Time from the start of the LLM generation request until the complete streamed response is received.
- **RAG pipeline latency:** Time from the start of retrieval until the complete generated response is received.

The benchmark's RAG pipeline latency does **not necessarily represent the complete user-request latency of the `app.py` planner loop**, because planner/model tool-selection latency can occur before retrieval.

### Baseline SLOs

The benchmark currently uses:

- RAG pipeline P95 ≤ **3000 ms**
- TTFT P95 ≤ **1200 ms**

Based on the baseline results:

```text
RAG pipeline P95: ~4777 ms → FAIL
TTFT P95:          ~1021 ms → PASS
```

The latest baseline indicates that retrieval is the primary latency bottleneck, accounting for approximately 3.7 seconds of the roughly 4.8-second RAG pipeline latency.

---

### Improved Latency

---

end-to-end | mean= 3292.5 p50= 3372.8 p95= 3668.1 p99= 3822.8 min= 2878.3 max= 3861.4

ttft | mean= 923.9 p50= 917.7 p95= 995.0 p99= 1001.0 min= 866.6 max= 1002.6

retrieval | mean= 2227.1 p50= 2158.6 p95= 2618.6 p99= 2777.8 min= 1953.8 max= 2817.6

generation | mean= 1065.3 p50= 1056.2 p95= 1236.8 p99= 1281.9 min= 883.4 max= 1293.2

---

avg answer length: 540 chars (latency varies with output length)

SLO: full answer p95 <= 3000 ms -> p95 = 3668 ms [FAIL]

SLO: first token (perceived) p95 <= 1200 ms -> p95 = 995 ms [PASS]

Reducing candidate_k from 15 to 10 substantially reduced retrieval latency, bringing RAG pipeline P95 down to ~3.66 seconds. TTFT remains within the target SLO at 995 milliseconds P95. Full-answer latency is slightly above the 3-second target and can be considered a future optimization rather than an immediate priority.

## Notes

- Some older documentation may describe capabilities that are no longer implemented exactly; this README reflects the current implementation.
- Evaluation scores and latency measurements are benchmark results for the current datasets/configuration and are not guarantees of production performance.
- The latency benchmark measures the RAG retrieval-and-generation path separately from planner-level latency in the command-line application.

---

## Quick start

```bash
python -m venv rag_env
source rag_env/bin/activate

pip install -r requirements.txt

# create a .env file with GEMINI_API_KEY and RETRIEVAL_METHOD

python vector_db/buildDB.py
python app.py
```

If you do not already have a `.env`, create one manually with the variables shown in the Setup section.

---

## Current status

This repository is a working local RAG prototype for grounded document QA, with indexing, retrieval, generation, and evaluation code present in the current project.

The current implementation includes:

- PDF parsing and chunking
- SentenceTransformer embeddings
- persistent ChromaDB storage
- query rewriting
- dense, MMR, and hybrid retrieval strategies
- BM25 lexical retrieval
- reciprocal rank fusion
- cross-encoder reranking
- Gemini-based grounded generation
- DeepEval-based evaluation
- latency benchmarking
- scope and toxicity evaluation

---

## Current Limitations

- Conversation memory exists only during the current session
- The latency benchmark is based on a small number of runs and should be expanded for more reliable percentile estimates

---

## Future Improvements

Potential future enhancements include:

- Docker support
- Long-term conversational memory
- Further retrieval and reranking latency optimization

---

## Tech Stack

- Python
- FastAPI
- Google Gemini
- ChromaDB
- Sentence Transformers
- `BAAI/bge-base-en-v1.5`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- BM25
- LangChain (retrieval utilities only)
- DeepEval
- PyMuPDF
