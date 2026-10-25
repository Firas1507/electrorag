# ElectroRAG

ElectroRAG is a local Retrieval-Augmented Generation (RAG) assistant for asking grounded questions about electronics datasheets. It combines semantic retrieval, BM25 keyword retrieval, cross-encoder reranking, and deterministic page-level citations before sending the selected evidence to a local Ollama model.

## Highlights

- Local-first workflow: datasheets, embeddings, retrieval, and generation stay on your machine.
- Hybrid retrieval with query expansion and cross-encoder reranking.
- Page and chunk metadata preserved through the pipeline for traceable answers.
- Optional Streamlit GUI for chat, datasheet upload, indexing, and diagnostics.
- Testable core modules for loading PDFs, chunking, retrieval, generation, and citation validation.

## Architecture

```text
PDF datasheets
      │
      ▼
load → chunk → embeddings + BM25 → hybrid retrieval
                                      │
                                      ▼
                              cross-encoder reranking
                                      │
                                      ▼
                              local Ollama generation
                                      │
                                      ▼
                            resolved page citations
```

## Requirements

- Python 3.10+
- A local [Ollama](https://ollama.com/) installation and a model configured for generation
- PDF datasheets placed in `data/datasheets/`
- Enough disk space for the embedding/reranker models and local vector index

## Installation

```bash
git clone https://github.com/<your-user>/electrorag.git
cd electrorag
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

Copy your PDF datasheets into `data/datasheets/`. The vector database and other generated artifacts are intentionally ignored by Git.

## Ask a question from the command line

```bash
python -m src.rag "What is the input voltage range of the TPS63020?"
```

The first run may download the configured sentence-transformer and reranker models. Make sure Ollama is running before asking questions.

## Run the GUI

```bash
pip install -r ElectroRAG_GUI/requirements-gui.txt
streamlit run ElectroRAG_GUI/gui_app.py
```

For a mock UI run that does not require the retrieval backend:

```powershell
.\ElectroRAG_GUI\run_gui.ps1 -Mode mock
```

See [ElectroRAG_GUI/README.md](ElectroRAG_GUI/README.md) for upload and indexing details.

## Tests

```bash
pytest
```

## Project layout

| Path | Purpose |
| --- | --- |
| `src/` | Retrieval, reranking, generation, citation, and evaluation pipeline |
| `electro_rag/` | Package entry point |
| `ElectroRAG_GUI/` | Streamlit front end |
| `tests/` | Automated tests |
| `data/` | Local datasheets and generated indexes; user data is not committed |

## License

MIT. See [LICENSE](LICENSE).
