# ElectroRAG GUI

Streamlit front end connected to the ElectroRAG core in the parent project directory.
Run these commands from this folder:

    python -m pip install -r requirements-gui.txt
    .\run_gui.ps1                 # live mode (real retrieval and generation)
    .\run_gui.ps1 -Mode mock      # fabricated demo data for UI testing

The live adapter loads the existing read-only ChromaDB index from `data/chroma_db`,
uses PDFs from `data/datasheets`, and sends generation requests to local Ollama.
Heavy models and the BM25 index load only when the first chat question is submitted.
Settings and optional path overrides are in `gui/config.py`.

To add datasheets, open **Datasheet Library**, choose one or more PDFs, then select
**Upload and index**. The PDFs are saved under `data/datasheets` and their extracted
text is added to the local ChromaDB index. Duplicate filenames are rejected.
