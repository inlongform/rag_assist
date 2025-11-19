# Isaac Sim / Isaac Lab RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) assistant for Isaac Sim and Isaac Lab, using:

- Qdrant for vector storage
- Nomic embeddings for text
- Ollama LLMs for code generation
- Gradio for the web UI

---

## 1. Set up the environment

Make sure you have **conda** installed.

#### a) Create conda environment with top-level packages:

```bash
conda env create -f environment.yml
conda activate your_env_name

pip install -r requirements.txt

```
## 2. install ollama

#### b) Install ollama
  - install from https://ollama.com/
```bash

ollama pull nomic-embed-text:latest
ollama pull qwen3-coder:latest 
ollama pull qwen2.5-coder:7b 

```
#### c) Run crawler
  - crawl locally served files (these files have to be compiled and run locally from repo)
    ```bash
      python deep_crawler_local.py
    ```
  - recursive crawl web
    ```bash
      python deep_crawler_web_v2.py
    ```
  - crawl local examples
    ```bash
      python crawl_examples.py
    ```


#### d) Prepare files
  - Combine examples into 1 dir
    ```bash
      python combine_files.py
    ```
  - Clean and chunk docs (clean_line_nums.py removes line nums from some examples)
    ```bash
      python 1_deep_crawl_clean_updated.py
    ```
  - Extract and prepare symbols
    ```bash
      python combine_symbols.py
      python clean_symbols.py
    ```
  - Start qDrant in docker
    ```bash
      docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
    ```
  - Ingest chunks to qDrant
    ```bash
      python ingest.py
    ```
  - Verify ingested docs
    ```bash
      python check_count.py
    ```
  - Run Sample Query
    ```bash
      python test_query.py
    ```


#### e) Launch
```bash
# start qdrant
ollama run

python ./app/app.py

```

#### f) 

  - navigate to http://127.0.0.1:7860
  - YAY!!!  