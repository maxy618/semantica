# Semantica

**Semantica** is a powerful, local command-line semantic search engine designed for developers. It allows you to search through your codebase, documents, and notes using natural language queries instead of exact keyword matching.

Built with performance and privacy in mind, Semantica runs entirely locally using state-of-the-art embedding models and vector search.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-Nuitka-yellow)

## ✨ Features

*   **Semantic Search:** Finds relevant code and text based on meaning, not just syntax.
*   **Privacy First:** Runs 100% locally. No data leaves your machine.
*   **Multi-Format Support:**
    *   **Code:** Python, JS, TypeScript, Go, Rust, C++, Java, SQL, etc.
    *   **Docs:** PDF, DOCX, Markdown, TXT, Log files.
*   **High Performance:**
    *   Uses **FastEmbed** for lightweight, fast inference.
    *   **FAISS** for efficient vector similarity search.
    *   Multiprocessing file scanner.
*   **Reranking:** Optional Cross-Encoder reranking step for high-precision results.
*   **Smart Caching:** Caches embeddings based on file hash and model parameters to speed up subsequent searches.
*   **Standalone Binary:** Available as a portable `.exe` (no Python environment required).

---

## 🚀 Installation

### Option 1: Download Release (Recommended)
Download the standalone executable `semantica.exe` from the **[Releases](https://github.com/maxy618/semantica/releases)** page.

Add it to your system PATH or run it directly:
```powershell
.\semantica.exe -p "C:\MyProject" -q "how to connect to database"
```

### Option 2: Run from Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/maxy618/semantica.git
    cd semantica
    ```

2.  **Set up the environment:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    *(Create a `requirements.txt` based on the imports or install manually)*
    ```bash
    pip install fastembed faiss-cpu numpy pypdf python-docx termcolor psutil nuitka spacy
    
    # Install the multi-language model for Spacy
    pip install https://github.com/explosion/spacy-models/releases/download/xx_sent_ud_sm-3.7.0/xx_sent_ud_sm-3.7.0-py3-none-any.whl
    ```

---

## 🛠 Building the Executable

To generate a standalone `.exe` that includes all dependencies (including machine learning models and tokenizers), use the following **Nuitka** command.

**Note:** This command ensures that `fastembed`, `faiss`, and `spacy` models are correctly bundled.

```powershell
python -m nuitka --standalone --onefile --mingw64 `
    --include-package=src `
    --include-package=fastembed `
    --include-package=faiss `
    --include-package=pypdf `
    --include-package=docx `
    --include-package=xx_sent_ud_sm `
    --include-package-data=xx_sent_ud_sm `
    --spacy-language-model=xx_sent_ud_sm `
    --include-package=tokenizers `
    --include-package=onnxruntime `
    --nofollow-import-to=matplotlib `
    --nofollow-import-to=pandas `
    --nofollow-import-to=scipy `
    --nofollow-import-to=IPython `
    --nofollow-import-to=pytest `
    --nofollow-import-to=tkinter `
    --output-filename=semantica.exe `
    src/main.py
```

---

## 📖 Usage

Run Semantica via the CLI. The tool will scan the directory, chunk the files, vectorise them (saving to cache), and perform the search.

### Basic Search
```powershell
semantica -p D:\projects\my-app -q "authentication middleware"
```

### Search with Reranking (Higher Accuracy)
Enables a second pass using a Cross-Encoder to re-order the top `k * factor` results.
```powershell
semantica -p . -q "user login logic" --rerank -k 5
```

### Command Line Arguments

| Argument | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| **Path** | `-p`, `--path` | Required | Target directory or file to search. |
| **Query** | `-q`, `--query` | Required | The natural language search query. |
| **Results** | `-k` | `3` | Number of final results to display. |
| **Rerank** | `--rerank` | `False` | Enable the reranking step (slower but more accurate). |
| **Chunk Size** | `-C` | `500` | Max characters per text chunk. |
| **Model** | `-M` | `bge-s` | Embedding model to use (see Supported Models). |
| **Reranker** | `-rM` | `jina-v2` | Reranker model to use. |
| **Factor** | `-f` | `5` | How many candidates to fetch before reranking (`k * f`). |
| **Ignore** | `-i` | `""` | Comma-separated list of extensions to ignore (e.g. `csv,log`). |
| **Purge** | `--purge` | `False` | Clears the embedding cache for the current path. |

---

## 🔍 Example Output

```text
semantica -p D:\projects\ -q "a function that recalculates probabilities taking into account the smoothing parameter" -k 2 -f 20 -i csv --rerank

[INFO] Processing 18 files...
[INFO] Loading embedding model: BAAI/bge-small-en-v1.5 (Threads: 20)...
[INFO] Encoding 532 chunks (Batch: 128, Free RAM: 8.6GB)...
[INFO] Searching...
[INFO] Loading reranker: jinaai/jina-reranker-v2-base-multilingual...

 Query: a function that recalculates probabilities taking into account the smoothing parameter 

1. [-1.3769] forex-prediction-bot\src\model_engine.py : Lines 25-39      
   def counts_to_probabilities(counter, temperature=DEFAULT_TEMPERATURE):
       total = sum(counter.values())
       if total == 0:
           return {k: 0.0 for k in counter}
       probs = {k: v / total for k, v in counter.items()}

2. [-1.6484] forex-prediction-bot\src\model_engine.py : Lines 18-32      
       if not os.path.exists(path_to_model):
           raise FileNotFoundError(f"{path_to_model} does not exist")    
       with open(path_to_model, 'rb') as file:
           model = pickle.load(file)
           return model
```

---

## 🧠 Supported Models

You can switch models using the `-M` (Embedding) and `-rM` (Reranking) flags.

### Embedding Models
| Key | HuggingFace Model | Description |
| :--- | :--- | :--- |
| `bge-s` | `BAAI/bge-small-en-v1.5` | **Default.** Fast and efficient. |
| `bge-m` | `BAAI/bge-base-en-v1.5` | Balanced speed/performance. |
| `minilm`| `sentence-transformers/all-MiniLM-L6-v2` | Very fast, older architecture. |
| `multi` | `intfloat/multilingual-e5-large` | Good for non-English text. |
| `code` | `jinaai/jina-embeddings-v2-base-code` | Optimized specifically for code. |
| `nomic` | `nomic-ai/nomic-embed-text-v1.5` | High performance long-context model. |

### Reranking Models
| Key | HuggingFace Model |
| :--- | :--- |
| `jina-v2` | `jinaai/jina-reranker-v2-base-multilingual` (**Default**) |
| `bge-r-b` | `BAAI/bge-reranker-base` |
| `bge-r-l` | `BAAI/bge-reranker-large` |

---

## 📂 Supported File Types

**Code:**
`.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.cpp`, `.c`, `.h`, `.cs`, `.go`, `.rs`, `.php`, `.rb`, `.sh`, `.sql`, `.html`, `.css`, `.json`, `.yaml`, `.xml`, `.dockerfile`, `.lua`, `.swift`, `.kt`

**Documents:**
`.txt`, `.md`, `.pdf`, `.docx`, `.odt`, `.rtf`, `.log`, `.tex`, `.csv`

_Note: Directories like `.git`, `node_modules`, `venv`, `dist`, etc., are automatically ignored._

---

## ⚙️ Configuration & Caching

*   **Cache Location:** embeddings are stored in `%LOCALAPPDATA%\VectorSearchProject\cache` (Windows) or `~/.cache/VectorSearchProject` (Linux/Mac).
*   **Cache Validity:** Cache is unique to the file path, chunk size, overlap, and model used. If you change arguments, a new index is built. Old cache files are automatically cleaned up after 7 days.

---

## 📜 License

This project is licensed under the MIT License.