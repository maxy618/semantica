# 🧠 Semantica

**Semantica** is a high-performance, local command-line semantic search engine designed specifically for developers. Unlike `grep` or standard IDE search tools that rely on exact keyword matching, Semantica understands the **meaning** behind your query.

It allows you to search through complex codebases, documentation, and notes using natural language, finding relevant code snippets even if the variable names or comments don't match your keywords exactly.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-purple?style=flat-square)
![Build](https://img.shields.io/badge/Build-Nuitka-yellow?style=flat-square)

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
    - [Basic Search](#basic-search)
    - [Search with Reranking](#search-with-reranking-high-precision)
    - [Managing Models](#managing-models)
- [Example Output](#-example-output)
- [Configuration & Arguments](#-configuration--arguments)
- [Supported Models](#-supported-models)
- [Supported File Types](#-supported-file-types)
- [Building the Executable](#-building-the-executable)
- [License](#-license)

---

## ✨ Features

*   **Context-Aware Search:** Finds code that *does* what you ask, not just code that *contains* the words you typed.
*   **100% Local & Private:** No API keys required. No code leaves your machine. All embeddings and inference happen on your CPU.
*   **Language Agnostic:** Works effectively on Python, Rust, Go, JavaScript, C++, SQL, and more.
*   **Document Intelligence:** seamlessly parses PDFs, Word docs, and Markdown alongside your code.
*   **High Performance Stack:**
    *   **FastEmbed:** Quantized ONNX runtime for blazing fast inference.
    *   **FAISS:** Facebook AI Similarity Search for sub-millisecond vector lookups.
    *   **Multiprocessing:** Parallel file encoding.
*   **Two-Stage Pipeline:** Uses lightweight embeddings for retrieval and heavy Cross-Encoders for high-precision reranking (optional).
*   **Smart Caching:** Hashes content and model parameters to prevent re-indexing unchanged files.

---

## 🔧 How It Works

1.  **Scan:** Semantica recursively scans your target directory, automatically ignoring noise (`.git`, `node_modules`, `__pycache__`, etc.).
2.  **Chunk:** Files are split into meaningful segments. Code is chunked by lines with overlap to preserve context; text is chunked by sentences.
3.  **Embed:** A Transformer model converts these chunks into dense vector representations.
4.  **Index:** Vectors are stored in a local FAISS index for rapid retrieval.
5.  **Search:** Your natural language query is converted to a vector and compared against the index using Cosine Similarity.
6.  **Rerank (Optional):** The top results are passed to a Cross-Encoder model to finely grade their relevance to the query.

---

## 🚀 Installation

### Option 1: Download Release (Recommended)
No Python required. Download the standalone executable `semantica.exe` from the **[Releases](https://github.com/maxy618/semantica/releases)** page.

Add it to your system PATH or run it directly:
```powershell
.\semantica.exe -p "C:\MyProject" -q "how is database connection handled?"
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
    ```bash
    pip install fastembed faiss-cpu numpy pypdf python-docx termcolor psutil nuitka spacy tqdm
    
    # Install the multi-language tokenizer for smart text splitting
    pip install https://github.com/explosion/spacy-models/releases/download/xx_sent_ud_sm-3.7.0/xx_sent_ud_sm-3.7.0-py3-none-any.whl
    ```

---

## 📖 Usage Guide

Run Semantica via the command line interface (CLI).

### Basic Search
The default mode uses the `bge-small` model. It's fast and effective for most use cases.
```powershell
semantica -p ./my-repo -q "authentication middleware logic"
```

### Search with Reranking (High Precision)
For complex queries, enable the **Reranker**. It retrieves more candidates (defined by `-f`) and sorts them using a more powerful model.
```powershell
semantica -p . -q "user login logic" --rerank -k 5
```

### Managing Models
You can purge downloaded model files to free up disk space.
```powershell
# Remove a specific model
semantica --purge-model minilm

# Remove the currently selected model
semantica -M bge-s --purge-model
```

---

## 🔍 Example Output

Semantica provides clean, readable output with syntax highlighting and relevance scores.

```text
> semantica -p D:\projects\ -q "a function creates an animation from images" -k 2 -i csv,md --rerank -C 1000
 Semantica CLI 
[SYSTEM] Parameters: 20 threads, 9.1GB RAM available
[SYSTEM] Target: D:\projects\
[SYSTEM] Model: BAAI/bge-small-en-v1.5
[SYSTEM] Reranker: jinaai/jina-reranker-v2-base-multilingual

[WARN] Cache miss. Scanning files...
[INFO] Processing 18 files...
[INFO] Loading embedding model: BAAI/bge-small-en-v1.5...
[INFO] Encoding 309 chunks (Batch: 128)...
[SUCCESS] Index built and saved to cache.

[INFO] Starting search sequence...
[INFO] Vectorizing query...
[INFO] Retrieved 10 candidates. Initializing reranker...
[INFO] Loading reranker: jinaai/jina-reranker-v2-base-multilingual...
[SUCCESS] Reranking complete.

 Query: a function creates an animation from images 

1.  27.31%  forex-prediction-bot\src\plotter.py : Lines 77-104
   def add_motion_trail(base_img, forward_img, progress):
       w, h = base_img.size
       trail = Image.new("RGBA", (w, h), (0,0,0,0))
       for i in range(TRAIL_SAMPLES):

2.  27.05%  forex-prediction-bot\src\plotter.py : Lines 91-117
           trail.paste(tmp, (dx,0), tmp)
       return Image.alpha_composite(base_img, trail)
   def build_transition_frames(img_a, img_b):
       frames = []


 Finished in 62.441s 

>
```

**Note:** The percentage score represents the semantic similarity.
*   **Green Pill:** High confidence (>75%)
*   **Yellow Pill:** Medium confidence (>50%)
*   **Red Pill:** Low confidence (but still the best found)

---

## ⚙️ Configuration & Arguments

| Argument | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| **Path** | `-p`, `--path` | **Required** | The directory or file to search. |
| **Query** | `-q`, `--query` | **Required** | Your natural language search phrase. |
| **Results** | `-k` | `3` | Number of final results to display. |
| **Model** | `-M` | `bge-s` | Embedding model to use (see table below). |
| **Rerank** | `--rerank` | `False` | Enable the secondary reranking pass. |
| **Reranker** | `-rM` | `jina-v2` | The model used for reranking. |
| **Factor** | `-f` | `5` | Multiplier for reranking (fetches `k * f` candidates). |
| **Chunk Size** | `-C` | `500` | Max characters per chunk. Smaller = more precise context. |
| **Overlap** | `-o` | `0.5` | Percentage (0.0-1.0) of overlap between chunks. |
| **Ignore** | `-i` | `""` | Extensions to skip (e.g. `csv,log,txt`). |
| **Depth** | `-d` | `None` | Recursion depth (1 = current dir only). |
| **Purge Cache**| `--purge` | `False` | Force rebuild of the index for the current path. |
| **Purge Model**| `-pm` | `None` | Delete model files from disk. |

---

## 🧠 Supported Models

Semantica supports various models optimized for different hardware and use cases. Use the **Key** in the `-M` or `-rM` argument.

### Embedding Models (First Pass)

| Key | HuggingFace Model | Size | Speed | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`bge-s`** | `BAAI/bge-small-en-v1.5` | ~130MB | ⭐⭐⭐⭐⭐ | **Default.** Excellent balance of speed and accuracy. |
| **`bge-m`** | `BAAI/bge-base-en-v1.5` | ~400MB | ⭐⭐⭐ | Better understanding of complex sentences. |
| **`minilm`**| `all-MiniLM-L6-v2` | ~80MB | ⭐⭐⭐⭐⭐ | Extremely fast, legacy standard. Good for English. |
| **`multi`** | `multilingual-e5-large` | ~2GB | ⭐ | Best for non-English languages (Ru, Cn, De, etc). |
| **`code`** | `jina-embeddings-v2-code` | ~500MB | ⭐⭐⭐ | Specifically trained on API docs and source code. |
| **`nomic`** | `nomic-embed-text-v1.5` | ~500MB | ⭐⭐ | Supports very long context windows. |

### Reranking Models (Second Pass)

| Key | HuggingFace Model | Description |
| :--- | :--- | :--- |
| **`jina-v2`** | `jina-reranker-v2-base-multilingual` | **Default.** State-of-the-art accuracy. |
| `bge-r-b` | `BAAI/bge-reranker-base` | Fast reranker by BAAI. |
| `bge-r-l` | `BAAI/bge-reranker-large` | Very heavy, maximum precision. |

---

## 📂 Supported File Types

Semantica automatically detects and processes these formats.

*   **Code:** `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.cpp`, `.c`, `.h`, `.cs`, `.go`, `.rs`, `.php`, `.rb`, `.sh`, `.sql`, `.html`, `.css`, `.json`, `.yaml`, `.xml`, `.dockerfile`, `.lua`, `.swift`, `.kt`
*   **Documents:** `.txt`, `.md`, `.pdf`, `.docx`, `.odt`, `.rtf`, `.log`, `.tex`, `.csv`

*System directories (`.git`, `venv`, `node_modules`, `dist`, etc.) are excluded by default.*

---

## 🛠 Building the Executable

To generate a standalone `.exe` that includes all dependencies (including machine learning models and tokenizers), use the following **Nuitka** command.

**Note:** This command ensures that `fastembed`, `faiss`, and `spacy` models are correctly bundled.

Windows

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

Linux

```bash
python3 -m nuitka \
  --standalone \
  --onefile \
  --include-package=src \
  --include-package=fastembed \
  --include-package=faiss \
  --include-package=pypdf \
  --include-package=docx \
  --include-package=xx_sent_ud_sm \
  --include-package-data=xx_sent_ud_sm \
  --spacy-language-model=xx_sent_ud_sm \
  --include-package=tokenizers \
  --include-package=onnxruntime \
  --nofollow-import-to=matplotlib \
  --nofollow-import-to=pandas \
  --nofollow-import-to=scipy \
  --nofollow-import-to=IPython \
  --nofollow-import-to=pytest \
  --nofollow-import-to=tkinter \
  --output-filename=semantica \
  src/main.py
```

---

## 📜 License

This project is licensed under the MIT License. You are free to use, modify, and distribute it.