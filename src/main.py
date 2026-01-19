import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import psutil
import warnings
import numpy as np
import faiss
import spacy
from termcolor import cprint, colored

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

try:
    from fastembed import TextEmbedding
    from reader import FileReader
except ImportError:
    print("Error: Missing dependencies (fastembed or reader.py).")
    sys.exit(1)


MODELS = ['small', 'bge-s', 'bge-m', 'bge-l', 'minilm', 'multi', 'code', 'nomic']

IGNORE_DIRS = {
    '.git', '.venv', 'venv', 'env', '.idea', '.vscode', '__pycache__', 
    '.egg-info', 'node_modules', 'dist', 'build', 'target', 'bin', 
    'obj', 'wheels', 'lib', 'scripts', 'include', 'share', 'lib64',
    'migrations', 'tests', 'test', 'data', 'assets', 'tmp', 'temp'
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('-c', '--chunk_size', type=int, default=200)
    parser.add_argument('-o', '--overlap', type=float, default=0.5)
    parser.add_argument('--purge', action='store_true')
    parser.add_argument('-m', '--model', type=str, choices=MODELS, default='small')
    return parser.parse_args()


def log_step(msg, step_type="info"):
    colors = {"info": "cyan", "error": "red", "success": "green", "warn": "yellow"}
    cprint(f"[{step_type.upper()}] {msg}", colors.get(step_type, "white"))


def get_optimal_execution_params():
    cpu_count = os.cpu_count() or 1
    threads = max(1, cpu_count)
    
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
    except:
        available_gb = 8
    
    if available_gb > 16:
        batch_size = 512
    elif available_gb > 8:
        batch_size = 256
    else:
        batch_size = 128
        
    return threads, batch_size


def get_model_name(size):
    models = {
        'bge-s': 'BAAI/bge-small-en-v1.5',
        'bge-m': 'BAAI/bge-base-en-v1.5',
        'bge-l': 'BAAI/bge-large-en-v1.5',
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'multi': 'intfloat/multilingual-e5-large',
        'code': 'jinaai/jina-embeddings-v2-base-code', 
        'nomic': 'nomic-ai/nomic-embed-text-v1.5',
    }
    return models.get(size, 'BAAI/bge-small-en-v1.5')


def get_all_files(path):
    if os.path.isfile(path):
        return [path]
    
    file_list = []
    for root, dirs, filenames in os.walk(path):
        # Фильтрация директорий на лету
        dirs[:] = [
            d for d in dirs 
            if d.lower() not in IGNORE_DIRS and not d.startswith('.')
        ]
        
        for filename in filenames:
            file_list.append(os.path.join(root, filename))
            
    return file_list


def get_cache_paths(input_path, model_name, chunk_size, overlap):
    app_data = os.getenv('LOCALAPPDATA')
    if not app_data:
        app_data = os.path.expanduser('~/.cache')
        
    cache_dir = os.path.join(app_data, 'VectorSearchProject', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    abs_path = os.path.abspath(input_path)
    params_str = f"{abs_path}_{model_name}_{chunk_size}_{overlap}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    base_path = os.path.join(cache_dir, params_hash)
    return base_path + ".npy", base_path + ".json", cache_dir


def cleanup_stale_cache(cache_dir, max_age_seconds=86400):
    if not os.path.exists(cache_dir):
        return
    
    now = time.time()
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        try:
            if os.path.isfile(file_path):
                if now - os.path.getmtime(file_path) > max_age_seconds:
                    os.remove(file_path)
        except OSError:
            pass


def manage_cache(purge_flag, cache_dir):
    if purge_flag and os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            log_step("Cache purged", "success")
            return True
        except Exception as e:
            log_step(f"Purge failed: {e}", "error")
            return False
    elif cache_dir:
        cleanup_stale_cache(cache_dir)
        return False


def chunk_content(content, content_type, args):
    chunks = []
    
    if content_type == 'code':
        lines = content.splitlines()
        total_lines = len(lines)
        current_idx = 0
        
        while current_idx < total_lines:
            buffer_text = []
            char_count = 0
            start_line = current_idx
            temp_idx = current_idx
            
            while temp_idx < total_lines:
                line = lines[temp_idx]
                buffer_text.append(line)
                char_count += len(line)
                temp_idx += 1
                if char_count >= args.chunk_size:
                    break
            
            end_line = temp_idx
            raw_text = "\n".join(buffer_text)
            clean_text = " ".join([l.strip() for l in buffer_text if l.strip()])
            
            if clean_text:
                chunks.append({
                    "text_embed": clean_text,
                    "text_raw": raw_text.rstrip(),
                    "lines": f"{start_line + 1}-{end_line}",
                    "type": "code"
                })
            
            lines_in_chunk = end_line - start_line
            step = max(1, int(lines_in_chunk * (1 - args.overlap)))
            current_idx += step

    else:
        try:
            if not spacy.util.is_package("xx_sent_ud_sm"):
                 spacy.cli.download("xx_sent_ud_sm")
            nlp = spacy.load("xx_sent_ud_sm")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        except:
            return []

        doc = nlp(content)
        sentences = [sent.text for sent in doc.sents]
        
        current_chunk = []
        current_length = 0
        idx = 0
        
        while idx < len(sentences):
            sent = sentences[idx]
            current_chunk.append(sent)
            current_length += len(sent)
            
            if current_length >= args.chunk_size:
                text_segment = " ".join(current_chunk)
                chunks.append({
                    "text_embed": text_segment,
                    "text_raw": text_segment,
                    "lines": "N/A",
                    "type": "text"
                })
                step_back = max(1, int(len(current_chunk) * args.overlap))
                idx = idx - step_back + 1
                current_chunk = []
                current_length = 0
            else:
                idx += 1
                
        if current_chunk:
            text_segment = " ".join(current_chunk)
            chunks.append({
                "text_embed": text_segment,
                "text_raw": text_segment,
                "lines": "N/A",
                "type": "text"
            })
            
    return chunks


def process_files(file_paths, root_path, args):
    reader = FileReader()
    all_chunks = []
    
    for fpath in file_paths:
        try:
            data = reader.read(fpath)
            if not data:
                continue
                
            content = data.get("content", "")
            ctype = data.get("type", "text")
            
            if not content.strip():
                continue
                
            file_chunks = chunk_content(content, ctype, args)
            
            if os.path.isdir(root_path):
                rel_path = os.path.relpath(fpath, root_path)
            else:
                rel_path = os.path.basename(fpath)
                
            for c in file_chunks:
                c['path'] = rel_path
                all_chunks.append(c)
                
        except Exception:
            continue
            
    return all_chunks


def load_index_from_cache(vec_path, meta_path):
    if os.path.exists(vec_path) and os.path.exists(meta_path):
        try:
            embeddings = np.load(vec_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            
            try:
                os.utime(vec_path, None)
                os.utime(meta_path, None)
            except OSError:
                pass
                
            return index, chunks
        except:
            return None, None
    return None, None


def build_and_save_index(chunks, model, vec_path, meta_path):
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    threads, batch_size = get_optimal_execution_params()
    texts = [c["text_embed"] for c in chunks]
    
    log_step(f"Encoding {len(texts)} chunks...", "info")
    
    embeddings_list = []
    generator = model.embed(texts, batch_size=batch_size)
    for vec in generator:
        embeddings_list.append(vec)
        
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    np.save(vec_path, embeddings)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    return index


def search(query, index, model, k, chunks):
    query_gen = model.embed([query])
    query_vec = list(query_gen)[0].astype(np.float32)
    query_vec = query_vec.reshape(1, -1)
    
    D, I = index.search(query_vec, k)
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(chunks):
            results.append((score, chunks[idx]))
    return results


def format_preview(text, is_code=False, max_lines=4, max_width=100):
    if is_code:
        lines = text.split('\n')
        preview_lines = []
        limit = max_lines + 2
        for i in range(min(len(lines), limit)):
            line = lines[i].rstrip()
            if len(line) > max_width:
                line = line[:max_width] + "..."
            if not line: continue
            preview_lines.append(line)
        return "\n   ".join(preview_lines)
    else:
        if len(text) > 300:
            return f"...{text[:300]}..."
        return f"...{text}..."


def main():
    args = get_args()
    
    if not os.path.exists(args.path):
        log_step(f"Path not found: {args.path}", "error")
        return

    model_name = get_model_name(args.model)
    vec_path, meta_path, cache_dir = get_cache_paths(args.path, model_name, args.chunk_size, args.overlap)
    
    if args.purge:
        manage_cache(True, cache_dir)
        return
    
    manage_cache(False, cache_dir)

    threads, _ = get_optimal_execution_params()
    try:
        model = TextEmbedding(model_name=model_name, threads=threads)
    except Exception as e:
        log_step(f"Model init failed: {e}", "error")
        return

    index = None
    chunks = []
    
    if vec_path:
        index, chunks = load_index_from_cache(vec_path, meta_path)
        if index:
            log_step("Loaded from cache", "success")

    if index is None:
        target_files = get_all_files(args.path)
        if not target_files:
            log_step("No valid files found to scan", "error")
            return

        log_step(f"Processing {len(target_files)} files...", "info")
        chunks = process_files(target_files, args.path, args)
        
        if not chunks:
            log_step("No valid text/code chunks found", "error")
            return
            
        index = build_and_save_index(chunks, model, vec_path, meta_path)

    log_step("Searching...", "info")
    results = search(args.query, index, model, args.k, chunks)
    
    print()
    cprint(f" Query: {args.query} ", "white", "on_blue", attrs=['bold'])
    print()

    if not results:
        cprint("No results.", "yellow")

    for i, (score, res) in enumerate(results, 1):
        score_str = f"[{score:.4f}]"
        path_str = colored(res.get('path', 'unknown'), "cyan", attrs=['underline'])
        
        if res.get('type') == 'code':
            meta = colored(f"Lines {res['lines']}", "yellow")
            header = f"{score_str} {path_str} : {meta}"
        else:
            header = f"{score_str} {path_str}"
            
        print(f"{i}. {header}")
        preview = format_preview(res['text_raw'], is_code=(res.get('type') == 'code'))
        print(f"   {colored(preview, 'grey')}")
        print()


if __name__ == "__main__":
    main()