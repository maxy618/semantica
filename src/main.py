import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from termcolor import cprint, colored


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('-c', '--chunk_size', type=int, default=200)
    parser.add_argument('-o', '--overlap', type=float, default=0.5)
    parser.add_argument('--code', action='store_true')
    parser.add_argument('-s', '--model_size', type=str, choices=['small', 'medium', 'large'], default='small')
    return parser.parse_args()


def log_step(msg, step_type="info"):
    if step_type == "info":
        cprint(f"[INFO] {msg}", "cyan")
    elif step_type == "error":
        cprint(f"[ERROR] {msg}", "red")
    elif step_type == "success":
        cprint(f"[OK] {msg}", "green")


def get_model_name(size):
    models = {
        'small': 'all-MiniLM-L6-v2',
        'medium': 'all-MiniLM-L12-v2',
        'large': 'all-mpnet-base-v2'
    }
    return models.get(size, 'all-MiniLM-L6-v2')


def read_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()


def create_chunks(lines, args):
    chunks = []
    total_lines = len(lines)
    
    if args.code:
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
            
            raw_text = "".join(buffer_text)
            clean_text_for_embed = " ".join([l.strip() for l in buffer_text if l.strip()])
            
            chunks.append({
                "text_embed": clean_text_for_embed,
                "text_raw": raw_text.rstrip(), 
                "lines": f"{start_line + 1}-{end_line}"
            })
            
            lines_in_chunk = end_line - start_line
            step = max(1, int(lines_in_chunk * (1 - args.overlap)))
            current_idx += step
    else:
        full_text = "".join(lines)
        clean_full = " ".join(full_text.split())
        
        step = int(args.chunk_size * (1 - args.overlap))
        if step < 1: step = 1
        
        for i in range(0, len(clean_full), step):
            segment = clean_full[i : i + args.chunk_size]
            if not segment: continue
            chunks.append({
                "text_embed": segment,
                "text_raw": segment,
                "lines": "N/A"
            })
            if len(segment) < args.chunk_size:
                break
                
    return chunks


def build_index(chunks, model_name):
    log_step(f"Loading model '{model_name}'...", "info")
    model = SentenceTransformer(model_name)
    
    texts = [c["text_embed"] for c in chunks]
    log_step(f"Encoding {len(texts)} chunks...", "info")
    
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    return index, model


def search(query, index, model, k, chunks):
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_vec = query_vec.astype(np.float32)
    
    D, I = index.search(query_vec, k)
    found_indices = I[0]
    
    results = []
    for idx in found_indices:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
            
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
            
            if not line:
                continue
                
            preview_lines.append(line)
        
        return "\n   ".join(preview_lines)
    else:
        return f"...{text}..."


def main():
    args = get_args()

    try:
        lines = read_lines(args.file)
    except FileNotFoundError:
        log_step(f"File not found: {args.file}", "error")
        return
    except Exception as e:
        log_step(f"Read error: {e}", "error")
        return

    chunks = create_chunks(lines, args)
    if not chunks:
        log_step("No text chunks created", "error")
        return

    model_name = get_model_name(args.model_size)
    index, model = build_index(chunks, model_name)
    
    log_step("Searching...", "info")
    results = search(args.query, index, model, args.k, chunks)
    
    print()
    cprint(f" Query: {args.query} ", "white", "on_blue", attrs=['bold'])
    print()

    for i, res in enumerate(results, 1):
        if args.code:
            header = colored(f"Lines {res['lines']}", "yellow", attrs=['bold'])
        else:
            header = colored("Match found", "yellow")
            
        print(f"{i}. {header}")
        
        preview = format_preview(res['text_raw'], is_code=args.code)
        print(f"   {colored(preview, 'grey')}")
        print()


if __name__ == "__main__":
    main()