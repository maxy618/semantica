import os
import psutil
from config import IGNORE_DIRS
from reader import FileReader
from utils import log, log_file_error
import xx_sent_ud_sm 


def get_all_files(path, ignore_exts=None, depth=None):
    if ignore_exts is None:
        ignore_exts = set()
        
    reader = FileReader()
    
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ignore_exts:
            return []
        return [path] if reader.is_supported(path) else []
    
    file_list = []
    base_depth = path.rstrip(os.sep).count(os.sep)

    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
        
        if depth is not None:
            current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
            if current_depth + 1 >= depth:
                dirs[:] = []
        
        for filename in filenames:
            if filename.startswith('.'):
                continue
                
            ext = os.path.splitext(filename)[1].lower()
            if ext in ignore_exts:
                continue
                
            full_path = os.path.join(root, filename)
            if reader.is_supported(full_path):
                file_list.append(full_path)
            
    return file_list


def check_ram_safety():
    mem = psutil.virtual_memory()
    if mem.available < 500 * 1024 * 1024 or mem.percent > 95:
        return False
    return True


def chunk_code(content, args):
    chunks = []
    lines = content.splitlines()
    total_lines = len(lines)
    HARD_LIMIT = args.chunk_size * 2
    
    i = 0
    while i < total_lines:
        chunk_lines = []
        current_size = 0
        start_line = i
        
        while i < total_lines:
            line = lines[i]
            if len(line) > HARD_LIMIT:
                line = line[:HARD_LIMIT] + "..."
            
            line_len = len(line) + 1 
            
            if current_size + line_len > args.chunk_size and chunk_lines:
                break
            chunk_lines.append(line)
            current_size += line_len
            i += 1
            
            if current_size >= args.chunk_size:
                break
            
        text = "\n".join(chunk_lines)
        if text.strip():
            chunks.append({
                "text_embed": text,
                "text_raw": text,
                "lines": f"{start_line + 1}-{i}",
                "type": "code"
            })
    
        step = max(1, int(len(chunk_lines) * (1 - args.overlap)))
        if step < len(chunk_lines):
            i = start_line + step
        else:
            pass
            
    return chunks


def chunk_text(content, args, nlp):
    try:
        doc = nlp(content)
        sentences = [s.text for s in doc.sents]
    except Exception:
        return chunk_code(content, args)
    
    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = []
        current_len = 0
        
        j = i
        while j < len(sentences):
            sent = sentences[j]
            if len(sent) > args.chunk_size * 2:
                sent = sent[:args.chunk_size * 2]

            if current_len + len(sent) > args.chunk_size and current_chunk:
                break
            
            current_chunk.append(sent)
            current_len += len(sent)
            j += 1
            
        text = " ".join(current_chunk)
        if text.strip():
            chunks.append({
                "text_embed": text,
                "text_raw": text,
                "lines": "N/A",
                "type": "text"
            })
            
        step = max(1, int((j - i) * (1 - args.overlap)))
        i += step
        
    return chunks


def process_files(file_paths, root_path, args):
    reader = FileReader()
    all_chunks = []
    nlp = None
    
    has_text = any(reader.get_file_type(fp) == 'text' for fp in file_paths)
    if has_text:
        try:
            nlp = xx_sent_ud_sm.load()
            nlp.add_pipe("sentencizer")
            nlp.max_length = 1_500_000
        except Exception as e:
            log(f"Spacy load failed: {e}", "error")

    for fpath in file_paths:
        data = reader.read(fpath)
        if not data: 
            continue
            
        if "error" in data:
            log_file_error(fpath, data['error'])
            continue
            
        content = data["content"]
        ctype = data["type"]
        if len(content) > 1_000_000:
            ctype = 'code'

        if ctype == 'text' and nlp:
            if not check_ram_safety():
                file_chunks = chunk_code(content, args)
            else:
                file_chunks = chunk_text(content, args, nlp)
        else:
            file_chunks = chunk_code(content, args)
            
        rel_path = os.path.relpath(fpath, root_path) if os.path.isdir(root_path) else os.path.basename(fpath)
            
        for c in file_chunks:
            c['path'] = rel_path
            all_chunks.append(c)
            
    return all_chunks