import os
import psutil
import warnings
from termcolor import cprint, colored


def setup_system():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


def get_hardware_info():
    cpu_count = os.cpu_count() or 1
    threads = max(1, cpu_count)
    
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
    except:
        available_gb = 8
        
    return threads, available_gb


def log(msg, level="info"):
    colors = {
        "info": "cyan", 
        "error": "red", 
        "success": "green", 
        "warn": "yellow",
        "system": "magenta"
    }
    color = colors.get(level, "white")
    prefix = level.upper()
    cprint(f"[{prefix}] {msg}", color)


def log_file_error(filepath, error):
    cprint(f"[WARN] Skipped {filepath}: {error}", "yellow")


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


def print_result(rank, score, res):
    score_str = f"[{score:.4f}]"
    path_str = colored(res.get('path', 'unknown'), "cyan", attrs=['underline'])
    is_code = (res.get('type') == 'code')
    
    if is_code:
        meta = colored(f"Lines {res['lines']}", "yellow")
        header = f"{score_str} {path_str} : {meta}"
    else:
        header = f"{score_str} {path_str}"
        
    print(f"{rank}. {header}")
    preview = format_preview(res['text_raw'], is_code=is_code)
    print(f"   {colored(preview, 'grey')}")
    print()