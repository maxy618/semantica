import os
import shutil
import psutil
import warnings
import logging
import tempfile
import numpy as np
from termcolor import cprint, colored


def setup_system():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LOGURU_LEVEL"] = "CRITICAL"
    logging.getLogger("fastembed").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def get_hardware_info():
    cpu_count = os.cpu_count() or 1
    threads = max(1, cpu_count)
    
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
    except:
        available_gb = 8
        
    return threads, available_gb


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    if score > 0.75:
        bg_color = "on_green"
    elif score > 0.5:
        bg_color = "on_yellow"
    else:
        bg_color = "on_red"

    score_str = f" {score:.2%} "
    score_pill = colored(score_str, "white", bg_color, attrs=['bold'])
    
    path_str = colored(res.get('path', 'unknown'), "cyan", attrs=['underline'])
    is_code = (res.get('type') == 'code')
    
    if is_code:
        meta = colored(f"Lines {res['lines']}", "yellow")
        header = f"{score_pill} {path_str} : {meta}"
    else:
        header = f"{score_pill} {path_str}"
        
    print(f"{rank}. {header}")
    preview = format_preview(res['text_raw'], is_code=is_code)
    print(f"   {colored(preview, 'grey')}")
    print()


def get_model_cache_roots():
    roots = [os.path.join(os.path.expanduser("~"), ".cache", "fastembed")]
    roots.append(os.path.join(tempfile.gettempdir(), "fastembed_cache"))
    return roots


def check_model_cache(model_name):
    slug = model_name.split("/")[-1]
    
    for root in get_model_cache_roots():
        if os.path.exists(root):
            for d in os.listdir(root):
                if slug in d:
                    full_path = os.path.join(root, d)
                    for _, _, files in os.walk(full_path):
                        if "model.onnx" in files:
                            return True
    return False


def delete_model_cache(model_name):
    slug = model_name.split("/")[-1]
    deleted = False
    
    for root in get_model_cache_roots():
        if os.path.exists(root):
            for d in os.listdir(root):
                if slug in d:
                    try:
                        shutil.rmtree(os.path.join(root, d))
                        deleted = True
                    except Exception:
                        pass
    return deleted