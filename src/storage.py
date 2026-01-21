import os
import shutil
import hashlib
import time
import config


def ensure_dirs():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)


def get_models_dir():
    ensure_dirs()
    return config.MODELS_DIR


def get_index_paths(input_path, model_name, chunk_size, overlap):
    ensure_dirs()
    
    abs_path = os.path.abspath(input_path)
    params_str = f"{abs_path}_{model_name}_{chunk_size}_{overlap}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    base_path = os.path.join(config.CACHE_DIR, params_hash)
    return base_path + ".npy", base_path + ".json"


def _get_model_slug(model_name):
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def _find_model_paths(model_name):
    if not os.path.exists(config.MODELS_DIR):
        return []

    slug = _get_model_slug(model_name)
    found_paths = []

    for item in os.listdir(config.MODELS_DIR):
        full_path = os.path.join(config.MODELS_DIR, item)
        
        if not os.path.isdir(full_path):
            continue
            
        if slug in item:
            has_onnx = False
            for root, _, files in os.walk(full_path):
                if any(f.endswith('.onnx') for f in files):
                    has_onnx = True
                    break
            
            if has_onnx:
                found_paths.append(full_path)
                
    return found_paths


def model_exists(model_name):
    paths = _find_model_paths(model_name)
    return len(paths) > 0


def delete_model(model_name):
    paths = _find_model_paths(model_name)
    deleted = False
    
    for path in paths:
        try:
            shutil.rmtree(path)
            deleted = True
        except OSError:
            pass
            
    return deleted


def purge_index_cache():
    if os.path.exists(config.CACHE_DIR):
        try:
            shutil.rmtree(config.CACHE_DIR)
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            return True
        except OSError:
            return False
    return False


def cleanup_old_indexes():
    if not os.path.exists(config.CACHE_DIR):
        return
        
    now = time.time()
    max_age = 86400 * 7 
    
    for filename in os.listdir(config.CACHE_DIR):
        fp = os.path.join(config.CACHE_DIR, filename)
        try:
            if os.path.isfile(fp) and (now - os.path.getmtime(fp) > max_age):
                os.remove(fp)
        except OSError:
            pass