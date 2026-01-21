from utils import setup_system, log, print_result, get_hardware_info
setup_system()
import multiprocessing
import argparse
import sys
import time
from termcolor import cprint
import config
import storage
from scanner import get_all_files, process_files
from engine import SearchEngine


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-q', '--query', type=str)
    parser.add_argument('-k', '--k', type=int, default=config.DEFAULT_ARGS['k'])
    parser.add_argument('-C', '--chunk_size', type=int, default=config.DEFAULT_ARGS['chunk_size'])
    parser.add_argument('-o', '--overlap', type=float, default=config.DEFAULT_ARGS['overlap'])
    parser.add_argument('-i', '--ignore', type=str, default="", help="Comma separated extensions to ignore (e.g. csv,txt)")
    parser.add_argument('-d', '--depth', type=int, default=None, help="Recursion depth (1 = current dir only)")
    parser.add_argument('--purge', action='store_true', help="Purge the vector index cache")
    parser.add_argument('-pm', '--purge-model', nargs='?', const='CURRENT', help="Delete model files. Can accept model name (e.g. -pm minilm)")
    parser.add_argument('-M', '--model', type=str, default=config.DEFAULT_ARGS['model'])
    parser.add_argument('-rM', '--rerank_model', type=str,default=config.DEFAULT_ARGS['rerank_model'])
    parser.add_argument('--rerank', action='store_true', help="Enable reranking step (disabled by default)")
    parser.add_argument('-f', '--factor', type=int, default=config.DEFAULT_ARGS['rerank_factor'])
    
    return parser.parse_args()


def resolve_model_name(raw_name, map_dict):
    if raw_name in map_dict:
        return map_dict[raw_name]
    return raw_name


def main():
    start_time = time.perf_counter()
    args = get_args()

    if args.purge_model:
        target_raw = args.model if args.purge_model == 'CURRENT' else args.purge_model
        
        target_model = resolve_model_name(target_raw, config.MODELS_MAPPING)
        if target_model == target_raw:
             target_model = resolve_model_name(target_raw, config.RERANK_MODELS_MAPPING)
        
        if storage.delete_model(target_model):
            log(f"Model files for '{target_model}' purged.", "success")
        else:
            log(f"No cache found for '{target_model}' to purge.", "warn")
        return

    if args.purge:
        if storage.purge_index_cache():
            log("Vector index cache purged.", "success")
        else:
            log("Cache was empty or locked.", "warn")
        return

    if not args.path:
        log("Argument -p/--path is required", "error")
        return
        
    if not args.query:
        log("Argument -q/--query is required for search", "error")
        return

    main_model = resolve_model_name(args.model, config.MODELS_MAPPING)
    rerank_model = resolve_model_name(args.rerank_model, config.RERANK_MODELS_MAPPING)
    
    threads, available_gb = get_hardware_info()
    
    cprint(f" {config.APP_NAME} CLI ", "white", "on_magenta", attrs=['bold'])
    log(f"Parameters: {threads} threads, {available_gb:.1f}GB RAM available", "system")
    log(f"Target: {args.path}", "system")
    log(f"Model: {main_model}", "system")
    if args.rerank:
        log(f"Reranker: {rerank_model}", "system")
    print()

    ignore_exts = set()
    if args.ignore:
        for ext in args.ignore.split(','):
            ext = ext.strip().lower()
            if ext:
                if not ext.startswith('.'):
                    ext = '.' + ext
                ignore_exts.add(ext)

    storage.cleanup_old_indexes()
    models_dir = storage.get_models_dir()
    
    engine = SearchEngine(main_model, threads, available_gb, models_dir)
    vec_path, meta_path = storage.get_index_paths(args.path, main_model, args.chunk_size, args.overlap)
    
    if not engine.load_index(vec_path, meta_path):
        log("Cache miss. Scanning files...", "warn")
        files = get_all_files(args.path, ignore_exts=ignore_exts, depth=args.depth)
        
        if not files:
            log("No files found (check path or ignored extensions)", "error")
            return

        log(f"Processing {len(files)} files...", "info")
        chunks = process_files(files, args.path, args)
        
        if not chunks:
            log("No content extracted", "error")
            return
            
        engine.build_index(chunks, vec_path, meta_path)

    print()
    log("Starting search sequence...", "info")
    
    results = engine.search(
        args.query, 
        args.k, 
        use_rerank=args.rerank, 
        factor=args.factor,
        rerank_model_name=rerank_model
    )
    
    print()
    cprint(f" Query: {args.query} ", "white", "on_blue", attrs=['bold'])
    print()

    if not results:
        cprint("No results found.", "yellow")

    for i, (score, res) in enumerate(results, 1):
        print_result(i, score, res)

    elapsed = time.perf_counter() - start_time
    print()
    cprint(f" Finished in {elapsed:.3f}s ", "black", "on_white", attrs=['bold'])
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print()
        log("Aborted", "warn")
        sys.exit(0)
    except Exception as e:
        log(f"Critical error: {e}", "error")
        sys.exit(1)