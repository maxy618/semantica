from utils import setup_system, log, print_result
setup_system()
import os
import argparse
import sys
from termcolor import cprint
import config
from scanner import get_all_files, process_files
from engine import SearchEngine


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-k', '--k', type=int, default=config.DEFAULT_ARGS['k'])
    parser.add_argument('-c', '--chunk_size', type=int, default=config.DEFAULT_ARGS['chunk_size'])
    parser.add_argument('-o', '--overlap', type=float, default=config.DEFAULT_ARGS['overlap'])
    parser.add_argument('--purge', action='store_true')
    
    parser.add_argument('-m', '--model', type=str,
                       default=config.DEFAULT_ARGS['model'])
    
    parser.add_argument('-rm', '--rerank_model', type=str,
                       default=config.DEFAULT_ARGS['rerank_model'])
                       
    parser.add_argument('-sm', '--spacy_model', type=str, 
                       default=config.DEFAULT_ARGS['spacy_model'])

    parser.add_argument('--rerank', action='store_true', help="Enable reranking step (disabled by default)")
    parser.add_argument('-f', '--factor', type=int, default=config.DEFAULT_ARGS['rerank_factor'])
    
    return parser.parse_args()


def main():
    args = get_args()
    
    if not os.path.exists(args.path):
        log(f"Path not found: {args.path}", "error")
        return

    try:
        main_model = config.MODELS_MAPPING.get(args.model, args.model)
        rerank_model = config.RERANK_MODELS_MAPPING.get(args.rerank_model, args.rerank_model)

        engine = SearchEngine(main_model)
        vec_path, meta_path, cache_dir = engine.get_cache_paths(args.path, args.chunk_size, args.overlap)
        
        if args.purge:
            engine.manage_cache(True, cache_dir)
            return
        
        engine.manage_cache(False, cache_dir)
        
        if not engine.load_index(vec_path, meta_path):
            files = get_all_files(args.path)
            if not files:
                log("No files found", "error")
                return

            log(f"Processing {len(files)} files...", "info")
            chunks = process_files(files, args.path, args)
            
            if not chunks:
                log("No content extracted", "error")
                return
                
            engine.build_index(chunks, vec_path, meta_path)
        else:
            log("Index loaded from cache", "success")

        log("Searching...", "info")
        
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

    except KeyboardInterrupt:
        print()
        log("Aborted", "warn")
        sys.exit(0)
    except Exception as e:
        log(f"Critical error: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()