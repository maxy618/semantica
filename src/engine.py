import os
import json
import time
import shutil
import hashlib
import numpy as np
import faiss
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from utils import log, get_hardware_info


class SearchEngine:
    def __init__(self, model_name):
        self.model_name = model_name
        self.threads, self.available_gb = get_hardware_info()
        self.model = None
        self.ranker = None
        self.ranker_name = None
        self.index = None
        self.chunks = []


    def load_model(self):
        if not self.model:
            try:
                log(f"Loading embedding model: {self.model_name} (Threads: {self.threads})...", "info")
                self.model = TextEmbedding(model_name=self.model_name, threads=self.threads)
            except Exception as e:
                raise RuntimeError(f"Embedding model init failed: {e}")


    def load_ranker(self, ranker_model_name):
        if self.ranker and self.ranker_name == ranker_model_name:
            return

        try:
            log(f"Loading reranker: {ranker_model_name}...", "info")
            self.ranker = TextCrossEncoder(model_name=ranker_model_name, threads=self.threads)
            self.ranker_name = ranker_model_name
        except Exception as e:
            log(f"Reranker init failed: {e}", "warn")
            self.ranker = None


    def get_safe_batch_size(self):
        """works on my machine"""
        available_gb = self.available_gb
        
        is_heavy = 'nomic' in self.model_name or 'large' in self.model_name
        
        if available_gb > 16:
            batch_size = 128 if is_heavy else 256
        elif available_gb > 8:
            batch_size = 64 if is_heavy else 128
        elif available_gb > 4:
            batch_size = 32 if is_heavy else 64
        else:
            batch_size = 16
            
        return batch_size, available_gb


    def get_cache_paths(self, input_path, chunk_size, overlap):
        app_data = os.getenv('LOCALAPPDATA')
        if not app_data:
            app_data = os.path.expanduser('~/.cache')
            
        cache_dir = os.path.join(app_data, 'VectorSearchProject', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        abs_path = os.path.abspath(input_path)
        params_str = f"{abs_path}_{self.model_name}_{chunk_size}_{overlap}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        base_path = os.path.join(cache_dir, params_hash)
        return base_path + ".npy", base_path + ".json", cache_dir


    def manage_cache(self, purge_flag, cache_dir):
        if purge_flag:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    log("Cache purged", "success")
                    return True
                except Exception as e:
                    log(f"Purge failed: {e}", "error")
            return False
            
        if cache_dir and os.path.exists(cache_dir):
            now = time.time()
            max_age = 86400 * 7 
            for filename in os.listdir(cache_dir):
                fp = os.path.join(cache_dir, filename)
                try:
                    if os.path.isfile(fp) and (now - os.path.getmtime(fp) > max_age):
                        os.remove(fp)
                except OSError:
                    pass
        return False


    def load_index(self, vec_path, meta_path):
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            try:
                embeddings = np.load(vec_path)
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                
                d = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(embeddings)
                
                os.utime(vec_path, None)
                os.utime(meta_path, None)
                return True
            except:
                return False
        return False


    def build_index(self, chunks, vec_path, meta_path):
        self.load_model()
        self.chunks = chunks
        
        batch_size, free_mem = self.get_safe_batch_size()
        log(f"Encoding {len(chunks)} chunks (Batch: {batch_size}, Free RAM: {free_mem:.1f}GB)...", "info")
        
        texts = [c["text_embed"] for c in chunks]

        embeddings_list = []

        try:
            for vec in self.model.embed(texts, batch_size=batch_size):
                embeddings_list.append(vec)
        except Exception as e:
            log(f"Encoding process crashed: {e}", "error")

            log("Retrying with batch_size=1...", "warn")
            embeddings_list = []
            try:
                for _, vec in enumerate(self.model.embed(texts, batch_size=1)):
                    embeddings_list.append(vec)
            except Exception as e2:
                 log(f"Critical failure on chunk. Try cleaning your data or checking for minified files.", "error")
                 return

        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        np.save(vec_path, embeddings)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)
        
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)


    def search(self, query, k, use_rerank=True, factor=5, rerank_model_name=None):
        self.load_model()
        if not self.index:
            return []

        initial_k = k * factor if use_rerank else k
        
        query_vec = list(self.model.embed([query]))[0].astype(np.float32)
        query_vec = query_vec.reshape(1, -1)
        
        D, I = self.index.search(query_vec, initial_k)
        
        candidates = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.chunks):
                candidates.append((score, self.chunks[idx]))

        if not candidates or not use_rerank:
            return candidates[:k]

        self.load_ranker(rerank_model_name)
        
        if self.ranker:
            try:
                cand_texts = [c[1]['text_raw'] for c in candidates]
                scores = list(self.ranker.rerank(query, cand_texts))
                
                reranked = []
                for idx, score in enumerate(scores):
                    reranked.append((score, candidates[idx][1]))
                
                reranked.sort(key=lambda x: x[0], reverse=True)
                return reranked[:k]
            except Exception as e:
                log(f"Reranking warn: {e}", "warn")
                
        return candidates[:k]