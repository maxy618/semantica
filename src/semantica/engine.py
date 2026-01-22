import os
import sys
import json
import numpy as np
import faiss
from tqdm import tqdm
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from .utils import log, sigmoid
from . import storage


class SearchEngine:
    def __init__(self, model_name, threads, available_gb, cache_dir_models, chunk_size=500):
        self.model_name = model_name
        self.threads = threads
        self.available_gb = available_gb
        self.cache_dir_models = cache_dir_models
        self.chunk_size = chunk_size
        self.model = None
        self.ranker = None
        self.ranker_name = None
        self.index = None
        self.chunks = []


    def load_model(self):
        if self.model:
            return

        try:
            if storage.model_exists(self.model_name):
                log(f"Loading embedding model: {self.model_name}...", "info")
            else:
                log(f"Model {self.model_name} not found locally. Downloading...", "warn")

            self.model = TextEmbedding(
                model_name=self.model_name, 
                threads=self.threads,
                cache_dir=self.cache_dir_models
            )
        except Exception as e:
            err_str = str(e)
            if "ONNXRuntimeError" in err_str or "NO_SUCHFILE" in err_str:
                log(f"Model corruption detected: {e}", "error")
                log("Purging corrupt model and retrying download...", "warn")
                
                storage.delete_model(self.model_name)
                
                try:
                    self.model = TextEmbedding(
                        model_name=self.model_name, 
                        threads=self.threads,
                        cache_dir=self.cache_dir_models
                    )
                    log("Model recovered successfully.", "success")
                except Exception as e2:
                    raise RuntimeError(f"Recovery failed: {e2}")
            else:
                raise RuntimeError(f"Embedding model init failed: {e}")


    def load_ranker(self, ranker_model_name):
        if self.ranker and self.ranker_name == ranker_model_name:
            return

        try:
            if storage.model_exists(ranker_model_name):
                log(f"Loading reranker: {ranker_model_name}...", "info")
            else:
                log(f"Reranker {ranker_model_name} not found locally. Downloading...", "warn")

            self.ranker = TextCrossEncoder(
                model_name=ranker_model_name, 
                threads=self.threads,
                cache_dir=self.cache_dir_models
            )
            self.ranker_name = ranker_model_name
        except Exception as e:
            err_str = str(e)
            if "ONNXRuntimeError" in err_str or "NO_SUCHFILE" in err_str:
                log(f"Reranker corruption detected: {e}", "error")
                log("Purging corrupt reranker and retrying download...", "warn")
                
                storage.delete_model(ranker_model_name)
                
                try:
                    self.ranker = TextCrossEncoder(
                        model_name=ranker_model_name, 
                        threads=self.threads,
                        cache_dir=self.cache_dir_models
                    )
                    self.ranker_name = ranker_model_name
                    log("Reranker recovered successfully.", "success")
                except Exception as e2:
                    log(f"Reranker recovery failed: {e2}", "warn")
                    self.ranker = None
            else:
                log(f"Reranker init failed: {e}", "warn")
                self.ranker = None


    def get_safe_batch_size(self):
        available_gb = self.available_gb
        is_heavy = 'nomic' in self.model_name or 'large' in self.model_name
        
        if available_gb > 16:
            base_batch = 128 if is_heavy else 256
        elif available_gb > 8:
            base_batch = 64 if is_heavy else 128
        elif available_gb > 4:
            base_batch = 32 if is_heavy else 64
        else:
            base_batch = 16
        
        ref_chunk_size = 500
        scale_factor = ref_chunk_size / max(1, self.chunk_size)
        adjusted_batch = int(base_batch * scale_factor)
        
        return max(1, adjusted_batch)


    def load_index(self, vec_path, meta_path):
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            try:
                embeddings = np.load(vec_path)
                faiss.normalize_L2(embeddings)

                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                
                d = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(embeddings)
                
                os.utime(vec_path, None)
                os.utime(meta_path, None)
                
                log(f"Index loaded from cache: {len(self.chunks)} vectors ready.", "success")
                return True
            except Exception as e:
                log(f"Cache load failed: {e}", "warn")
                return False
        return False


    def build_index(self, chunks, vec_path, meta_path, save_to_disk=True):
        self.load_model()
        self.chunks = chunks
        
        batch_size = self.get_safe_batch_size()
        log(f"Encoding {len(chunks)} chunks (Batch: {batch_size}, ChunkSize: {self.chunk_size})...", "info")
        
        texts = [c["text_embed"] for c in chunks]
        embeddings_list = []
        
        try:
            gen = self.model.embed(texts, batch_size=batch_size)
            
            for vec in tqdm(gen, total=len(texts), file=sys.stdout, leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                embeddings_list.append(vec)
                
        except Exception as e:
            log(f"Encoding process crashed: {e}", "error")
            log("Retrying with batch_size=1...", "warn")
            
            embeddings_list = []
            try:
                gen = self.model.embed(texts, batch_size=1)
                for vec in tqdm(gen, total=len(texts), file=sys.stdout, leave=False):
                    embeddings_list.append(vec)
            except Exception as e2:
                 log(f"Critical failure on chunk: {e2}", "error")
                 return

        embeddings = np.array(embeddings_list, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        if save_to_disk:
            np.save(vec_path, embeddings)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f)
        
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        if save_to_disk:
            log("Index built and saved to cache.", "success")
        else:
            log("Index built in memory (NoCache mode).", "success")


    def search(self, query, k, use_rerank=True, factor=5, rerank_model_name=None):
        self.load_model()
        if not self.index:
            return []

        log("Vectorizing query...", "info")
        initial_k = k * factor if use_rerank else k
        
        query_vec = list(self.model.embed([query]))[0].astype(np.float32)
        query_vec = query_vec.reshape(1, -1)
        
        faiss.normalize_L2(query_vec)
        
        D, I = self.index.search(query_vec, initial_k)
        
        candidates = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.chunks):
                candidates.append((score, self.chunks[idx]))

        if not candidates:
            return []

        if not use_rerank:
            log(f"Returning top {k} results (no rerank).", "info")
            return candidates[:k]

        log(f"Retrieved {len(candidates)} candidates. Initializing reranker...", "info")
        self.load_ranker(rerank_model_name)
        
        if self.ranker:
            try:
                cand_texts = [c[1]['text_raw'] for c in candidates]
                raw_scores = list(self.ranker.rerank(query, cand_texts))
                
                reranked = []
                for idx, raw_score in enumerate(raw_scores):
                    prob_score = sigmoid(raw_score)
                    reranked.append((prob_score, candidates[idx][1]))
                
                reranked.sort(key=lambda x: x[0], reverse=True)
                log("Reranking complete.", "success")
                return reranked[:k]
            except Exception as e:
                log(f"Reranking warn: {e}", "warn")
                
        return candidates[:k]