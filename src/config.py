DEFAULT_ARGS = {
    'k': 3,
    'chunk_size': 500,
    'overlap': 0.5,
    'rerank_factor': 5,
    'rerank_model': 'jina-v2',
    'model': 'bge-s',
}


MODELS_MAPPING = {
    'bge-s': 'BAAI/bge-small-en-v1.5',
    'bge-m': 'BAAI/bge-base-en-v1.5',
    'bge-l': 'BAAI/bge-large-en-v1.5',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'multi': 'intfloat/multilingual-e5-large',
    'code': 'jinaai/jina-embeddings-v2-base-code', 
    'nomic': 'nomic-ai/nomic-embed-text-v1.5',
}


RERANK_MODELS_MAPPING = {
    'bge-r-b': 'BAAI/bge-reranker-base',
    'bge-r-l': 'BAAI/bge-reranker-large',
    'jina-v2': 'jinaai/jina-reranker-v2-base-multilingual',
}


IGNORE_DIRS = {
    '.git', '.venv', 'venv', 'env', '.idea', '.vscode', '__pycache__', 
    '.egg-info', 'node_modules', 'dist', 'build', 'target', 'bin', 
    'obj', 'wheels', 'lib', 'scripts', 'include', 'share', 'lib64',
    'migrations', 'tests', 'test', 'data', 'assets', 'tmp', 'temp'
}


FILE_TYPES = {
    'text': {
        '.txt', '.md', '.markdown', '.rst',
        '.pdf', '.docx', '.doc', '.odt', '.rtf', 
        '.tsv', '.log', '.tex', '.csv'
    },
    'code': {
        '.py', '.pyw', '.js', '.jsx', '.ts', '.tsx',
        '.java', '.cpp', '.c', '.h', '.hpp', '.cs', 
        '.go', '.rs', '.php', '.rb', '.sh', '.bash', '.zsh',
        '.sql', '.html', '.css', '.scss', '.less',
        '.json', '.yaml', '.yml', '.xml', '.toml', '.ini',
        '.dockerfile', '.lua', '.pl', '.swift', '.kt',
    },
}