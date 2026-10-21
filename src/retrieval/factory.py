from robust_retrieval.retrieval.bm25 import BM25Retriever
from robust_retrieval.retrieval.dense import DenseRetriever


RETRIEVER_BUILDERS = {
    "bm25": build_bm25,
    "dense": build_dense,
}

def build_bm25(config, documents):
    return BM25Retriever(documents)

def build_dense(config, documents):
    dense_config = config["dense"]

    retriever = DenseRetriever(
        model_name=dense_config["model_name"],
        batch_size=dense_config.get("batch_size", 64),
        device=dense_config.get("device"),
    )

    retriever.index_documents(documents)
    return retriever

def build_retriever(config, documents):
    model_type = config["retrieval"]["model"]

    if model_type not in RETRIEVER_BUILDERS:
        raise ValueError(f"Unknown retrieval model: {model_type}")

    return RETRIEVER_BUILDERS[model_type](config, documents)