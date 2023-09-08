from .bert import BERTEmbedding # 기존 Embedding
from .bert_hyper import BERTEmbedding_Hyper # Hyperbolic Embedding

EMBEDDING = {
    BERTEmbedding.code(): BERTEmbedding,
    BERTEmbedding_Hyper.code(): BERTEmbedding_Hyper
}

def model_embedding(args):
    embedding = EMBEDDING[args.model_embedding]
    return embedding