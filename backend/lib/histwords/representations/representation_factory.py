# original by William Leif Hamilton, https://github.com/williamleif/histwords
# See 'license' in the 'histwords' folder
# File changed to make import paths work for 4CAT

from backend.lib.histwords.representations.embedding import SVDEmbedding, Embedding, GigaEmbedding
from backend.lib.histwords.representations.explicit import Explicit

def create_representation(rep_type, path, *args, **kwargs):
    if rep_type == 'Explicit' or rep_type == 'PPMI':
        return Explicit.load(path, *args, **kwargs)
    elif rep_type == 'SVD':
        return SVDEmbedding(path, *args, **kwargs)
    elif rep_type == 'GIGA':
        return GigaEmbedding(path, *args, **kwargs)
    elif rep_type:
        return Embedding.load(path, *args, **kwargs)
