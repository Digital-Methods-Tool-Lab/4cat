# original by William Leif Hamilton, https://github.com/williamleif/histwords
# See 'license' in the 'histwords' folder
# File changed for Py3 compatibility and to make import paths work for 4CAT

import numpy as np
import os

import pickle

from backend.lib.histwords.representations import sparse_io

def load_matrix(f):
    if not f.endswith('.bin'):
        f += ".bin"
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()})
    return sparse_io.retrieve_mat_as_coo(f).tocsr()

def load_vocabulary(mat, path):
    if os.path.isfile(path.split(".")[0] + "-index.pkl"):
        path = path.split(".")[0] + "-index.pkl"
    else:
        print("Could not find local index. Attempting to load directory wide index...")
        path = "/".join(path.split("/")[:-1]) + "/index.pkl"

    with open(path) as vocabfile:
        index = pickle.load(vocabfile)

    vocab = sorted(index, key = lambda word : index[word])
    iw = vocab[:mat.shape[0]]
    ic = vocab[:mat.shape[1]]
    return iw, ic
