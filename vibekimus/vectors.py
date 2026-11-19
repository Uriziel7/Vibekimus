"""Unit-length vectors: cosine similarity is then a plain dot product."""
import numpy as np


def unit(vector):
    return vector / np.linalg.norm(vector)


def unit_rows(matrix):
    """Every row of a 2-D array scaled to length 1."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
