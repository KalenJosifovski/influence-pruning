"""Tanimoto similarity and Butina clustering over binary fingerprints.

Adapted from the preceding towards-global-models pipeline.
"""

import numpy as np
import pandas as pd
from rdkit.ML.Cluster import Butina

MAX_DOT_PRODUCT_ELEMENTS = 8_000_000


def fingerprint_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Stack per-molecule fingerprints into a dense uint8 matrix.

    :param frame: standardised frame whose ``fingerprint`` column holds uint8 arrays.
    :returns: ``(n_molecules, n_bits)`` uint8 matrix.
    :raises ValueError: the frame is empty or contains unparsed molecules.
    """
    arrays = frame["fingerprint"].tolist()
    if not arrays:
        raise ValueError("cannot build a fingerprint matrix from an empty frame")
    if any(array is None for array in arrays):
        raise ValueError(
            "frame contains molecules without fingerprints; filter parse failures first"
        )
    return np.vstack([np.asarray(array, dtype=np.uint8) for array in arrays])


def tanimoto_matrix(fingerprints: np.ndarray) -> np.ndarray:
    """Compute the full pairwise Tanimoto similarity matrix for binary fingerprints."""
    dense = fingerprints.astype(np.float32)
    counts = dense.sum(axis=1)
    dot = dense @ dense.T
    denominator = counts[:, None] + counts[None, :] - dot
    return np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)


def max_tanimoto(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute each query molecule's maximum Tanimoto similarity to any reference molecule.

    :param query: ``(n_query, n_bits)`` binary fingerprint matrix.
    :param reference: ``(n_reference, n_bits)`` binary fingerprint matrix.
    :returns: ``(n_query,)`` float32 array of maximum similarities.
    """
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return np.zeros(query.shape[0], dtype=np.float32)
    dense_query = query.astype(np.float32)
    dense_reference = reference.astype(np.float32)
    reference_counts = dense_reference.sum(axis=1)
    best = np.zeros(dense_query.shape[0], dtype=np.float32)
    chunk_rows = max(1, MAX_DOT_PRODUCT_ELEMENTS // dense_reference.shape[0])
    for start in range(0, dense_query.shape[0], chunk_rows):
        block = dense_query[start : start + chunk_rows]
        dot = block @ dense_reference.T
        denominator = block.sum(axis=1)[:, None] + reference_counts[None, :] - dot
        similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
        best[start : start + chunk_rows] = similarity.max(axis=1)
    return np.clip(best, 0.0, 1.0)


def cluster_labels(fingerprints: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Assign integer cluster labels to molecules for the cluster bootstrap.

    :param fingerprints: ``(n_molecules, n_bits)`` binary fingerprint matrix.
    :param distance_threshold: Tanimoto distance cut, i.e. ``1 - similarity``.
    :returns: ``(n_molecules,)`` integer cluster labels.
    """
    clusters = butina_clusters(fingerprints, distance_threshold)
    labels = np.zeros(fingerprints.shape[0], dtype=int)
    for index, cluster in enumerate(clusters):
        for member in cluster:
            labels[member] = index
    return labels


def butina_clusters(
    fingerprints: np.ndarray, distance_threshold: float
) -> tuple[tuple[int, ...], ...]:
    """Cluster fingerprints with Butina clustering on Tanimoto distances.

    :param fingerprints: ``(n_molecules, n_bits)`` binary fingerprint matrix.
    :param distance_threshold: Tanimoto distance cut, i.e. ``1 - similarity``.
    :returns: tuple of clusters, each a tuple of row indices.
    """
    n_molecules = fingerprints.shape[0]
    if n_molecules == 0:
        return ()
    if n_molecules == 1:
        return ((0,),)
    distance = np.subtract(1.0, tanimoto_matrix(fingerprints), dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    condensed = distance[np.tril_indices(n_molecules, k=-1)]
    del distance
    clusters = Butina.ClusterData(condensed, n_molecules, distance_threshold, isDistData=True)
    return tuple(tuple(int(member) for member in cluster) for cluster in clusters)
