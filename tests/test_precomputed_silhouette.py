import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

from riskfolio.src.AuxFunctions import std_silhouette_score


def test_cluster_count_uses_pairwise_asset_distances():
    # A year of synthetic daily returns driven by four latent factors.
    rng = np.random.default_rng(7)
    factors = rng.normal(0, 0.01, (252, 4))
    returns = factors @ rng.normal(0, 1, (4, 16)) * 0.7
    returns += rng.normal(0, 0.01, (252, 16))
    corr = pd.DataFrame(returns).corr()
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    np.fill_diagonal(dist.values, 0)
    clustering = linkage(squareform(dist, checks=False), method="ward")

    # Independent calculation of the intended standardized silhouette scores.
    scores = []
    for k in (2, 3, 4):
        labels = cut_tree(clustering, n_clusters=k).ravel()
        s = silhouette_samples(dist, labels, metric="precomputed")
        scores.append(s.mean() / s.std())
    expected = 2 + int(np.argmax(scores))
    assert expected == 3
    actual, labels = std_silhouette_score(dist, clustering)
    assert actual == expected
    assert len(set(labels)) == expected


def test_roundoff_on_distance_diagonal_is_not_a_feature():
    rng = np.random.default_rng(7)
    factors = rng.normal(0, 0.01, (252, 4))
    returns = factors @ rng.normal(0, 1, (4, 16)) * 0.7
    returns += rng.normal(0, 0.01, (252, 16))
    dist = np.sqrt(np.clip((1 - pd.DataFrame(returns).corr()) / 2, 0, 1))
    np.fill_diagonal(dist.values, 1e-8)
    clustering = linkage(squareform(dist, checks=False), method="ward")
    original = dist.copy()
    k, _ = std_silhouette_score(dist, clustering)
    assert k == 3
    pd.testing.assert_frame_equal(dist, original)
