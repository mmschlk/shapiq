import numpy as np
from math import comb
import itertools

def index_to_subset(idx, d, ell):
    # Convert index to subset
    for k in range(1, ell + 1):
        c = comb(d, k)
        if idx < c:
            subset = []
            x = 0
            for i in range(k):
                while comb(d - x - 1, k - i - 1) <= idx:
                    idx -= comb(d - x - 1, k - i - 1)
                    x += 1
                subset.append(x)
                x += 1
            return tuple(subset)
        idx -= c
    raise IndexError

def subset_to_index(subset, d, ell):
    # Convert subset to index
    k = len(subset)
    idx = sum(comb(d, i) for i in range(1, k))
    last = -1
    for i in range(k):
        for j in range(last + 1, subset[i]):
            idx += comb(d - j - 1, k - i - 1)
        last = subset[i]
    return idx

get_d_ell = lambda d, ell: np.sum([comb(d, k) for k in range(1, ell + 1)]).astype(int)

def build_AtA(d, ell):
    # Build the AtA matrix, assuming all subsets of size 1 to ell are included
    AtA_entries = {0:0}
    for overlap in range(1, 2*ell+1):
        AtA_entries[overlap] = np.sum([
            comb(d - overlap, k - overlap) / ( comb(d, k) * k * (d-k)) * (d-1) for k in range(overlap, d)
        ])

    d_ell = get_d_ell(d, ell)
    AtA = np.zeros((d_ell, d_ell))
    for row_idx in range(d_ell):
        row_subset = index_to_subset(row_idx, d, ell)
        assert subset_to_index(row_subset, d, ell) == row_idx, "Indexing error"
        for col_idx in range(row_idx, d_ell):
            col_subset = index_to_subset(col_idx, d, ell)
            union_size = len(set(row_subset).union(set(col_subset)))
            AtA[row_idx, col_idx] = AtA[col_idx, row_idx] = AtA_entries[union_size]

    return AtA

def get_leverage_score(AtA, d, ell, row_S):
    # Compute leverage score for row_S
    d_ell = get_d_ell(d, ell)
    proj = np.eye(d_ell) - np.ones((d_ell, d_ell)) / d_ell
    mu_S = (d-1) / ( comb(d, len(row_S)) * len(row_S) * (d - len(row_S)) )

    A_S = np.zeros((1,d_ell))
    for set_size in range(1, min(ell, len(row_S))+1):
        for subset in itertools.combinations(row_S, set_size):
            col_idx = subset_to_index(subset, d, ell)
            A_S[0, col_idx] = np.sqrt(mu_S)

    PA_S = proj @ A_S.T
    AtA_invA_S = np.linalg.solve(proj @ AtA @ proj, PA_S)
    lev_score = PA_S.T @ AtA_invA_S

    return lev_score[0,0]

def get_leverage_scores(d, ell, normalize=True):
    # Compute leverage scores for all subsets of size 1 to ell
    AtA = build_AtA(d, ell)
    lev_by_dist = {}
    # By symmetry, if all subsets of size 1,...,ell are included,
    # the leverage scores are the same for all subsets of the same size.
    for set_size in range(1, d):
        row_S = set(range(set_size))
        lev_by_dist[set_size] = get_leverage_score(AtA, d, ell, row_S)
    if normalize:
        total = np.sum([
            lev_by_dist[size] * comb(d, size) for size in lev_by_dist
        ])
        for size in lev_by_dist:
            lev_by_dist[size] /= total

    return lev_by_dist


if __name__ == "__main__":
    d = 20
    ell = 1
    print(get_leverage_scores(d, ell))
