from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from edge_switching.qos_model import Criterion

SAATY_RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

STANDARD_CRITERIA: List[Criterion] = [
    Criterion.LATENCY,
    Criterion.EFFECTIVE_BW,
    Criterion.SERVER_LOAD,
    Criterion.PACKET_LOSS,
    Criterion.AVAILABILITY,
]


def compute_ahp_weights(comparison_matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    matrix = np.array(comparison_matrix, dtype=float)
    n = matrix.shape[0]
    if matrix.shape[1] != n:
        raise ValueError(f"AHP comparison matrix must be square, got {matrix.shape}")
    if n not in SAATY_RANDOM_INDEX:
        raise ValueError(f"Dimension {n} not supported in Saaty RI table")

    for i in range(n):
        if not np.isclose(matrix[i, i], 1.0):
            raise ValueError(f"Diagonal element matrix[{i},{i}] must be 1.0")
        for j in range(i + 1, n):
            if not np.isclose(matrix[i, j] * matrix[j, i], 1.0, atol=1e-3):
                raise ValueError(f"Matrix is not reciprocal at ({i},{j}): {matrix[i, j]} vs {matrix[j, i]}")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = int(np.argmax(np.real(eigenvalues)))
    lambda_max = float(np.real(eigenvalues[max_idx]))

    principal_vector = np.real(eigenvectors[:, max_idx])
    if np.any(principal_vector < 0):
        principal_vector = -principal_vector
    weights = principal_vector / np.sum(principal_vector)

    if n <= 2:
        ci = 0.0
        cr = 0.0
    else:
        ci = float((lambda_max - n) / (n - 1))
        ri = SAATY_RANDOM_INDEX[n]
        cr = float(ci / ri) if ri > 0 else 0.0

    return weights, ci, cr

AHP_VIDEO_STREAMING_MATRIX = np.array([
    [1.0,     1.0/3.0, 2.0,     1.0/2.0, 1.0/4.0],  # Latency
    [3.0,     1.0,     4.0,     2.0,     1.0/2.0],  # Effective Bandwidth
    [1.0/2.0, 1.0/4.0, 1.0,     1.0/3.0, 1.0/5.0],  # Server Load
    [2.0,     1.0/2.0, 3.0,     1.0,     1.0/3.0],  # Packet Loss
    [4.0,     2.0,     5.0,     3.0,     1.0],      # Availability
], dtype=float)

AHP_INTERACTIVE_MATRIX = np.array([
    [1.0,     4.0,     3.0,     2.0,     1.0/2.0],  # Latency
    [1.0/4.0, 1.0,     1.0/2.0, 1.0/3.0, 1.0/5.0],  # Effective Bandwidth
    [1.0/3.0, 2.0,     1.0,     1.0/2.0, 1.0/4.0],  # Server Load
    [1.0/2.0, 3.0,     2.0,     1.0,     1.0/3.0],  # Packet Loss
    [2.0,     5.0,     4.0,     3.0,     1.0],      # Availability
], dtype=float)

AHP_BALANCED_MATRIX = np.array([
    [1.0,     1.0,     2.0,     1.0,     1.0/2.0],  # Latency
    [1.0,     1.0,     2.0,     1.0,     1.0/2.0],  # Effective Bandwidth
    [1.0/2.0, 1.0/2.0, 1.0,     1.0/2.0, 1.0/3.0],  # Server Load
    [1.0,     1.0,     2.0,     1.0,     1.0/2.0],  # Packet Loss
    [2.0,     2.0,     3.0,     2.0,     1.0],      # Availability
], dtype=float)

def get_literature_ahp_profile(
    profile_name: str = "video_streaming"
) -> Tuple[Dict[Criterion, float], float, float]:
    matrices = {
        "video_streaming": AHP_VIDEO_STREAMING_MATRIX,
        "interactive_gaming": AHP_INTERACTIVE_MATRIX,
        "balanced_edge": AHP_BALANCED_MATRIX,
    }
    if profile_name not in matrices:
        raise ValueError(f"Unknown AHP profile: {profile_name}. Options: {list(matrices.keys())}")

    matrix = matrices[profile_name]
    weights, ci, cr = compute_ahp_weights(matrix)

    # Saaty consistency threshold: CR must be < 0.10
    if cr > 0.10:
        raise ValueError(f"AHP Profile {profile_name} failed consistency check: CR={cr:.4f} > 0.10")

    weight_dict = {crit: float(weights[i]) for i, crit in enumerate(STANDARD_CRITERIA)}
    return weight_dict, ci, cr

# OBJECTIVE WEIGHTING METHODS

def compute_entropy_weights(decision_matrix_U: np.ndarray) -> np.ndarray:
    U = np.array(decision_matrix_U, dtype=float)
    M, K = U.shape
    if M <= 1:
        return np.ones(K, dtype=float) / float(K)

    col_sums = np.sum(U, axis=0)
    col_sums = np.where(col_sums == 0, 1e-12, col_sums)
    P = U / col_sums

    entropy = np.zeros(K, dtype=float)
    denom = np.log(M)
    for j in range(K):
        p_col = P[:, j]
        valid = p_col > 1e-12
        if np.any(valid):
            entropy[j] = -np.sum(p_col[valid] * np.log(p_col[valid])) / denom
        else:
            entropy[j] = 1.0
    divergence = 1.0 - np.clip(entropy, 0.0, 1.0)
    total_div = float(np.sum(divergence))
    if total_div <= 1e-12:
        return np.ones(K, dtype=float) / float(K)

    return divergence / total_div

def compute_critic_weights(decision_matrix_U: np.ndarray) -> np.ndarray:
    U = np.array(decision_matrix_U, dtype=float)
    M, K = U.shape
    if M <= 2:
        return np.ones(K, dtype=float) / float(K)

    std_devs = np.std(U, axis=0, ddof=1)

    corr_matrix = np.eye(K, dtype=float)
    for j in range(K):
        for k in range(j + 1, K):
            if std_devs[j] > 1e-8 and std_devs[k] > 1e-8:
                r = float(np.corrcoef(U[:, j], U[:, k])[0, 1])
                if np.isnan(r):
                    r = 0.0
            else:
                r = 0.0
            corr_matrix[j, k] = r
            corr_matrix[k, j] = r

    C = np.zeros(K, dtype=float)
    for j in range(K):
        conflict_sum = float(np.sum(1.0 - corr_matrix[j, :]))
        C[j] = std_devs[j] * conflict_sum

    total_C = float(np.sum(C))
    if total_C <= 1e-12:
        return np.ones(K, dtype=float) / float(K)

    return C / total_C

def compute_equal_weights(K: int = 5) -> np.ndarray:
    return np.ones(K, dtype=float) / float(K)

def get_all_weight_methods(
    decision_matrix_U: Optional[np.ndarray] = None
) -> Dict[str, Dict[Criterion, float]]:
    res = {}
    for prof in ["video_streaming", "interactive_gaming", "balanced_edge"]:
        w_dict, _, _ = get_literature_ahp_profile(prof)
        res[f"AHP-{prof}"] = w_dict

    eq_w = compute_equal_weights(len(STANDARD_CRITERIA))
    res["Equal"] = {c: float(eq_w[i]) for i, c in enumerate(STANDARD_CRITERIA)}

    if decision_matrix_U is not None and len(decision_matrix_U) >= 2:
        ent_w = compute_entropy_weights(decision_matrix_U)
        crit_w = compute_critic_weights(decision_matrix_U)
        res["Entropy"] = {c: float(ent_w[i]) for i, c in enumerate(STANDARD_CRITERIA)}
        res["CRITIC"] = {c: float(crit_w[i]) for i, c in enumerate(STANDARD_CRITERIA)}

    return res