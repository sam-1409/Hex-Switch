"""
HCI-Guided and Objective Criteria Weighting Module.
Target Venue: IEEE PerCom 2027 (A4E Workshop)

Implements:
1. Analytic Hierarchy Process (AHP) with mathematical consistency checking:
   - Principal Eigenvector calculation
   - Consistency Index (CI) and Consistency Ratio (CR < 0.10 validation per Saaty's theorem)
   - Three literature-derived profiles: Video Streaming (ITU-T P.1203 proxy),
     Low-Latency Interactive (Cloud Gaming/AR), and Balanced Edge.
2. Objective Weighting Methods:
   - Shannon Entropy Weighting (dispersion-based objective information)
   - CRITIC Method (Criteria Importance Through Intercriteria Correlation)
   - Equal Weights Baseline (1/K)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from edge_switching.qos_model import Criterion

# Saaty's empirical Random Index (RI) lookup table for n = 1 .. 10
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

# Standard criteria ordering used across all matrices
STANDARD_CRITERIA: List[Criterion] = [
    Criterion.LATENCY,
    Criterion.EFFECTIVE_BW,
    Criterion.SERVER_LOAD,
    Criterion.PACKET_LOSS,
    Criterion.AVAILABILITY,
]


def compute_ahp_weights(comparison_matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute priority weights, Principal Eigenvalue (lambda_max),
    Consistency Index (CI), and Consistency Ratio (CR) for a pairwise comparison matrix.

    Returns:
        weights: np.ndarray of shape (n,) normalized to sum to 1.0.
        ci: Consistency Index = (lambda_max - n) / (n - 1)
        cr: Consistency Ratio = CI / RI[n]
    """
    matrix = np.array(comparison_matrix, dtype=float)
    n = matrix.shape[0]
    if matrix.shape[1] != n:
        raise ValueError(f"AHP comparison matrix must be square, got {matrix.shape}")
    if n not in SAATY_RANDOM_INDEX:
        raise ValueError(f"Dimension {n} not supported in Saaty RI table")

    # Verify reciprocity: A[i, j] * A[j, i] == 1.0
    for i in range(n):
        if not np.isclose(matrix[i, i], 1.0):
            raise ValueError(f"Diagonal element matrix[{i},{i}] must be 1.0")
        for j in range(i + 1, n):
            if not np.isclose(matrix[i, j] * matrix[j, i], 1.0, atol=1e-3):
                raise ValueError(f"Matrix is not reciprocal at ({i},{j}): {matrix[i, j]} vs {matrix[j, i]}")

    # Calculate principal eigenvector using exact eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = int(np.argmax(np.real(eigenvalues)))
    lambda_max = float(np.real(eigenvalues[max_idx]))

    principal_vector = np.real(eigenvectors[:, max_idx])
    # Ensure all components positive
    if np.any(principal_vector < 0):
        principal_vector = -principal_vector
    weights = principal_vector / np.sum(principal_vector)

    # Consistency Index & Ratio
    if n <= 2:
        ci = 0.0
        cr = 0.0
    else:
        ci = float((lambda_max - n) / (n - 1))
        ri = SAATY_RANDOM_INDEX[n]
        cr = float(ci / ri) if ri > 0 else 0.0

    return weights, ci, cr


# ---------------------------------------------------------------------------
# LITERATURE-DERIVED AHP PAIRWISE MATRICES
# Criteria order: [LATENCY, EFFECTIVE_BW, SERVER_LOAD, PACKET_LOSS, AVAILABILITY]
# ---------------------------------------------------------------------------

# 1. Video Streaming Profile (ITU-T P.1203 / ACM Multimedia QoE standard):
#    Effective bandwidth and packet loss are prioritized over raw latency;
#    server availability is non-negotiable.
AHP_VIDEO_STREAMING_MATRIX = np.array([
    [1.0,     1.0/3.0, 2.0,     1.0/2.0, 1.0/4.0],  # Latency
    [3.0,     1.0,     4.0,     2.0,     1.0/2.0],  # Effective Bandwidth
    [1.0/2.0, 1.0/4.0, 1.0,     1.0/3.0, 1.0/5.0],  # Server Load
    [2.0,     1.0/2.0, 3.0,     1.0,     1.0/3.0],  # Packet Loss
    [4.0,     2.0,     5.0,     3.0,     1.0],      # Availability
], dtype=float)

# 2. Low-Latency Interactive Profile (Cloud Gaming / Teleoperation / AR):
#    Low round-trip latency and zero packet loss are prioritized over high bitrate.
AHP_INTERACTIVE_MATRIX = np.array([
    [1.0,     4.0,     3.0,     2.0,     1.0/2.0],  # Latency
    [1.0/4.0, 1.0,     1.0/2.0, 1.0/3.0, 1.0/5.0],  # Effective Bandwidth
    [1.0/3.0, 2.0,     1.0,     1.0/2.0, 1.0/4.0],  # Server Load
    [1.0/2.0, 3.0,     2.0,     1.0,     1.0/3.0],  # Packet Loss
    [2.0,     5.0,     4.0,     3.0,     1.0],      # Availability
], dtype=float)

# 3. Balanced Edge Profile:
#    Moderate balance across edge resource dimensions.
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
    """
    Retrieve literature-derived AHP criteria weights with consistency verification.

    Returns:
        (weight_dict, consistency_index, consistency_ratio)
    """
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


# ===========================================================================
# OBJECTIVE WEIGHTING METHODS
# ===========================================================================

def compute_entropy_weights(decision_matrix_U: np.ndarray) -> np.ndarray:
    """
    Shannon Entropy Objective Weighting.
    Weights criteria according to dispersion across candidate alternatives:
        p_ij = u_ij / sum_i(u_ij)
        E_j = -1 / ln(M) * sum_i(p_ij * ln(p_ij))
        d_j = 1 - E_j
        w_j = d_j / sum(d_k)
    """
    U = np.array(decision_matrix_U, dtype=float)
    M, K = U.shape
    if M <= 1:
        return np.ones(K, dtype=float) / float(K)

    # Shift slightly away from 0 for numerical stability
    col_sums = np.sum(U, axis=0)
    # Avoid zero division
    col_sums = np.where(col_sums == 0, 1e-12, col_sums)
    P = U / col_sums

    # Compute entropy per criterion
    entropy = np.zeros(K, dtype=float)
    denom = np.log(M)
    for j in range(K):
        p_col = P[:, j]
        # Only compute p * log(p) for non-zero probabilities (lim p->0 p*log(p) = 0)
        valid = p_col > 1e-12
        if np.any(valid):
            entropy[j] = -np.sum(p_col[valid] * np.log(p_col[valid])) / denom
        else:
            entropy[j] = 1.0  # Zero information dispersion

    # Degree of divergence / contrast intensity
    divergence = 1.0 - np.clip(entropy, 0.0, 1.0)
    total_div = float(np.sum(divergence))
    if total_div <= 1e-12:
        return np.ones(K, dtype=float) / float(K)

    return divergence / total_div


def compute_critic_weights(decision_matrix_U: np.ndarray) -> np.ndarray:
    """
    CRITIC (Criteria Importance Through Intercriteria Correlation) Weighting.
    Measures both contrast intensity (standard deviation) and conflict (correlation):
        C_j = sigma_j * sum_k(1 - r_jk)
        w_j = C_j / sum(C_k)
    """
    U = np.array(decision_matrix_U, dtype=float)
    M, K = U.shape
    if M <= 2:
        return np.ones(K, dtype=float) / float(K)

    # 1. Contrast intensity: standard deviation per criterion
    std_devs = np.std(U, axis=0, ddof=1)

    # 2. Conflict: Pearson correlation matrix
    # If standard deviation is 0, correlation is undefined; treat as 0 correlation
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

    # 3. Information quantity C_j
    C = np.zeros(K, dtype=float)
    for j in range(K):
        conflict_sum = float(np.sum(1.0 - corr_matrix[j, :]))
        C[j] = std_devs[j] * conflict_sum

    total_C = float(np.sum(C))
    if total_C <= 1e-12:
        return np.ones(K, dtype=float) / float(K)

    return C / total_C


def compute_equal_weights(K: int = 5) -> np.ndarray:
    """Equal weighting baseline: w_k = 1/K."""
    return np.ones(K, dtype=float) / float(K)


def get_all_weight_methods(
    decision_matrix_U: Optional[np.ndarray] = None
) -> Dict[str, Dict[Criterion, float]]:
    """
    Assemble complete comparative suite of subjective (AHP) and objective (Entropy, CRITIC, Equal) weights.
    """
    res = {}
    # 1. Literature AHP profiles
    for prof in ["video_streaming", "interactive_gaming", "balanced_edge"]:
        w_dict, _, _ = get_literature_ahp_profile(prof)
        res[f"AHP-{prof}"] = w_dict

    # 2. Equal weights
    eq_w = compute_equal_weights(len(STANDARD_CRITERIA))
    res["Equal"] = {c: float(eq_w[i]) for i, c in enumerate(STANDARD_CRITERIA)}

    # 3. Objective weights if decision matrix provided
    if decision_matrix_U is not None and len(decision_matrix_U) >= 2:
        ent_w = compute_entropy_weights(decision_matrix_U)
        crit_w = compute_critic_weights(decision_matrix_U)
        res["Entropy"] = {c: float(ent_w[i]) for i, c in enumerate(STANDARD_CRITERIA)}
        res["CRITIC"] = {c: float(crit_w[i]) for i, c in enumerate(STANDARD_CRITERIA)}

    return res
