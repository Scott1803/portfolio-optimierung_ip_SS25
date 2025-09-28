from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Sequence, Optional

from scipy.optimize import minimize, OptimizeResult

# Type-only imports; keep runtime dependency light/flexible
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from utils.read_data import SocialImpactOptimizationInputData, IdMappingInputData
    except Exception:  # pragma: no cover
        from read_data import SocialImpactOptimizationInputData, IdMappingInputData
else:
    SocialImpactOptimizationInputData = object  # type: ignore
    IdMappingInputData = object  # type: ignore


# =============================
# Helpers (vectorized)
# =============================

def _project_country_indicator_tensor(
    improvements_indicators_x_projects: np.ndarray,  # (K, I) or (indicators, projects)
    country_indicator_scores_weighted: np.ndarray,   # (K, 1 + C) -> [weight, country1..countryC]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build N with shape (I, C, K):
      N[i, c, k] = R[i, k] * need[c, k],
      where need[c, k] = 1 - baseline[c, k] from the scores table.
    """
    if country_indicator_scores_weighted.ndim != 2 or country_indicator_scores_weighted.shape[1] < 2:
        raise ValueError("country_indicator_scores_weighted must be (K, 1 + C)")

    K = country_indicator_scores_weighted.shape[0]
    C = country_indicator_scores_weighted.shape[1] - 1

    baseline_ck = country_indicator_scores_weighted[:, 1:].T  # -> (C, K)
    need_ck = 1.0 - baseline_ck

    R = improvements_indicators_x_projects
    if R.shape[0] == K:  # transpose (K, I) -> (I, K)
        R = R.T
    I = R.shape[0]
    if R.shape[1] != K:
        raise ValueError(f"Improvement matrix inconsistent: R has K={R.shape[1]} but scores have K={K}")

    N = R[:, None, :] * need_ck[None, :, :]   # (I, C, K)
    return N.astype(float, copy=False), need_ck


def _project_value_added_coefficients(N: np.ndarray, project_country_weights: np.ndarray) -> np.ndarray:
    """
    proj_coef[i] = Σ_{c,k} A[i,c] * N[i,c,k]
    Accepts A as (I, C) or (C, I).
    """
    if project_country_weights.shape == (N.shape[0], N.shape[1]):  # (I, C)
        A = project_country_weights
    elif project_country_weights.shape == (N.shape[1], N.shape[0]):  # (C, I)
        A = project_country_weights.T
    else:
        raise ValueError(
            f"project_country_weights must be (I,C) or (C,I), got {project_country_weights.shape} while N is {N.shape}"
        )
    proj_coef = (A[:, :, None] * N).sum(axis=(1, 2))  # (I,)
    return proj_coef


def _pairs_from_dependency_matrix(dep_matrix: np.ndarray | None) -> tuple[tuple[int, int], ...]:
    """
    Convert a 0/1 dependency matrix into pairs (dep, base) meaning y[dep] ≤ y[base].

    Convention (as in ia_projekt_abhaengigkeiten):
      • Columns = projects WITH dependencies
      • Rows    = the dependencies of those columns
      → If dep_matrix[row, col] == 1 → project 'col' depends on project 'row'.

    Returns a tuple of zero-based (dep, base) pairs.
    """
    if dep_matrix is None:
        return tuple()
    M = np.asarray(dep_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("dependency_matrix must be a square matrix (I x I).")
    I = M.shape[0]
    pairs: list[tuple[int, int]] = []
    for base in range(I):         # row index
        for dep in range(I):      # col index
            if base == dep:
                continue
            if M[base, dep] >= 0.5:  # treat ≥0.5 as '1'
                pairs.append((dep, base))
    return tuple(pairs)


# =============================
# Public API
# =============================

def generate_project_country_indicator_tensor(
    input_data: "SocialImpactOptimizationInputData",
) -> np.ndarray:
    N, _ = _project_country_indicator_tensor(
        improvements_indicators_x_projects=input_data.sia_indikatoren_verbesserungen,
        country_indicator_scores_weighted=input_data.country_indicator_scores_weighted,
    )
    return N


def generate_project_value_added_coefficients(pci_tensor: np.ndarray, country_weights: np.ndarray) -> np.ndarray:
    return _project_value_added_coefficients(pci_tensor, country_weights)


@dataclass
class _PenaltyContext:
    N: np.ndarray                     # (I, C, K)
    proj_coef: np.ndarray             # (I,)
    m: np.ndarray                     # (I,)
    xmax: np.ndarray                  # (I,)
    B_total: float
    exclusivity_pairs: tuple[tuple[int, int], ...]  # pairs of zero-based indices
    dep_matrix: np.ndarray | None = None 


def _default_exclusivity_pairs(I: int) -> tuple[tuple[int, int], ...]:
    """
    Choose exclusivity pairs based on known dataset layouts.
    New order (I>=12): P3↔P11, P4↔P12 = (2,10), (3,11) (zero-based).
    Fallback legacy: (2,3), (4,5).
    """
    if I >= 12:
        return ((2, 10), (3, 11))
    return ((2, 3), (4, 5))


def _make_start(m: np.ndarray, xmax: np.ndarray, B_total: float) -> np.ndarray:
    I = len(m)
    x0 = np.minimum(xmax, np.full(I, B_total / max(1.0, I)))
    y0 = np.full(I, 0.5)
    return np.concatenate([x0, y0])


def _make_bounds(xmax: np.ndarray) -> list[tuple[float, float]]:
    I = len(xmax)
    b_x = [(0.0, float(xmax[i])) for i in range(I)]
    b_y = [(0.0, 1.0)] * I
    return b_x + b_y


def penalty_objective_function(
    vars_flat: np.ndarray,
    ctx: _PenaltyContext,
    RHO_BUDGET: float = 1e6,   # ↑ stronger and symmetric
    RHO_BOX: float = 1e6,
    RHO_EXC1: float = 1e5,
    RHO_BIN: float = 1e5,
    RHO_DEP: float = 1e8,
) -> float:
    I = ctx.m.shape[0]
    x = vars_flat[:I]
    y = vars_flat[I:]

    # Utility
    u = ctx.xmax - ctx.m
    den = np.log1p(u)
    den = np.where(den > 0, den, 1.0)
    t = np.minimum(u, np.maximum(0.0, y * x - ctx.m))
    phi = np.log1p(t) / den
    fneg = -np.sum(ctx.proj_coef * phi)

    pen = 0.0

    # >>> Symmetric budget tracking (enforce Σ y_i x_i ≈ B_total) <<<
    spent = float(np.sum(y * x))
    pen += RHO_BUDGET * (spent - ctx.B_total) ** 2

    # Soft box coupling
    pen += RHO_BOX * (
        np.sum(np.maximum(0.0, ctx.m * y - x) ** 2) +
        np.sum(np.maximum(0.0, x - ctx.xmax * y) ** 2)
    )

    # Exclusivity
    for a, b in getattr(ctx, "exclusivity_pairs", ()) or ():
        if a < I and b < I:
            pen += RHO_EXC1 * max(0.0, y[a] + y[b] - 1.0) ** 2

    # Dependencies: rows = prerequisites p, cols = dependents d → enforce y_d ≤ y_p
    D = getattr(ctx, "dep_matrix", None)
    if D is not None:
        Dm = (np.asarray(D, dtype=float) >= 0.5).astype(float)
        if Dm.shape != (I, I):
            raise ValueError(f"Dependency matrix must be {I}x{I}, got {Dm.shape}")
        dep_pen = 0.0
        for p in range(I):
            for d in range(I):
                if Dm[p, d] >= 0.5:
                    dep_pen += max(0.0, float(y[d] - y[p])) ** 2
        pen += RHO_DEP * dep_pen

    # Encourage binary y
    pen += RHO_BIN * np.sum((y * (1.0 - y)) ** 2)

    return float(fneg + pen)


def run_optimization(
    input_data: "SocialImpactOptimizationInputData",
    total_budget: float,
    maxiter: int = 800,
    ftol: float = 1e-9,
) -> OptimizeResult:
    N = generate_project_country_indicator_tensor(input_data)
    proj_coef = generate_project_value_added_coefficients(N, input_data.project_country_weights)
    m = np.asarray(input_data.projects_budget_min_max[0, :], dtype=float)
    xmax = np.asarray(input_data.projects_budget_min_max[1, :], dtype=float)
    I = m.shape[0]

    # Try both common names; one of them should be present in your unified reader output
    dep_mat = getattr(input_data, "ia_projekt_abhaengigkeiten", None)
    if dep_mat is None:
        dep_mat = getattr(input_data, "sia_projekt_abhaengigkeiten", None)

    ctx = _PenaltyContext(
        N=N,
        proj_coef=proj_coef,
        m=m,
        xmax=xmax,
        B_total=total_budget,
        exclusivity_pairs=_default_exclusivity_pairs(I),
        dep_matrix=dep_mat,
    )

    x0 = _make_start(m, xmax, total_budget)
    bounds = _make_bounds(xmax)
    res = minimize(
        penalty_objective_function,
        x0,
        args=(ctx,),
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(maxiter=maxiter, ftol=ftol, disp=False),
    )
    return res


def _postprocess_solution(
    res: OptimizeResult,
    m: np.ndarray,
    xmax: np.ndarray,
    B_total: float,
    exclusivity_pairs: Sequence[tuple[int, int]] | None = None,
    dependency_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    I = m.shape[0]
    x = np.array(res.x[:I], dtype=float, copy=True)
    y = np.array(res.x[I:], dtype=float, copy=True)

    # Round y to binary
    y = (y >= 0.5).astype(float)

    # Enforce exclusivity (keep larger x)
    pairs = tuple(exclusivity_pairs or _default_exclusivity_pairs(I))
    for a, b in pairs:
        if a < I and b < I and y[a] + y[b] > 1:
            y[b if x[b] < x[a] else a] = 0.0

    # HARD dependency closure: if a chosen project lacks any prerequisite, drop it
    if dependency_matrix is not None:
        Dm = (np.asarray(dependency_matrix, dtype=float) >= 0.5).astype(int)
        if Dm.shape != (I, I):
            raise ValueError(f"Dependency matrix must be {I}x{I}, got {Dm.shape}")
        changed = True
        while changed:
            changed = False
            for d in range(I):
                if y[d] < 0.5:
                    continue
                prereqs = np.where(Dm[:, d] == 1)[0]
                if prereqs.size > 0 and (y[prereqs] < 0.5).any():
                    y[d] = 0.0
                    x[d] = 0.0
                    changed = True

    # Box coupling
    x = np.minimum(x, xmax * y)
    x = np.maximum(x, m * y)

    # Final budget reconciliation: trim if over, fill if under
    spent = float(np.sum(y * x))
    tol = 1e-9

    if spent > B_total + tol:
        active = y > 0.5
        room = x[active] - m[active]
        if room.sum() > 0:
            over = spent - B_total
            x[active] -= over * (room / room.sum())
            x[active] = np.maximum(x[active], m[active])

    elif spent < B_total - tol:
        active = y > 0.5
        # distribute additional budget across headroom to xmax
        headroom = np.maximum(0.0, xmax[active] - x[active])
        if headroom.sum() > 0:
            need = min(B_total - spent, float(headroom.sum()))
            x[active] += need * (headroom / headroom.sum())
            x[active] = np.minimum(x[active], xmax[active])
        # If no active projects (all y==0), we leave it; typically the solver won’t end here with the symmetric budget penalty.

    return x, y



def process_optimization_result(
    optimizer_result: OptimizeResult,
    input_data: "SocialImpactOptimizationInputData",
    total_budget: float | None = None,
    mappings: "IdMappingInputData" | None = None,
    dependency_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a (2, I) matrix:
      Row 0: invested amounts per project (M€)
      Row 1: **weighted social impact** per project (proj_coef[i] * φ_i)
    """
    m = np.asarray(input_data.projects_budget_min_max[0, :], dtype=float)
    xmax = np.asarray(input_data.projects_budget_min_max[1, :], dtype=float)
    I = m.shape[0]

    # Recompute needed pieces for φ and proj_coef
    N = generate_project_country_indicator_tensor(input_data)
    proj_coef = generate_project_value_added_coefficients(N, input_data.project_country_weights)

    dep_mat = getattr(input_data, "ia_projekt_abhaengigkeiten", None)
    x_opt, y_opt = _postprocess_solution(
        optimizer_result,
        m,
        xmax,
        B_total=float(total_budget or 0.0),
        exclusivity_pairs=_default_exclusivity_pairs(I),
        dependency_matrix=dep_mat,  # ← pass here
    )


    # φ_i
    u = xmax - m
    den = np.log1p(u)
    den = np.where(den > 0, den, 1.0)
    t = np.minimum(u, np.maximum(0.0, y_opt * x_opt - m))
    phi = np.divide(np.log1p(t), den, out=np.zeros_like(den), where=den > 0)

    # Legacy-compatible impact: proj_coef * φ
    impacts = proj_coef * phi
    out = np.vstack([x_opt, impacts]).astype(float)
    return out


# --------------------------------
# Fixed-portfolio simulator (unchanged)
# --------------------------------

def _compute_phi_per_project(x: np.ndarray, y: np.ndarray, m: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    u = xmax - m
    den = np.log1p(u)
    den = np.where(den > 0, den, 1.0)
    t = np.minimum(u, np.maximum(0.0, y * x - m))
    phi = np.divide(np.log1p(t), den, out=np.zeros_like(den), where=den > 0)
    return phi


def calculate_impact_for_chosen_portfolio(
    portfolio: np.ndarray,
    input_data: "SocialImpactOptimizationInputData",
    total_budget: float | None = None,
    enforce_budget: bool = False,
) -> np.ndarray:
    """
    Given a fixed portfolio x (0 or within [m_i, xmax_i]), return:
      Row 0: possibly adjusted x
      Row 1: φ_i (0..1) per project
    """
    x = np.asarray(portfolio, dtype=float).copy()
    m = np.asarray(input_data.projects_budget_min_max[0, :], dtype=float)
    xmax = np.asarray(input_data.projects_budget_min_max[1, :], dtype=float)

    if x.shape[0] != m.shape[0]:
        raise ValueError(f"Portfolio length {x.shape[0]} does not match project count {m.shape[0]}")
    y = (x > 0).astype(float)
    x = np.clip(x, 0.0, xmax)

    if enforce_budget and (total_budget is not None):
        spent = float(np.sum(y * x))
        if spent > total_budget + 1e-9:
            active = y > 0.5
            room = np.maximum(0.0, x[active] - m[active])
            if room.sum() > 0:
                over = spent - float(total_budget)
                x[active] -= over * (room / room.sum())
                x[active] = np.maximum(x[active], m[active])

    phi = _compute_phi_per_project(x, y, m, xmax)
    return np.vstack([x, phi]).astype(float)
