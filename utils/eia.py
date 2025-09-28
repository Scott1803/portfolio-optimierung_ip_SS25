from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from scipy.optimize import milp, LinearConstraint, Bounds

# Type hints only; avoid runtime import path issues
if TYPE_CHECKING:
    try:
        from utils.read_data import EconomicImpactOptimizationInputData  # package-local path
    except Exception:  # pragma: no cover
        from read_data import EconomicImpactOptimizationInputData  # fallback
else:
    EconomicImpactOptimizationInputData = object  # type: ignore


# -----------------------------
# Value-added coefficient builder
# -----------------------------

def generate_value_added_multipliers(input_data: "EconomicImpactOptimizationInputData") -> np.ndarray:
    """Compute per-project value-added-per-million multipliers.

    For project j, country c, sector s:
        base_c = sum_s v_a[c,s] * o_mult[c,s] * sector_share[s,j]
        multiplier_j += base_c * country_share[c,j]

    Returns
    -------
    np.ndarray of shape (I,), where I is the number of projects.
    """
    # Sector split per project: (S, I)
    sector_split = np.asarray(input_data.ia_projekt_budget_aufteilung, dtype=float)  # S x I
    S, I = sector_split.shape

    # Country shares per project: (C, I)
    country_split = np.asarray(input_data.ia_laender_aufteilung, dtype=float)  # C x I
    C = country_split.shape[0]

    # ia_sektorenwerte: list of C arrays, each S x 2 -> [v_a, o_mult]
    if len(input_data.ia_sektorenwerte) != C:
        raise ValueError(
            f"ia_sektorenwerte has {len(input_data.ia_sektorenwerte)} countries but ia_laender_aufteilung has {C}."
        )

    multipliers = np.zeros(I, dtype=float)

    for c in range(C):
        sekt = np.asarray(input_data.ia_sektorenwerte[c], dtype=float)  # S x 2
        if sekt.shape[0] != S or sekt.shape[1] < 2:
            raise ValueError(
                f"Country {c} sector matrix has shape {sekt.shape}, expected (S={S}, 2)."
            )
        combined = sekt[:, 0] * sekt[:, 1]                 # v_a * o_mult  -> (S,)
        base_per_project = combined @ sector_split          # (S,) @ (S,I) -> (I,)
        multipliers += base_per_project * country_split[c]  # weight by country share for each project

    return multipliers  # (I,)


# -----------------------------
# Economic impact calculation for a fixed portfolio
# -----------------------------

def calculate_economic_impact(
    invested_amounts: np.ndarray,
    input_data: "EconomicImpactOptimizationInputData",
    total_budget: float | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return a 2×I matrix for a user-specified portfolio.

    Parameters
    ----------
    invested_amounts : np.ndarray
        Vector of length I with either 0 or a valid budget in [min_i, max_i] for each project in order.
    input_data : EconomicImpactOptimizationInputData
        Unified IA input data (limits, sector/country splits, dependencies).
    total_budget : float | None
        If provided, validates that the sum of investments does not exceed this value (within tolerance).

    Returns
    -------
    np.ndarray
        Shape (2, I): row 0 = investments; row 1 = economic value added per project.
    """
    x = np.asarray(invested_amounts, dtype=float)
    mins = np.asarray(input_data.projects_budget_min_max[0, :], dtype=float)
    maxs = np.asarray(input_data.projects_budget_min_max[1, :], dtype=float)

    if x.ndim != 1 or x.shape[0] != mins.shape[0]:
        raise ValueError(f"invested_amounts must be length {mins.shape[0]} vector. Got shape {x.shape}.")

    # Validate per-project: 0 or in [min, max]
    for i in range(x.shape[0]):
        xi = x[i]
        if xi <= tol:
            continue
        if not (mins[i] - tol <= xi <= maxs[i] + tol):
            raise ValueError(
                f"Investment for project {i+1} = {xi:.6f} must be 0 or within [{mins[i]:.6f}, {maxs[i]:.6f}]."
            )

    # Dependency check: if x[j] > 0 and D[i,j] == 1, then x[i] > 0
    deps = np.asarray(input_data.ia_projekt_abhaengigkeiten, dtype=float)  # (I, I)
    if deps.shape != (x.shape[0], x.shape[0]):
        raise ValueError(f"Dependency matrix must be {(x.shape[0], x.shape[0])}, got {deps.shape}.")
    active = x > tol
    # For each j, all i with deps[i,j]==1 must be active too
    for j in range(x.shape[0]):
        if not active[j]:
            continue
        required = np.where(deps[:, j] >= 0.5)[0]
        missing = [i for i in required if not active[i]]
        if missing:
            raise ValueError(
                f"Dependency violation for project {j+1}: requires {', '.join('P'+str(i+1) for i in missing)} to be funded."
            )

    # Budget check
    if total_budget is not None and (x.sum() - float(total_budget)) > tol:
        raise ValueError(
            f"Total invested {x.sum():.6f} exceeds budget {float(total_budget):.6f}."
        )

    # Impacts
    multipliers = generate_value_added_multipliers(input_data)  # (I,)
    impacts = x * multipliers
    return np.vstack([x, impacts]).astype(float)


# -----------------------------
# MILP: economically optimal investments
# -----------------------------

def calculate_economically_optimal_project_investments(
    input_data: "EconomicImpactOptimizationInputData",
    total_budget: float,
) -> np.ndarray:
    """Solve MILP to maximize total economic value and return a 2×I result matrix.

    Row 0 = optimal investments x (M€)
    Row 1 = per-project value added (x * multiplier)
    """
    mins = np.asarray(input_data.projects_budget_min_max[0, :], dtype=float)
    maxs = np.asarray(input_data.projects_budget_min_max[1, :], dtype=float)
    I = mins.shape[0]

    multipliers = generate_value_added_multipliers(input_data)  # (I,)

    # Decision variables: [x(0..I-1), y(0..I-1)] of length 2I
    # Objective: maximize sum(multiplier[i] * x[i]) -> minimize negative
    c = np.concatenate([-multipliers, np.zeros(I, dtype=float)])

    constraints: list[LinearConstraint] = []

    # Budget equality: sum x == total_budget
    budget_row = np.concatenate([np.ones(I, dtype=float), np.zeros(I, dtype=float)])
    constraints.append(LinearConstraint(budget_row, lb=total_budget, ub=total_budget))

    # Linking constraints per project: m_i*y_i <= x_i <= M_i*y_i
    for i in range(I):
        row_lb = np.zeros(2 * I, dtype=float)
        row_lb[i] = 1.0
        row_lb[I + i] = -mins[i]
        constraints.append(LinearConstraint(row_lb, lb=0.0, ub=np.inf))

        row_ub = np.zeros(2 * I, dtype=float)
        row_ub[i] = 1.0
        row_ub[I + i] = -maxs[i]
        constraints.append(LinearConstraint(row_ub, lb=-np.inf, ub=0.0))

    # Dependencies: rows are dependencies of columns -> if D[i,j]==1 then y_j <= y_i
    D = np.asarray(input_data.ia_projekt_abhaengigkeiten, dtype=float)
    if D.shape != (I, I):
        raise ValueError(f"Dependency matrix must be {(I, I)}, got {D.shape}.")
    for i in range(I):
        for j in range(I):
            if D[i, j] >= 0.5:
                row = np.zeros(2 * I, dtype=float)
                row[I + j] = 1.0   # + y_j
                row[I + i] = -1.0  # - y_i
                constraints.append(LinearConstraint(row, lb=-np.inf, ub=0.0))

    # Bounds: 0 <= x_i <= max_i ; 0 <= y_i <= 1
    bounds = Bounds(
        lb=np.concatenate([np.zeros(I, dtype=float), np.zeros(I, dtype=float)]),
        ub=np.concatenate([maxs, np.ones(I, dtype=float)])
    )

    # Integrality: x continuous (0), y binary (1)
    integrality = np.concatenate([np.zeros(I, dtype=int), np.ones(I, dtype=int)])

    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    x = np.asarray(result.x[:I], dtype=float)
    impacts = x * multipliers
    return np.vstack([x, impacts]).astype(float)
