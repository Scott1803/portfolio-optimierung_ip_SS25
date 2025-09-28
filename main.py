from __future__ import annotations

import re
import sys
import numpy as np

# Robust imports to work in both package and local layouts
try:
    from utils.read_data import (
        read_social_impact_optimization_input_data,
        read_economic_impact_optimization_input_data,
    )
except Exception:  # pragma: no cover
    from read_data import (
        read_social_impact_optimization_input_data,
        read_economic_impact_optimization_input_data,
    )

try:
    from utils.sia_new import (
        run_optimization as run_sia_optimization,
        process_optimization_result as process_sia_result,
    )
except Exception:  # pragma: no cover
    from sia_new import (
        run_optimization as run_sia_optimization,
        process_optimization_result as process_sia_result,
    )

try:
    from utils.eia import (
        calculate_economically_optimal_project_investments as run_ia_optimization,
    )
except Exception:  # pragma: no cover
    from eia import (
        calculate_economically_optimal_project_investments as run_ia_optimization,
    )

try:
    from utils.display_results import show_table_for_results
except Exception:  # pragma: no cover
    from display_results import show_table_for_results


HEADER = r"""
===============================================================
 Optimizing investment portfolios in eastern Europe
 for economic and social impacts

 Authors: Abhiman Arjun, Emil Levin​ Blauschke,​ Scott​ Clements, Alpha Saliou Diallo,
 Till Lennart​ ​Geisel, Shady Soumaya, Loran Toga, Din Zekic​
===============================================================
To start, enter the total budget to invest
"""


def parse_budget(s: str) -> float:
    s = s.strip().replace(",", ".")
    if not re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", s):
        raise ValueError("Budget must be a number (e.g., 25, 25.0, 12,5).")
    return float(s)


def parse_split(s: str) -> tuple[int, int]:
    s = s.strip()
    if not re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", s):
        raise ValueError("Split must be two whole numbers with a slash, e.g., 40/60.")
    left, right = s.split("/")
    a = int(left.strip())
    b = int(right.strip())
    if a < 0 or b < 0 or a + b != 100:
        raise ValueError("Split numbers must be >= 0 and add up to 100.")
    return a, b


def _attach_dependency_matrix(si_input, ia_input) -> np.ndarray | None:
    """Attach IA dependency matrix to SIA input so SIA model can enforce deps."""
    dep_mat = getattr(ia_input, "ia_projekt_abhaengigkeiten", None)
    if dep_mat is None:
        dep_mat = getattr(ia_input, "sia_projekt_abhaengigkeiten", None)
    if dep_mat is None:
        return None

    M = np.asarray(dep_mat, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            f"Dependency matrix must be square; got shape {M.shape} (rows=prereqs, cols=dependents)."
        )
    M = (M >= 0.5).astype(float)
    np.fill_diagonal(M, 0.0)
    setattr(si_input, "ia_projekt_abhaengigkeiten", M)
    return M


def _validate_combined_minmax(x_combined: np.ndarray, ia_input) -> None:
    """Ensure each project is either 0 or within [min_i, max_i]."""
    mins = np.asarray(ia_input.projects_budget_min_max[0, :], dtype=float)
    maxs = np.asarray(ia_input.projects_budget_min_max[1, :], dtype=float)
    tol = 1e-9

    if x_combined.shape != mins.shape:
        raise ValueError(f"Combined vector length {x_combined.size} does not match project count {mins.size}.")

    active = x_combined > tol
    low_viol = active & (x_combined < mins - tol)
    high_viol = active & (x_combined > maxs + tol)

    if np.any(low_viol) or np.any(high_viol):
        bad = np.where(low_viol | high_viol)[0]
        msgs = [f"P{i+1}: {x_combined[i]:.3f} (min {mins[i]:.3f}, max {maxs[i]:.3f})" for i in bad]
        raise ValueError("Combined investment violates per-project limits:\n  " + "\n  ".join(msgs))


def _validate_portfolio_sum(x: np.ndarray, expected_sum: float, label: str, tol: float = 1e-6) -> None:
    s = float(np.sum(x))
    if abs(s - expected_sum) > tol:
        raise ValueError(f"{label} portfolio sums to {s:.6f} ≠ expected {expected_sum:.6f} (|Δ|>{tol}).")


def main() -> None:
    print(HEADER)
    # -------- Budget input --------
    try:
        total_budget_str = input("Total Budget (M€): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    try:
        total_budget = parse_budget(total_budget_str)
        if total_budget <= 0:
            raise ValueError("Budget must be > 0.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # -------- Split input --------
    print("\nHow should we split this budget?\n"
          "Enter two whole-number percentages that sum to 100, like '40/60'\n"
          " (left = Social Impact, right = Economic Impact)")
    try:
        split_str = input("Budget split (SIA/IA): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    try:
        sia_pct, ia_pct = parse_split(split_str)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    sia_budget = total_budget * (sia_pct / 100.0)
    ia_budget = total_budget * (ia_pct / 100.0)

    # -------- Read inputs --------
    try:
        si_input = read_social_impact_optimization_input_data()
        ia_input = read_economic_impact_optimization_input_data()
    except FileNotFoundError as e:
        print(f"Input file error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Missing sheet/table: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to read input data: {e}")
        sys.exit(1)

    # -------- Wire dependency matrix from IA → SIA --------
    try:
        _attach_dependency_matrix(si_input, ia_input)
    except Exception as e:
        print(f"Dependency matrix error: {e}")
        sys.exit(1)

    # -------- Run SIA --------
    try:
        sia_res = run_sia_optimization(si_input, total_budget=sia_budget)
        sia_out = process_sia_result(sia_res, si_input, total_budget=sia_budget)
        # sia_out[0,:] = investments; we IGNORE sia_out[1,:] (weighted), recompute φ in display
    except Exception as e:
        print(f"SIA optimization failed: {e}")
        sys.exit(1)

    # -------- Run IA --------
    try:
        ia_out = run_ia_optimization(ia_input, total_budget=ia_budget)  # (2, I)
        # ia_out[0,:] = investments; ia_out[1,:] = economic impact (M€)
    except Exception as e:
        print(f"IA optimization failed: {e}")
        sys.exit(1)

    # -------- Validate per-side sums (hard equality for IA; soft for SIA but we still check) --------
    try:
        _validate_portfolio_sum(ia_out[0, :], ia_budget, "IA")
        _validate_portfolio_sum(sia_out[0, :], sia_budget, "SIA")
    except Exception as e:
        print(f"Budget sum error:\n{e}")
        sys.exit(1)

    # -------- Validate combined per-project min/max --------
    try:
        x_combined = np.asarray(ia_out[0, :], dtype=float) + np.asarray(sia_out[0, :], dtype=float)
        _validate_combined_minmax(x_combined, ia_input)
        # Optional: also check combined total equals total_budget
        _validate_portfolio_sum(x_combined, sia_budget + ia_budget, "Combined")
    except Exception as e:
        print(f"Combined-portfolio limit error:\n{e}")
        sys.exit(1)

    # -------- Display --------
    print("\n===============================================================")
    print(f" Using total budget: {total_budget:.3f} M€ "
          f"(SIA {sia_pct}% → {sia_budget:.3f} M€, IA {ia_pct}% → {ia_budget:.3f} M€)")
    print("===============================================================\n")

    try:
        show_table_for_results(
            economic_result=ia_out,
            social_result=sia_out,
            si_input=si_input,
            ia_input=ia_input,
        )
    except Exception as e:
        print(f"Display failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
