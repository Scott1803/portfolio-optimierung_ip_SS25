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
    """
    Accepts integer/float with comma or dot decimal separator.
    Raises ValueError on invalid.
    """
    s = s.strip().replace(",", ".")
    if not re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", s):
        raise ValueError("Budget must be a number (e.g., 25, 25.0, 12,5).")
    return float(s)


def parse_split(s: str) -> tuple[int, int]:
    """
    Expects 'A/B' where A, B are whole numbers that sum to 100.
    Returns (sia_pct, ia_pct).
    """
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
    """
    Pull dependency matrix from IA input and attach to SIA input so the SIA code can consume it.
    Validates shape and (optionally) that it has at least one non-zero dependency.
    """
    dep_mat = getattr(ia_input, "ia_projekt_abhaengigkeiten", None)
    if dep_mat is None:
        # Try alternative name used in some sheets
        dep_mat = getattr(ia_input, "sia_projekt_abhaengigkeiten", None)

    if dep_mat is None:
        # Nothing to attach — SIA will behave without deps
        return None

    M = np.asarray(dep_mat, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            f"Dependency matrix must be square; got shape {M.shape}. "
            "Expected I×I with rows = prerequisites, cols = dependents."
        )

    # Zero diagonal and binarize, just to be safe
    M = (M >= 0.5).astype(float)
    np.fill_diagonal(M, 0.0)

    # Attach onto the SIA input so run_optimization() and process_optimization_result() can pick it up
    setattr(si_input, "ia_projekt_abhaengigkeiten", M)
    return M


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
        dep_mat = _attach_dependency_matrix(si_input, ia_input)
    except Exception as e:
        print(f"Dependency matrix error: {e}")
        sys.exit(1)

    # -------- Run SIA --------
    try:
        sia_res = run_sia_optimization(si_input, total_budget=sia_budget)
        # Pass matrix to post-processor too (process_optimization_result supports it)
        if dep_mat is not None:
            sia_out = process_sia_result(
                sia_res, si_input, total_budget=sia_budget, dependency_matrix=dep_mat
            )
        else:
            sia_out = process_sia_result(sia_res, si_input, total_budget=sia_budget)
        # Convert to % **only if** your display expects normalized share; if your SIA
        # returns weighted impact already, comment the next line.
        # sia_out[1, :] *= 100.0
    except Exception as e:
        print(f"SIA optimization failed: {e}")
        sys.exit(1)

    # -------- Run IA --------
    try:
        ia_out = run_ia_optimization(ia_input, total_budget=ia_budget)  # (2, I)
    except Exception as e:
        print(f"IA optimization failed: {e}")
        sys.exit(1)

    # -------- Display --------
    print("\n===============================================================")
    print(f" Using total budget: {total_budget:.3f} M€ "
          f"(SIA {sia_pct}% → {sia_budget:.3f} M€, IA {ia_pct}% → {ia_budget:.3f} M€)")
    print("===============================================================\n")

    try:
        show_table_for_results(economic_result=ia_out, social_result=sia_out)
    except Exception as e:
        print(f"Display failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
