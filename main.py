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
    from utils.display_results import show_sia_table, show_eia_table
except Exception:  # pragma: no cover
    from display_results import show_sia_table, show_eia_table


HEADER = r"""
===============================================================
 Optimizing investment portfolios in eastern Europe
 for economic and social impacts

 Authors: Abhiman Arjun, Emil Levin​ Blauschke,​ Scott​ Clements, Alpha Saliou Diallo,
 Till Lennart​ ​Geisel, Shady Soumaya, Loran Toga, Din Zekic​
===============================================================
To start, enter the total budget to invest

NOTE (temporary patch):
- We run Social Impact Analysis (SIA) with the FULL budget.
- We run Economic Impact Analysis (EIA) with the FULL budget.
- Results are shown in TWO SEPARATE TABLES without cross-effects.
"""


def parse_budget(s: str) -> float:
    s = s.strip().replace(",", ".")
    if not re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", s):
        raise ValueError("Budget must be a number (e.g., 25, 25.0, 12,5).")
    return float(s)


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

    # -------- Run SIA (full budget) --------
    try:
        sia_res = run_sia_optimization(si_input, total_budget=total_budget)
        sia_out = process_sia_result(sia_res, si_input, total_budget=total_budget)  # (2, I) [x; weighted score]
        _validate_portfolio_sum(sia_out[0, :], total_budget, "SIA")
    except Exception as e:
        print(f"SIA optimization failed: {e}")
        sys.exit(1)

    # -------- Run EIA (full budget) --------
    try:
        ia_out = run_ia_optimization(ia_input, total_budget=total_budget)  # (2, I) [x; econ impact in M€]
        _validate_portfolio_sum(ia_out[0, :], total_budget, "EIA")
    except Exception as e:
        print(f"IA optimization failed: {e}")
        sys.exit(1)

    # -------- Display (two separate tables) --------
    print("\n===============================================================")
    print(f" Using total budget: {total_budget:.3f} M€ for BOTH analyses (temporary patch)")
    print("===============================================================\n")

    try:
        show_sia_table(social_result=sia_out, si_input=si_input)
        print("\n")  # Spacer between tables
        show_eia_table(economic_result=ia_out)
    except Exception as e:
        print(f"Display failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
