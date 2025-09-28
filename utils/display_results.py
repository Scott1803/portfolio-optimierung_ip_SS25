from __future__ import annotations
import numpy as np

# Robust imports to work in both package and local layouts
try:
    # Returns [ row0 = adjusted x, row1 = φ (0..1) ]
    from utils.sia_new import calculate_impact_for_chosen_portfolio as sia_calc_portfolio
except Exception:  # pragma: no cover
    from sia_new import calculate_impact_for_chosen_portfolio as sia_calc_portfolio

try:
    # New EIA API: accepts full-length x, returns (2, I): [x, economic impact in M€]
    from utils.eia import calculate_economic_impact as eia_calc_impact
except Exception:  # pragma: no cover
    from eia import calculate_economic_impact as eia_calc_impact


def _print_line(char: str = "=", width: int = 100) -> None:
    print(char * width)


def _fmt_bool_mark(cond: bool) -> str:
    return "✅" if cond else "❌"


def show_sia_table(
    *,
    social_result: np.ndarray,
    si_input,
) -> None:
    """
    Print a table for the Social Impact Analysis only.

    Columns:
      Project | Funded | Budget (M€) | Social Impact (%)

    Notes
    -----
    - The input `social_result` is a (2, I) array where:
        row0 = chosen investments x (M€)
        row1 = weighted score (ignored here)
    - We recompute φ(x) with the SIA model to get comparable % values:
        φ%(x) = 100 * φ(x)
    """
    if social_result.ndim != 2 or social_result.shape[0] != 2:
        raise ValueError("social_result must be shaped (2, I)")

    x_sia = np.asarray(social_result[0, :], dtype=float)
    I = x_sia.size

    # Recompute φ% using the SIA model (ignore weighted row in social_result)
    # We avoid strict budget enforcement here since the optimizer already respected it.
    sia_eval = sia_calc_portfolio(x_sia, si_input, total_budget=None, enforce_budget=False)
    phi_percent = 100.0 * np.asarray(sia_eval[1, :], dtype=float)

    _print_line("=")
    print("SOCIAL IMPACT ANALYSIS")
    _print_line("=")
    print(f"{'Project':<10} {'Funded':<8} {'Budget (M€)':<14} {'Social Impact (%)':<20}")
    _print_line("-")

    total_budget = 0.0
    total_social = 0.0
    eps = 1e-9

    for i in range(I):
        funded = x_sia[i] > eps
        pid = f"P{i+1}"
        funded_mark = _fmt_bool_mark(funded)
        budget_str = f"{x_sia[i]:<14.3f}" if funded else f"{0:<14.3f}"
        social_str = f"{phi_percent[i]:<20.2f}" if funded else f"{'-':<20}"
        print(f"{pid:<10} {funded_mark:<8} {budget_str} {social_str}")

        if funded:
            total_budget += x_sia[i]
            total_social += phi_percent[i]

    _print_line("=")
    print(
        f"{'SUM':<10} {'':<8} "
        f"{total_budget:<14.3f} "
        f"{total_social:<20.2f}"
    )
    _print_line("=")


def show_eia_table(
    *,
    economic_result: np.ndarray,
) -> None:
    """
    Print a table for the Economic Impact Analysis only.

    Columns:
      Project | Funded | Budget (M€) | Economic Impact (M€)

    Notes
    -----
    - The input `economic_result` is a (2, I) array where:
        row0 = chosen investments x (M€)
        row1 = economic impact per project in M€
    """
    if economic_result.ndim != 2 or economic_result.shape[0] != 2:
        raise ValueError("economic_result must be shaped (2, I)")

    x_ia = np.asarray(economic_result[0, :], dtype=float)
    econ = np.asarray(economic_result[1, :], dtype=float)
    I = x_ia.size

    _print_line("=")
    print("ECONOMIC IMPACT ANALYSIS")
    _print_line("=")
    print(f"{'Project':<10} {'Funded':<8} {'Budget (M€)':<14} {'Economic Impact (M€)':<22}")
    _print_line("-")

    total_budget = 0.0
    total_economic = 0.0
    eps = 1e-9

    for i in range(I):
        funded = x_ia[i] > eps
        pid = f"P{i+1}"
        funded_mark = _fmt_bool_mark(funded)
        budget_str = f"{x_ia[i]:<14.3f}" if funded else f"{0:<14.3f}"
        econ_str = f"{econ[i]:<22.6f}" if funded else f"{'-':<22}"
        print(f"{pid:<10} {funded_mark:<8} {budget_str} {econ_str}")

        if funded:
            total_budget += x_ia[i]
            total_economic += econ[i]

    _print_line("=")
    print(
        f"{'SUM':<10} {'':<8} "
        f"{total_budget:<14.3f} "
        f"{total_economic:<22.6f}"
    )
    _print_line("=")
