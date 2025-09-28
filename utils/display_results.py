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


def _call_eia_calc(x: np.ndarray, ia_input) -> np.ndarray:
    """
    Call EIA 'calculate_economic_impact' and return the (2, I) result.
    Supports either signature: (x) or (x, ia_input).
    """
    return eia_calc_impact(x, ia_input)  # type: ignore[arg-type]


def show_table_for_results(
    economic_result: np.ndarray,
    social_result: np.ndarray,
    *,
    si_input,
    ia_input,
) -> None:
    """
    Joint table with *consistent units*:

    - Budget shown = x_IA + x_SIA per project
    - Economic Impact (M€) = IA_economic_impact + EIA(x_SIA)_economic_impact
    - Social Impact (%)    = 100*(φ(x_SIA) + φ(x_IA))  [both φ come from SIA model]

    Notes
    -----
    - We do NOT use social_result[1] because that’s a weighted score (proj_coef*φ).
      We recompute φ (%) for both portfolios so units match.
    """
    if economic_result.shape != social_result.shape:
        raise ValueError("Economic and social results must have the same shape")
    if economic_result.ndim != 2 or economic_result.shape[0] != 2:
        raise ValueError("Results must be shaped (2, I)")

    I = economic_result.shape[1]

    # Extract investments and IA's own economic impact
    x_ia = np.asarray(economic_result[0, :], dtype=float)
    econ_base = np.asarray(economic_result[1, :], dtype=float)  # M€

    # Extract SIA investments (ignore social_result[1] because it’s weighted, not %)
    x_sia = np.asarray(social_result[0, :], dtype=float)

    # --- Cross impacts in consistent units ---

    # Social from BOTH portfolios, using SIA φ (0..1) → %:
    # φ_SIA_from_SIA:
    sia_from_sia = sia_calc_portfolio(x_sia, si_input, total_budget=None, enforce_budget=False)
    soc_from_sia_percent = 100.0 * np.asarray(sia_from_sia[1, :], dtype=float)

    # φ_SIA_from_IA:
    sia_from_ia = sia_calc_portfolio(x_ia, si_input, total_budget=None, enforce_budget=False)
    soc_from_ia_percent = 100.0 * np.asarray(sia_from_ia[1, :], dtype=float)

    # Economic from SIA portfolio, using EIA:
    eia_on_sia = _call_eia_calc(x_sia, ia_input)
    econ_from_sia = np.asarray(eia_on_sia[1, :], dtype=float)  # M€

    # Combined per-project impacts
    econ_total = econ_base + econ_from_sia                    # M€
    soc_total_percent = soc_from_sia_percent + soc_from_ia_percent  # %

    # Combined budgets
    budgets = x_ia + x_sia

    # ---- Printing ----
    print("=" * 100)
    print("JOINT OPTIMIZATION RESULT")
    print("=" * 100)
    print(f"{'Project':<10} {'Funded':<8} {'Budget (M€)':<14} {'Economic Impact':<20} {'Social Impact (%)':<20}")
    print("-" * 100)

    total_budget = 0.0
    total_economic = 0.0
    total_social = 0.0

    for i in range(I):
        pid = f"P{i+1}"
        funded = budgets[i] > 1e-9
        funded_mark = "✅" if funded else "❌"
        budget_str = f"{budgets[i]:<14.3f}" if funded else f"{0:<14.3f}"
        econ_str = f"{econ_total[i]:<20.6f}" if funded else f"{'-':<20}"
        soc_str = f"{soc_total_percent[i]:<20.2f}" if funded else f"{'-':<20}"

        print(f"{pid:<10} {funded_mark:<8} {budget_str} {econ_str} {soc_str}")

        if funded:
            total_budget += budgets[i]
            total_economic += econ_total[i]
            total_social += soc_total_percent[i]

    print("=" * 100)
    print(
        f"{'SUM':<10} {'':<8} "
        f"{total_budget:<14.3f} "
        f"{total_economic:<20.6f} "
        f"{total_social:<20.2f}"
    )
    print("=" * 100)
