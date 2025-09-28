import numpy as np

def show_table_for_results(economic_result: np.ndarray, social_result: np.ndarray):
    """
    Fetches the mappings from the input data sheet, then uses it and the analysis results to generate the result
    table in the commandline, with the following columns:
    1. Projekt ID
    2. Projekt Name
    3. Funded (checkmark / cross)
    4. Budget (amount of money invested into project)
    5. Economic impact (in added millions)
    6. Social impact (in percent of maximum achievable impact)

    In addition, a "sum" row is displayed under all project columns, that displays the total invested amount, total economic and total social impact.

    @important:
    If `economic_result` and `social_result` both contain an investment in a project, their investments are added together and displayed as one
    in the table.
    """
    if economic_result.shape != social_result.shape:
        raise ValueError("Economic and social results must have the same shape")

    I = economic_result.shape[1]

    # Merge budgets: sum of investments from both
    budgets = economic_result[0, :] + social_result[0, :]
    econ_impact = economic_result[1, :]
    soc_impact = social_result[1, :]

    # Header
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
        econ_str = f"{econ_impact[i]:<20.6f}" if funded else f"{'-':<20}"
        soc_str = f"{soc_impact[i]:<20.2f}" if funded else f"{'-':<20}"

        print(f"{pid:<10} {funded_mark:<8} {budget_str} {econ_str} {soc_str}")

        if funded:
            total_budget += budgets[i]
            total_economic += econ_impact[i]
            total_social += soc_impact[i]

    # Summary row
    print("=" * 100)
    print(
        f"{'SUM':<10} {'':<8} "
        f"{total_budget:<14.3f} "
        f"{total_economic:<20.6f} "
        f"{total_social:<20.2f}"
    )
    print("=" * 100)
