import openpyxl as px
import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

# -------------
# Daten werden eingelesen (hier typen bestimmen und in andere Datei extrahieren)
# -------------

# Read entire "Aufgabenteilung" excel sheet
xls = pd.ExcelFile('Aufgabenteilung.xlsx')

#Tabellen einlesen
b_initParams = pd.read_excel(xls, 'SIA_Vorlage', usecols='H:L',nrows=13, decimal=',')
a_länderWeigths = pd.read_excel(xls, 'Länder', usecols='B:F', decimal=',')

A = a_länderWeigths.to_numpy().astype(float)
B = b_initParams.to_numpy().astype(float)
R = np.array([
    [ 0.2581,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.1720, -0.1720,  0.0000,  0.0000,  0.1720],
    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0398,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.2518,  0.0000,  0.0000,  0.1259,  0.1259,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.1667, -0.1667, -0.1111,  0.1111,  0.0000],
    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [-0.0500, -0.1500,  0.0000,  0.0000,  0.0500,  0.0000,  0.1500,  0.0000,  0.0000, -0.0500,  0.1000, -0.1000],
    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0792,  0.0000],
    [ 0.1166,  0.0000,  0.2333,  0.3500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  0.0776,  0.0776,  0.1552,  0.2329,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
    [-0.0360, -0.0360, -0.0360,  0.0000,  0.0730,  0.0360, -0.0360, -0.0360,  0.0000,  0.0000,  0.0000,  0.0000]
], dtype=float)

# b: (K,C)  baseline   | r: (K,I)  relative Wirkungsgrade
K, I = R.shape
K2, C = B.shape
print(K)
print(K2)
assert K == K2, "Dimensionen der Wirkungsgrade stimmen nicht überein"

# Minimale Budgets / Projekt
m = np.array([3.00,4.00,2.00,2.48,1.50,1.90,0.10,0.00,0.00,4.00,6.00,3.00], dtype=float)

# Maximale Budgets / Projekt
xmax = np.array([17.00,19.00,6.50,7.45,8.30,9.05,0.53,25.00,25.00,9.00,12.00,8.00], dtype=float)

# Gesamtbudget
B_total = 25.0  # Gesamtbudget (Mio €)


t = R[0,0] * (1.0-B[0,:])
print(t)
# Gegeben:
# R: (K,I)  -> Projekte × Indikatoren (deine harte Matrix)
# B: (K,C)  -> Länder × Indikatoren (Baseline). Falls du es als (K,C) eingelesen hast: B = B.T
# A_mask: (I,C) -> Projekt-Land-Maske (0/1 oder Anteile)  [optional]
# v: (C,)  -> Ländergewichte (optional)
# m, x_max: (I,) -> Budgets (nur für A_i unten)

B = B.T  # jetzt (C,K)
need = 1.0 - B
print(need)

#3D Tensor für alle Projekte, Länder, Indikatoren
N = np.zeros((I,C,K), dtype=float)
# Zusätzlich: Liste mit C×K-Matrizen pro Projekt (leichter zu inspizieren)
M_per_project = []

for i in range(I):
    # leere C×K-Matrix für Projekt i
    M_i = np.zeros((C, K), dtype=float)
    for k in range(K):
        # Skalar R[i,k] * Länder-Vektor need[:,k]  -> Länge C
        # R[i,k] ist der relative Wirkungsgrad von Projekt i auf Indikator k
        # need[c,k] ist die mögliche Verbesserung des Landes c für Indikator k
        # M_i[c,k] = R[i,k] * need[c,k] ist dann der maximale Verbesserungspotenzial bei maximalem Budget des Indikators k durch Projekt i im Land c
        M_i[:, k-1] = R[i, k-1] * need[:, k-1]

    # in den 3D-Tensor übernehmen
    N[i, :, :] = M_i
    M_per_project.append(M_i)

print(N)    
#N-i,c,k steht jetzt in einer 3D-Matrix also i MAtrizen mit c zeilen und k Spalten, wobei i die Anzahl der Projekte sind, c die Anzahl der Länder und k die Anzahl der Indikatoren



# -------------
# Generiert ein vektor (array) mit einem Projekt-Koeffizienten (totaler impact des projekts über alle länder und indikatoren, respektierend der Ländergewichte in A)
# -------------

#Funktion: SIA-i,c,k(x_i) = N-i,c,k * ln(1+min{x_max-m_i, max{0,x_i-m_i}})/ln(1+x_max-m_i)
#def build_W()
def project_coefficients_loopy(N, A):
    """
    N: (I,C,K)    -> N[i,c,k]
    A: (C,I) ODER (I,C)
    Rückgabe: proj_coef[i] = Σ_{c,k} A[i,c] bzw. A[c,i] * N[i,c,k]
    """
    I, C, K = N.shape
    proj_coef = np.zeros(I, dtype=float)

    if A.shape == (C, I):
        # A ist Länder × Projekte -> Spaltenvektor A[:, i] gehört zu Projekt i
        for i in range(I):
            s = 0.0
            for c in range(C):
                a_ci = A[c, i]
                for k in range(K):
                    s += a_ci * N[i, c, k]
            proj_coef[i] = s

    elif A.shape == (I, C):
        # A ist Projekte × Länder -> Zeilenvektor A[i, :] gehört zu Projekt i
        for i in range(I):
            s = 0.0
            for c in range(C):
                a_ic = A[i, c]
                for k in range(K):
                    s += a_ic * N[i, c, k]
            proj_coef[i] = s

    else:
        raise ValueError(f"Unerwartete A-Form: {A.shape} (erwartet (C,I) oder (I,C))")
    return proj_coef  # (I,)

def penalty_objectives(vars, N, A, m, xmax, B_total, rho_bud=1e5, rho_box=1e6,rho_exc1=1e5,rho_bin=1e5):
    I = len(m)
    x = vars[:I] #Budgetvariablen
    y = vars[I:] #Binärvariablen

    u = xmax - m
    den = np.log1p(u); den = np.where(den > 0, den, 1.0)

    # t_i = min{x_max - m_i, max{0, y_i*x_i - m_i}}
    t = np.minimum(u, np.maximum(0.0, y*x - m))

    #hier Ländergewichtung anwenden
    #proj_coef[i] = Σ_{c,k} A[i,c] * N[i,c,k]
    proj_coef = project_coefficients_loopy(N, A)  # (I,)
    # Nutzen = Σ proj_coef[i] * ln(1+t_i)/ln(1+u_i)
    fneg = -np.sum(proj_coef * np.log1p(t)/den)

    #penalty Terme
    pen = rho_bud * max(0.0, np.sum(y*x) - B_total)**2  # Budgetüberschreitung
    pen += rho_box * np.sum(np.maximum(0.0,m*y-x)**2 + np.maximum(0.0, x - xmax*y)**2)  # Box-Beschränkungen
    pen += rho_exc1 * max(0.0, y[2]+y[3] -1.0)**2  # Exklusivität 1: Projekte 3 und 4
    pen += rho_exc1 * max(0.0, y[4]+y[5] -1.0)**2  # Exklusivität 2: Projekte 5 und 6
    pen += rho_bin * np.sum((y*(1.0-y))**2)  # Binärstrafe
    
    return fneg + pen

def make_start(m, xmax, B_total):
    I = len(m)
    x0 = np.minimum(xmax, np.full(I, B_total/max(1.0,I)))
    y0 = np.full(I, 0.5)
    return np.concatenate([x0, y0])

def make_bounds(xmax):
    I = len(xmax)
    b_x = [(0.0, float(xmax[i])) for i in range(I)]
    b_y = [(0.0,1.0)]*I
    return b_x + b_y

vars0 = make_start(m, xmax, B_total)
bounds = make_bounds(xmax)

res = minimize(
    penalty_objectives, vars0,
    args=(N, A, m, xmax, B_total),
    bounds=bounds,
    method='L-BFGS-B',
    options=dict(maxiter=800, ftol=1e-9, disp=True)
)
print("Status:", res.message)
print("Objektwert:", -res.fun)
print("x_opt:", np.round(res.x[:len(m)],3))
print("y_opt:", np.round(res.x[len(m):],3))

def postprocess(res, m, xmax, B_total):
    I = len(m)
    x = res.x[:I].copy()
    y = res.x[I:].copy()

    # y hart runden auf {0,1}
    y = (y >= 0.5).astype(float)

    # Exklusivität erzwingen (z. B. P3/P3+ = Indizes 2,3; P4/P4+ = 4,5)
    if y[2] + y[3] > 1:
        y[3 if x[3] < x[2] else 2] = 0.0
    if y[4] + y[5] > 1:
        y[5 if x[5] < x[4] else 4] = 0.0

    # Kopplung: m_i*y_i ≤ x_i ≤ xmax_i*y_i
    x = np.minimum(x, xmax * y)
    x = np.maximum(x, m * y)

    # Budget prüfen und ggf. anpassen
    spent = float(np.sum(y * x))
    if spent > B_total + 1e-9:
        active = y > 0.5
        room = x[active] - m[active]
        if room.sum() > 0:
            over = spent - B_total
            x[active] -= over * (room / room.sum())
            x[active] = np.maximum(x[active], m[active])

    return x, y

def compute_project_impacts(N, A, x, y, m, xmax):
    I = len(m)
    u   = xmax - m
    den = np.log1p(u); den = np.where(den > 0, den, 1.0)
    s   = np.maximum(0.0, y*x - m)
    t   = np.minimum(u, s)
    phi = np.log1p(t) / den                       # (I,)

    # Gesamtgewichtung pro Projekt (Σ_{c,k} A[i,c]*N_raw[i,c,k])
    proj_coef = (A[:, :, None] * N).sum(axis=(1,2))  # (I,)

    impacts = proj_coef * phi
    return impacts

x_opt, y_opt = postprocess(res, m, xmax, B_total)

impacts = compute_project_impacts(N, A, x_opt, y_opt, m, xmax)
print("Projekt-Wirkungsgrade:")
for i, val in enumerate(impacts, start=1):
    print(f"P{i}: {val:.4f}")

# -------------
# MARK: Ausgabe Ergebnisse (irrelevant)
# -------------

#Ab hier wird die Ergebnistabelle erstellt, mithilfe von ChatGPT. Habe ich so übernommen
def build_result_table_ordered(x, y, impacts, m, xmax, B_total,
                               project_names=None, sort_by=None, ascending=False):
    """
    Ergebnis-Tabelle in *vorgegebener Reihenfolge*:
      ✓ (gewählt) / ✖ (nicht gewählt)
      Budget x, Mindest m, Max x_max, Zusatz s, Auslastung s/u, Budgetanteil, Impact
    sort_by=None  => keine Sortierung (Reihenfolge bleibt erhalten).
    """
    x       = np.asarray(x, dtype=float)
    y       = np.asarray(y, dtype=float)
    m       = np.asarray(m, dtype=float)
    xmax    = np.asarray(xmax, dtype=float)
    impacts = np.asarray(impacts, dtype=float)

    I = len(m)
    if project_names is None:
        project_names = [f"P{i+1}" for i in range(I)]

    u    = xmax - m
    s    = np.maximum(0.0, y * x - m)
    util = np.divide(s, u, out=np.zeros_like(s), where=u > 0)

    budget_eff = y * x
    share = np.divide(budget_eff, B_total, out=np.zeros_like(budget_eff), where=B_total > 0)

    mark = np.where(y >= 0.5, "✓", "✖")

    df = pd.DataFrame({
        "Projekt": project_names,
        "Markierung": mark,
        "Aktiv (y)": y.astype(int),
        "Budget x [Mio]": np.round(x, 3),
        "Mindest m": np.round(m, 3),
        "Max x_max": np.round(xmax, 3),
        "Zusatz s = x - m*y": np.round(s, 3),
        "Auslastung s/u": np.round(util, 3),
        "Budgetanteil": np.round(share, 3),
        "Impact": np.round(impacts, 4),
    })

    # Nur sortieren, wenn explizit gewünscht
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, kind="mergesort").reset_index(drop=True)
        # mergesort = stabile Sortierung, bewahrt Reihenfolge bei Gleichständen

    total = pd.DataFrame({
        "Projekt": ["Σ"],
        "Markierung": [""],
        "Aktiv (y)": [int(y.sum())],
        "Budget x [Mio]": [np.round(budget_eff.sum(), 3)],
        "Mindest m": [np.round((m * y).sum(), 3)],
        "Max x_max": [np.round((xmax * y).sum(), 3)],
        "Zusatz s = x - m*y": [np.round(s.sum(), 3)],
        "Auslastung s/u": [np.nan],
        "Budgetanteil": [np.round(share.sum(), 3)],
        "Impact": [np.round(impacts.sum(), 4)],
    })

    return pd.concat([df, total], ignore_index=True)

project_names = ["P1","P2","P3","P3+","P4","P4+","P5","P6","P7","P8","P9","P10"]

tbl = build_result_table_ordered(
    x_opt, y_opt, impacts, m, xmax, B_total,
    project_names=project_names,
    sort_by=None  # <- keine Sortierung, Reihenfolge bleibt
)

print(tbl.to_string(index=False)) 


#ab hier die neu SIA TAbelle

def compute_improvements_ck(N, A, x, y, m, xmax):
    """
    Δ(c,k) = Σ_i A[i,c] * N[i,c,k] * φ_i
    φ_i = ln(1 + t_i) / ln(1 + u_i)
    t_i = min{u_i, max{0, y_i*x_i - m_i}},  u_i = xmax_i - m_i

    Erwartet:
      N: (I,C,K)   (dein 3D-Tensor, OHNE A eingebacken)
      A: (I,C)     Länderanteile (Zeilensumme≈1)
      x,y,m,xmax: (I,)
    """
    u   = xmax - m
    den = np.log1p(u); den = np.where(den > 0, den, 1.0)
    s   = np.maximum(0.0, y * x - m)
    t   = np.minimum(u, s)
    phi = np.log1p(t) / den                      # (I,)

    # A erst hier anwenden:
    N_tilde = A[:, :, None] * N                  # (I,C,K)
    Delta_ck = (N_tilde * phi[:, None, None]).sum(axis=0)  # (C,K)
    return Delta_ck

def build_optimized_tables_from_Bdf(B_df, N, A, x_opt, y_opt, m, xmax):
    """
    Nimmt deine ursprüngliche Baseline-DF (wie 'SIA_Vorlage' eingelesen, aber in Form (C,K)),
    berechnet Δ und liefert:
      - improvements_df (Δ)
      - new_scores_df  (B + Δ, auf [0,1] geklippt)
    mit den GLEICHEN Zeilen-/Spaltenlabels wie B_df.
    """
    # Falls du B als NumPy (C,K) hast, mach vorher: B_df = pd.DataFrame(B, index=..., columns=...)
    Delta_ck = compute_improvements_ck(N, A, x_opt, y_opt, m, xmax)  # (C,K)

    # DataFrames mit denselben Labels wie Baseline
    improvements_df = pd.DataFrame(Delta_ck, index=B_df.index, columns=B_df.columns)
    new_scores_df   = pd.DataFrame(np.clip(B_df.values + Delta_ck, 0.0, 1.0),
                                   index=B_df.index, columns=B_df.columns)
    return improvements_df, new_scores_df

# b_initParams: so wie du es gelesen hast (12 Zeilen, 5 Spalten; je nach Vorlage)
# Wir brauchen B_df als (C,K) = (Länder × Indikatoren).
B_df_raw = b_initParams.copy()     # unverändert aus Excel
B_df = B_df_raw.T                  # jetzt (C,K)

# Falls in Excel keine klaren Kopf-/Zeilenlabels stehen, kannst du Fallback-Labels setzen:
if B_df.index.dtype == 'int64':   # kein Ländername vorhanden
    B_df.index = [f"Land{c+1}" for c in range(B_df.shape[0])]
if B_df.columns.dtype == 'int64': # kein Indikatorname vorhanden
    B_df.columns = [f"Ind{k+1}" for k in range(B_df.shape[1])]

imp_df, new_df = build_optimized_tables_from_Bdf(B_df, N, A, x_opt, y_opt, m, xmax)

print("Δ(c,k) – absolute Verbesserung (0–1 Skala):")
print(imp_df.to_string())
print("\nOptimierte SIA-Matrix (B + Δ, auf [0,1] geklippt):")
print(new_df.to_string())


def generate_project_country_indicator_tensor() -> np.ndarray:
    """
    Parses the read input data into a 3D Tensor,
    representing projects on it's x, indicators on it's y and indicators on it's z axis.
    """
    pass;

def generate_project_value_added_coefficients(pci_tensor: np.ndarray, country_weights: np.ndarray) -> np.ndarray:
    """
    Processes the project / country / indicator tensor and country matrix into a vector
    (array) with the length of the project amount, that contains a value added coefficient per million per project
    """
    pass;

def penalty_objective_function() -> float:
    """
    Is used by the optimizer to find the least possible penalty for
    an investment portfolio, by calling over and over until smallest return value is found.
    """
    RHO_BUDGET=1e5
    RHO_BOX=1e6
    RHO_EXC1=1e5
    RHO_BIN=1e5

    pass;

def run_optimization() -> OptimizeResult:
    """
    Uses the provided indicator 
    """
    pass;

def process_optimization_result(optimizer_result: OptimizeResult) -> np.ndarray:
    """
    processes the raw result returned by the optimization function into a 2D matrix that contains two rows:
    1. Element i of the first row contains the budget invested into project i
    2. Element i of the second row is the social impact (in percent of the maximum achievable impact) of the investment into project i

    This is the normed final output for SIA, that can be displayed to the user
    """
    pass;