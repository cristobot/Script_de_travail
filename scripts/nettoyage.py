"""
nettoyage.py : nettoyage du dataset Kidney Disease (Thème 3, diagnostic de la MRC).

Ce script reproduit, de bout en bout, le nettoyage réalisé dans `cleaning_data.ipynb`.
Il produit trois fichiers dans le dossier de sortie :
    - kidney_clean.csv          : données corrigées, SANS aucune valeur inventée (NaN conservés) ;
    - kidney_clean_imputed.csv  : mêmes lignes, valeurs manquantes imputées ;
    - journal_nettoyage.csv     : journal de toutes les actions (quoi, combien, pourquoi).

Principe : chaque étape est suivie de contrôles (`controle`). Si un résultat diffère de la
valeur attendue, le script s'arrête avec une erreur explicite. Les valeurs attendues sont
celles observées sur le fichier d'origine (400 lignes x 26 colonnes).

Dépendances : pandas, numpy, scipy, scikit-learn.

Utilisation :
    python scripts/nettoyage.py --excel "data/Base Kidney diseases.xlsx" --sortie data
"""

from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (active IterativeImputer)
from sklearn.impute import IterativeImputer


# CONFIGURATION
FEUILLE = "kidney_disease"
NB_LIGNES, NB_COLONNES = 400, 26

COLS_NUM = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
COLS_CAT = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "classification"]

# Valeurs observées lors de l'inspection du fichier d'origine (servent de contrôle).
DATES_ATTENDUES = {"sg": 190, "sc": 236, "pot": 264, "hemo": 295, "rc": 245, "bu": 1, "sod": 1}
INTERROGATIONS_ATTENDUES = {"pcv": 1, "wc": 1, "rc": 1}
NAN_NUM_ATTENDUS = {"age": 9, "bp": 12, "sg": 47, "al": 46, "su": 49, "bgr": 44, "bu": 19,
                    "sc": 17, "sod": 87, "pot": 88, "hemo": 52, "pcv": 71, "wc": 106, "rc": 131}

MODALITES_ATTENDUES = {
    "rbc": {"normal": 201, "abnormal": 47}, "pc": {"normal": 259, "abnormal": 76},
    "pcc": {"notpresent": 354, "present": 42}, "ba": {"notpresent": 374, "present": 22},
    "htn": {"no": 251, "yes": 147}, "dm": {"no": 261, "yes": 137}, "cad": {"no": 364, "yes": 34},
    "appet": {"good": 317, "poor": 82}, "pe": {"no": 323, "yes": 76}, "ane": {"no": 339, "yes": 60},
    "classification": {"ckd": 250, "notckd": 150},
}

# Valeurs physiologiquement impossibles : (id du patient, variable). Mises à NaN.
VALEURS_IMPOSSIBLES = [(21, "sod"), (61, "pot"), (128, "pot")]

# Bornes volontairement larges : elles ne servent qu'à détecter l'impossible, pas l'extrême.
BORNES = {
    "age": (0, 120), "bp": (30, 250), "sg": (1.000, 1.050), "al": (0, 5), "su": (0, 5),
    "bgr": (20, 800), "bu": (1, 500), "sc": (0.1, 100), "sod": (90, 180), "pot": (1.5, 9),
    "hemo": (3, 20), "pcv": (5, 60), "wc": (1000, 50000), "rc": (1.5, 9),
}

CAT_UNKNOWN = ["rbc", "pc"]                                              # manquants fréquents -> 'unknown'
CAT_MODE = ["pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]       # manquants rares -> mode

MODALITES_IMPUTEES_ATTENDUES = {
    "rbc": {"normal": 201, "unknown": 152, "abnormal": 47}, "pc": {"normal": 259, "abnormal": 76, "unknown": 65},
    "pcc": {"notpresent": 358, "present": 42}, "ba": {"notpresent": 378, "present": 22},
    "htn": {"no": 253, "yes": 147}, "dm": {"no": 263, "yes": 137}, "cad": {"no": 366, "yes": 34},
    "appet": {"good": 318, "poor": 82}, "pe": {"no": 324, "yes": 76}, "ane": {"no": 340, "yes": 60},
}

JOURNAL: list[dict] = []   # journal consolidé de toutes les actions


# OUTILS
def est_date(valeur) -> bool:
    """Vrai si la cellule contient une date (Excel a converti un décimal en date)."""
    return isinstance(valeur, (dt.datetime, pd.Timestamp))


def controle(nom: str, obtenu, attendu) -> None:
    """Compare un résultat à la valeur attendue ; arrête le script en cas d'écart."""
    if obtenu != attendu:
        raise AssertionError(f"[ÉCHEC] {nom} : obtenu {obtenu!r}, attendu {attendu!r}")
    print(f"[OK] {nom} : {obtenu}")


def journaliser(etape: str, action: str, variables: str, cellules: int, justification: str) -> None:
    """Ajoute une ligne au journal de nettoyage."""
    JOURNAL.append({"etape": etape, "action": action, "variables": variables,
                    "cellules": cellules, "justification": justification})


def numeriques(serie: pd.Series) -> pd.Series:
    """Conversion numérique de contrôle (les valeurs non convertibles deviennent NaN)."""
    return pd.to_numeric(serie, errors="coerce")


# ÉTAPE 1 : charger le fichier brut et en vérifier l'état initial
def charger_brut(chemin_excel: Path) -> pd.DataFrame:
    """Lit le fichier Excel sans aucune conversion (dtype=object).

    Pourquoi : on garde une copie brute intacte pour pouvoir prouver, à chaque étape,
    ce qui a été modifié. Forcer `object` empêche pandas de convertir quoi que ce soit.
    """
    return pd.read_excel(chemin_excel, sheet_name=FEUILLE, dtype=object)


def auditer_brut(df_raw: pd.DataFrame) -> None:
    """Vérifie que le fichier correspond à ce qui a été observé lors de l'inspection."""
    controle("dimensions", df_raw.shape, (NB_LIGNES, NB_COLONNES))
    controle("identifiants uniques", df_raw["id"].is_unique, True)
    dates = df_raw.apply(lambda s: int(s.map(est_date).sum()))
    controle("dates par colonne", dates[dates > 0].to_dict(), DATES_ATTENDUES)
    interrogations = df_raw.apply(lambda s: int((s == "?").sum()))
    controle("marqueurs '?' par colonne", interrogations[interrogations > 0].to_dict(), INTERROGATIONS_ATTENDUES)


# ÉTAPE 2 : décoder les décimaux convertis en dates par Excel
def decoder_sg(valeur):
    """Date -> gravité spécifique : 1 + mois / 100 (ex. février -> 1,02)."""
    return round(1 + valeur.month / 100, 3) if est_date(valeur) else valeur


def decoder_jour_mois(valeur):
    """Date -> valeur décimale : jour + mois / 10 (ex. 15 avril -> 15,4)."""
    return round(valeur.day + valeur.month / 10, 1) if est_date(valeur) else valeur


REGLES_DECODAGE = {
    "sg": decoder_sg,
    "sc": decoder_jour_mois, "pot": decoder_jour_mois, "hemo": decoder_jour_mois,
    "rc": decoder_jour_mois, "bu": decoder_jour_mois, "sod": decoder_jour_mois,
}


def decoder_dates(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Applique la règle de décodage, colonne par colonne, sur une copie de df_raw.

    Pourquoi : Excel a lu des décimaux comme des dates (partie entière -> jour, partie
    décimale -> mois). La conversion est réversible : on récupère ainsi 1 232 mesures
    valides au lieu de les perdre.
    """
    df = df_raw.copy()
    for colonne, regle in REGLES_DECODAGE.items():
        df[colonne] = df[colonne].map(regle)
    return df


def verifier_decodage(df_raw: pd.DataFrame, df: pd.DataFrame) -> None:
    """Contrôle chaque colonne décodée : plus de dates, aucune perte, valeurs plausibles."""
    controle("dates restantes", int(df.apply(lambda s: s.map(est_date).sum()).sum()), 0)

    # Aucune perte silencieuse (sauf rc, où le '?' devient NaN dans cette conversion de contrôle).
    for colonne in REGLES_DECODAGE:
        avant = int(df_raw[colonne].isna().sum())
        apres = int(numeriques(df[colonne]).isna().sum())
        controle(f"NaN {colonne} (avant -> après conversion)", apres,
                 avant + INTERROGATIONS_ATTENDUES.get(colonne, 0))

    # Plausibilité des valeurs obtenues
    controle("sg : valeurs possibles", [float(x) for x in sorted(numeriques(df["sg"]).dropna().unique())],
             [1.005, 1.01, 1.015, 1.02, 1.025])
    hemo = numeriques(df["hemo"]).dropna()
    controle("hemo : min et max", (float(hemo.min()), float(hemo.max())), (3.1, 17.8))
    controle("sc : 5 plus grandes", sorted(numeriques(df["sc"]).dropna())[-5:], [18.1, 24.0, 32.0, 48.1, 76.0])
    pot = numeriques(df["pot"]).dropna()
    controle("pot : 5 plus petites", sorted(pot)[:5], [2.5, 2.5, 2.7, 2.8, 2.9])
    controle("pot : 5 plus grandes", sorted(pot)[-5:], [6.5, 6.6, 7.6, 39.0, 47.0])
    rc = numeriques(df["rc"]).dropna()
    controle("rc : 5 plus petites", sorted(rc)[:5], [2.1, 2.1, 2.3, 2.4, 2.5])
    controle("rc : 5 plus grandes", sorted(rc)[-5:], [6.5, 6.5, 6.5, 6.5, 8.0])

    # Les deux cellules uniques sont tracées par identifiant.
    bu = numeriques(df["bu"])
    controle("bu décodée (id 173)", float(bu[df["id"] == 173].iloc[0]), 1.5)
    controle("bu : 5 plus petites", sorted(bu.dropna())[:5], [1.5, 10.0, 10.0, 15.0, 15.0])
    sod = numeriques(df["sod"])
    controle("sod décodée (id 21)", float(sod[df["id"] == 21].iloc[0]), 4.5)
    controle("sod : 5 plus petites", sorted(sod.dropna())[:5], [4.5, 104.0, 111.0, 113.0, 113.0])

    for colonne, n in DATES_ATTENDUES.items():
        journaliser("2", "dates décodées en nombres", colonne, n,
                    "Excel a converti des décimaux en dates ; conversion réversible, valeurs validées par plausibilité")


# ÉTAPE 3 : remplacer le marqueur '?' par une vraie valeur manquante
def remplacer_interrogations(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace '?' par NaN dans pcv, wc, rc.

    Pourquoi : '?' est un code de valeur manquante, pas une mesure. Le laisser bloquerait
    la conversion numérique ; le laisser à pd.to_numeric(errors='coerce') le transformerait
    en NaN sans trace. Une décision de nettoyage doit être explicite.
    """
    df = df.copy()
    colonnes = list(INTERROGATIONS_ATTENDUES)
    ids = {c: df.loc[df[c] == "?", "id"].tolist() for c in colonnes}
    controle("ids portant un '?'", ids, {"pcv": [66], "wc": [185], "rc": [162]})
    df[colonnes] = df[colonnes].replace("?", np.nan)
    controle("'?' restants", int((df == "?").sum().sum()), 0)
    controle("NaN pcv, wc, rc", df[colonnes].isna().sum().to_dict(), {"pcv": 71, "wc": 106, "rc": 131})
    journaliser("3", "'?' remplacé par NaN", "pcv, wc, rc", 3, "'?' est un code de valeur manquante")
    return df


# ÉTAPE 4 : convertir les variables quantitatives en nombres
def convertir_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les 14 variables quantitatives en float64.

    On utilise errors='raise' (et non 'coerce') : toute valeur inattendue arrête le script
    au lieu de devenir un NaN en silence.
    """
    df = df.copy()
    avant = df[COLS_NUM].isna().sum()
    for colonne in COLS_NUM:
        df[colonne] = pd.to_numeric(df[colonne], errors="raise").astype("float64")
    controle("écart total de NaN après conversion", int((df[COLS_NUM].isna().sum() - avant).abs().sum()), 0)
    controle("types numériques", int((df[COLS_NUM].dtypes == "float64").sum()), 14)
    controle("NaN par variable numérique", df[COLS_NUM].isna().sum().to_dict(), NAN_NUM_ATTENDUS)
    journaliser("4", "conversion en float64", "14 variables numériques", 14,
                "mesures stockées en texte ; errors='raise' pour interdire toute perte silencieuse")
    return df


# ÉTAPE 5 : contrôler les variables catégorielles (aucune modification)
def controler_categories(df: pd.DataFrame) -> None:
    """Vérifie l'absence d'espaces, de variantes de casse et de modalités inattendues."""
    normalise = df[COLS_CAT].apply(lambda s: s.str.strip().str.lower())
    ecarts = int((normalise.fillna("NA") != df[COLS_CAT].fillna("NA")).sum().sum())
    controle("cellules modifiées par strip/lower", ecarts, 0)
    for colonne, attendues in MODALITES_ATTENDUES.items():
        controle(f"modalités {colonne}", df[colonne].value_counts().to_dict(), attendues)
    journaliser("5", "contrôle sans correction", "11 variables catégorielles", 0,
                "aucune modalité parasite : aucune correction nécessaire")


# ÉTAPE 6 : mettre à NaN les valeurs physiologiquement impossibles
def traiter_valeurs_impossibles(df: pd.DataFrame) -> pd.DataFrame:
    """Met à NaN trois valeurs incompatibles avec la vie (sod = 4,5 ; pot = 39 et 47).

    Critère : valeur impossible chez un patient vivant, pas simplement extrême. Supprimer
    la ligne perdrait des mesures valides ; « corriger » (39 -> 3,9 ?) serait une
    spéculation invérifiable. Le NaN est ensuite traité comme les autres manquants.
    """
    df = df.copy()
    for id_patient, colonne in VALEURS_IMPOSSIBLES:
        ligne = df.index[df["id"] == id_patient][0]
        ancienne = float(df.at[ligne, colonne])
        df.at[ligne, colonne] = np.nan
        journaliser("6", f"valeur mise à NaN (id {id_patient}, ancienne valeur {ancienne})", colonne, 1,
                    "valeur physiologiquement impossible ; correction non démontrable")
    controle("NaN sod", int(df["sod"].isna().sum()), 88)
    controle("NaN pot", int(df["pot"].isna().sum()), 90)
    controle("sod minimum", float(df["sod"].min()), 104.0)
    controle("pot maximum", float(df["pot"].max()), 7.6)
    return df


# ÉTAPE 7 : vérifier qu'il ne reste aucune autre valeur impossible
def verifier_bornes(df: pd.DataFrame) -> None:
    """Contrôle les 14 variables avec des bornes larges : il ne doit rien rester hors bornes."""
    hors = {c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in BORNES.items()}
    controle("valeurs hors bornes (14 variables)", {c: n for c, n in hors.items() if n}, {})


# ÉTAPE 8 : valeurs extrêmes mais possibles (conservées)
def verifier_extremes(df: pd.DataFrame) -> None:
    """Démontre que les extrêmes de sc et bu sont cohérents avec la maladie, donc conservés.

    Si ces extrêmes sont réels, ils doivent appartenir à des patients ckd. Les supprimer ou
    les plafonner détruirait le signal que l'analyse cherche à mesurer.
    """
    extremes = df[(df["sc"] > 10) | (df["bu"] > 250)]
    controle("lignes avec sc > 10 ou bu > 250", len(extremes), 24)
    controle("classes de ces lignes", extremes["classification"].value_counts().to_dict(), {"ckd": 24})
    sains = df[df["classification"] == "notckd"]
    controle("notckd : sc max, bu max", (float(sains["sc"].max()), float(sains["bu"].max())), (1.2, 50.0))
    journaliser("8", "extrêmes conservés sans modification", "sc, bu", len(extremes),
                "valeurs possibles en insuffisance rénale avancée, toutes chez des ckd. "
                "Doute non vérifiable : id 148 (sc 48,1 / bu 26) et id 6 (sc 24 / bu 54)")

    # Cas particulier : bu = 1,5 (id 173), incohérente avec sc = 7,3 mais non démontrablement impossible.
    incoherentes = df.loc[df["bu"] < df["sc"], "id"].tolist()
    controle("ids avec bu < sc", incoherentes, [148, 173])
    journaliser("8", "valeur conservée et signalée comme douteuse (id 173)", "bu", 1,
                "bu = 1,5 non démontrablement impossible ; à inclure dans une analyse de sensibilité")


# ÉTAPE 9 : mécanisme des valeurs manquantes (information, aucune modification)
def analyser_manquants(df: pd.DataFrame) -> pd.DataFrame:
    """Compare le taux de manquants chez les ckd et les notckd (test du khi-deux)."""
    colonnes = [c for c in df.columns if c not in ("id", "classification")]
    ckd = df["classification"] == "ckd"
    lignes = []
    for colonne in colonnes:
        na = df[colonne].isna()
        if na.sum() == 0:
            continue
        p = chi2_contingency(pd.crosstab(na, df["classification"]), correction=False)[1]
        lignes.append({"variable": colonne, "n_NaN": int(na.sum()),
                       "%_ckd": round(na[ckd].mean() * 100, 1),
                       "%_notckd": round(na[~ckd].mean() * 100, 1), "p": p})
    tableau = pd.DataFrame(lignes).sort_values("n_NaN", ascending=False).set_index("variable")
    controle("total de cellules manquantes", int(tableau["n_NaN"].sum()), 1015)
    controle("rbc : % manquants ckd / notckd", (tableau.loc["rbc", "%_ckd"], tableau.loc["rbc", "%_notckd"]), (57.2, 6.0))
    journaliser("9", "analyse des manquants par classe (khi-deux)", "toutes", 1015,
                "les manquants sont fortement liés à la classe pour 12 variables : ils ne sont pas aléatoires")
    return tableau


# ÉTAPE 10 : imputation des variables catégorielles
def imputer_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Catégorie 'unknown' pour rbc et pc ; mode pour les variables à très peu de manquants."""
    df_imp = df.copy()
    for colonne in CAT_UNKNOWN:
        n = int(df_imp[colonne].isna().sum())
        df_imp[colonne] = df_imp[colonne].fillna("unknown")
        journaliser("10", "catégorie 'unknown'", colonne, n,
                    "manquants nombreux et liés à la classe : l'absence d'examen est conservée comme information")
    for colonne in CAT_MODE:
        n = int(df_imp[colonne].isna().sum())
        mode = df_imp[colonne].mode()[0]
        df_imp[colonne] = df_imp[colonne].fillna(mode)
        journaliser("10", f"mode = '{mode}'", colonne, n, "manquants très rares (<= 1 %), effet négligeable")

    controle("manquants catégoriels dans df_imp", int(df_imp[COLS_CAT].isna().sum().sum()), 0)
    controle("manquants catégoriels dans df (inchangé)", int(df[COLS_CAT].isna().sum().sum()), 234)
    for colonne, attendues in MODALITES_IMPUTEES_ATTENDUES.items():
        controle(f"répartition {colonne} après imputation", df_imp[colonne].value_counts().to_dict(), attendues)
    return df_imp


# ÉTAPE 11 : imputation des variables numériques
def imputer_numeriques(df: pd.DataFrame, df_imp: pd.DataFrame, avec_classe: bool) -> pd.DataFrame:
    """Imputation itérative multivariée (IterativeImputer), avec trois garde-fous.

    Chaque valeur manquante est estimée à partir des autres mesures du patient (corrélations
    fortes, par exemple hemo / pcv / rc). Si `avec_classe`, la classe (ckd = 1) sert de
    variable d'appui : sans elle, les valeurs estimées sont tirées vers la moyenne générale
    et atténuent légèrement les différences entre classes. La classe n'est jamais modifiée.
    ATTENTION : avec la classe, le fichier imputé ne convient pas à un modèle prédictif
    évalué sur des données non vues (la cible a participé à remplir les variables).

    Garde-fous : (1) valeurs ramenées dans la plage observée ; (2) al et su arrondis à des
    entiers de 0 à 5 ; (3) sg ramené à ses 5 paliers (pas de 0,005).
    """
    df_imp = df_imp.copy()
    X = df[COLS_NUM].copy()
    if avec_classe:
        X["ckd"] = (df["classification"] == "ckd").astype(float)

    imputeur = IterativeImputer(max_iter=10, random_state=0)
    A = pd.DataFrame(imputeur.fit_transform(X), columns=X.columns, index=df.index)[COLS_NUM]

    mini, maxi = df[COLS_NUM].min(), df[COLS_NUM].max()
    hors = ((A < mini) | (A > maxi)).sum()
    print("   hors plage avant correction :", hors[hors > 0].to_dict(), "(dépend de la version de scikit-learn)")
    A = A.clip(mini, maxi, axis=1)
    for colonne in ["al", "su"]:
        A[colonne] = A[colonne].round().clip(0, 5)
    A["sg"] = ((A["sg"] / 0.005).round() * 0.005).round(3).clip(1.005, 1.025)

    # Seules les cellules NaN sont remplacées : les valeurs observées ne sont jamais touchées.
    df_imp[COLS_NUM] = df[COLS_NUM].where(df[COLS_NUM].notna(), A)

    observees = df[COLS_NUM].notna()
    controle("NaN numériques dans df_imp", int(df_imp[COLS_NUM].isna().sum().sum()), 0)
    controle("valeurs observées inchangées",
             bool((df_imp[COLS_NUM].where(observees) == df[COLS_NUM].where(observees)).where(observees, True).all().all()), True)
    controle("toutes les valeurs dans la plage observée",
             bool(((df_imp[COLS_NUM] >= mini) & (df_imp[COLS_NUM] <= maxi)).all().all()), True)
    controle("sg : paliers", [float(x) for x in sorted(df_imp["sg"].unique())], [1.005, 1.01, 1.015, 1.02, 1.025])
    controle("al et su : entiers", bool((df_imp[["al", "su"]] % 1 == 0).all().all()), True)

    total = 0
    for colonne in COLS_NUM:
        n = int(df[colonne].isna().sum())
        total += n
        journaliser("11", "imputation itérative" + (" (classe en appui)" if avec_classe else " (sans la classe)"),
                    colonne, n, "estimation à partir des autres mesures du patient ; garde-fous de plage")
    controle("cellules numériques imputées", total, 781)
    return df_imp


# ÉTAPE 12 : vérification globale avant / après
def verifier_global(df: pd.DataFrame, df_imp: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Prouve que df_imp ne diffère de df que par les cellules complétées."""
    colonnes = [c for c in df.columns if c not in ("id", "classification")]
    controle("formes", (df.shape, df_imp.shape), ((NB_LIGNES, NB_COLONNES), (NB_LIGNES, NB_COLONNES)))
    controle("id et classification identiques",
             (bool((df["id"] == df_imp["id"]).all()), bool((df["classification"] == df_imp["classification"]).all())),
             (True, True))
    controle("NaN dans df / df_imp", (int(df.isna().sum().sum()), int(df_imp.isna().sum().sum())), (1015, 0))

    modifiees = df_imp[colonnes].astype(str) != df[colonnes].fillna("NA").astype(str)
    controle("cellules modifiées", int(modifiees.sum().sum()), 1015)
    controle("modifiées = exactement les cellules manquantes", bool((modifiees == df[colonnes].isna()).all().all()), True)
    controle("cible", df_imp["classification"].value_counts().to_dict(), {"ckd": 250, "notckd": 150})
    controle("types", df_imp.dtypes.astype(str).value_counts().to_dict(), {"float64": 14, "object": 12})
    controle("copie brute intacte (dates dans sg)", int(df_raw["sg"].map(est_date).sum()), 190)

    # Relations entre variables : information (le seuil de 0,15 est volontairement large).
    c1 = df[COLS_NUM].corr(method="spearman").values
    c2 = df_imp[COLS_NUM].corr(method="spearman").values
    ecart = np.abs(c1 - c2)
    np.fill_diagonal(ecart, 0)
    i, j = np.unravel_index(ecart.argmax(), ecart.shape)
    print(f"   écart max de corrélation de Spearman : {ecart.max():.3f} ({COLS_NUM[i]}-{COLS_NUM[j]}), "
          f"écart moyen : {ecart[np.triu_indices(len(COLS_NUM), 1)].mean():.3f}")
    controle("structure de corrélation préservée (écart max < 0,15)", bool(ecart.max() < 0.15), True)


# ÉTAPE 13 : export et relecture
def exporter(df: pd.DataFrame, df_imp: pd.DataFrame, dossier: Path, nom_imputé: str) -> None:
    """Écrit les CSV et le journal, puis les relit pour vérifier qu'ils sont fidèles."""
    dossier.mkdir(parents=True, exist_ok=True)
    chemin_clean = dossier / "kidney_clean.csv"
    chemin_imp = dossier / nom_imputé
    chemin_journal = dossier / "journal_nettoyage.csv"

    df.to_csv(chemin_clean, index=False, encoding="utf-8")
    df_imp.to_csv(chemin_imp, index=False, encoding="utf-8")
    pd.DataFrame(JOURNAL).to_csv(chemin_journal, index=False, encoding="utf-8")

    a, b = pd.read_csv(chemin_clean), pd.read_csv(chemin_imp)
    controle("relecture : formes", (a.shape, b.shape), ((NB_LIGNES, NB_COLONNES), (NB_LIGNES, NB_COLONNES)))
    controle("relecture : NaN", (int(a.isna().sum().sum()), int(b.isna().sum().sum())), (1015, 0))
    pd.testing.assert_frame_equal(a, df.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(b, df_imp.reset_index(drop=True), check_dtype=False)
    print("[OK] relecture : contenu identique à la version en mémoire")
    print(f"\nFichiers écrits dans {dossier.resolve()} :")
    for chemin in (chemin_clean, chemin_imp, chemin_journal):
        print("   -", chemin.name)


# PROGRAMME PRINCIPAL
def main() -> None:
    parseur = argparse.ArgumentParser(description="Nettoyage du dataset Kidney Disease")
    parseur.add_argument("--excel", type=Path, required=True, help="chemin du fichier Excel d'origine")
    parseur.add_argument("--sortie", type=Path, default=Path("."), help="dossier de sortie")
    parseur.add_argument("--sans-classe", action="store_true",
                         help="imputer les variables numériques sans utiliser la classe (fichier distinct)")
    args = parseur.parse_args()

    print("ÉTAPE 1 : chargement et audit du fichier brut")
    df_raw = charger_brut(args.excel)
    auditer_brut(df_raw)

    print("\nÉTAPE 2 : décodage des dates")
    df = decoder_dates(df_raw)
    verifier_decodage(df_raw, df)

    print("\nÉTAPE 3 : remplacement du marqueur '?'")
    df = remplacer_interrogations(df)

    print("\nÉTAPE 4 : conversion des variables quantitatives")
    df = convertir_numeriques(df)

    print("\nÉTAPE 5 : contrôle des variables catégorielles")
    controler_categories(df)

    print("\nÉTAPE 6 : valeurs physiologiquement impossibles")
    df = traiter_valeurs_impossibles(df)

    print("\nÉTAPE 7 : contrôle des bornes")
    verifier_bornes(df)

    print("\nÉTAPE 8 : valeurs extrêmes conservées")
    verifier_extremes(df)

    print("\nÉTAPE 9 : analyse des valeurs manquantes")
    manquants = analyser_manquants(df)
    print(manquants.round(5).to_string())

    print("\nÉTAPE 10 : imputation des variables catégorielles")
    df_imp = imputer_categories(df)

    avec_classe = not args.sans_classe
    print(f"\nÉTAPE 11 : imputation des variables numériques ({'avec' if avec_classe else 'sans'} la classe)")
    df_imp = imputer_numeriques(df, df_imp, avec_classe)

    print("\nÉTAPE 12 : vérification globale avant / après")
    verifier_global(df, df_imp, df_raw)

    print("\nÉTAPE 13 : export et relecture")
    nom_imp = "kidney_clean_imputed.csv" if avec_classe else "kidney_clean_imputed_sans_classe.csv"
    exporter(df, df_imp, args.sortie, nom_imp)


if __name__ == "__main__":
    main()