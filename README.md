# Diagnostic de la maladie rénale chronique (MRC / CKD)

Projet du cours « Applications de data analytics », IFRI (Université d'Abomey-Calavi), L3 Intelligence Artificielle. Thème 3.

Ce dossier contient le script du travail : nettoyage des données et analyses (uni/bi/multivariée). Il n'y a pas de modèle prédictif.

## Contenu

```
projet_theme3/
├── README.md
├── requirements.txt
├── data/
│   ├── Base Kidney diseases.xlsx   données d'origine (400 patients × 26 colonnes)
│   ├── kidney_clean.csv            données corrigées, valeurs manquantes conservées
│   ├── kidney_clean_imputed.csv    mêmes données, valeurs manquantes imputées
│   └── journal_nettoyage.csv       journal des modifications
├── notebooks/
│   ├── 01_inspection.ipynb
│   ├── 02_analyse_univariee_bivariee.ipynb
│   ├── 03_analyses_inferentielles.ipynb
│   ├── 04_analyse_multivariée.ipynb
│   └── 05_analyse_MRC.ipynb
└── scripts/
    └── nettoyage.py
```

## Installation

```
pip install -r requirements.txt
```

## Ordre d'exécution

1. **Inspection.** Le notebook `01_inspection.ipynb` lit l'Excel d'origine et met en évidence les problèmes à corriger (dates, marqueurs `?`, types, manquants).
2. **Nettoyage.** Depuis la racine du projet :

   ```
   python scripts/nettoyage.py --excel "data/Base Kidney diseases.xlsx" --sortie data
   ```

   Le script s'arrête avec une erreur si un contrôle échoue. Il écrit les trois fichiers de `data/`.
   L'option `--sans-classe` produit une variante imputée sans utiliser la classe, dans un fichier distinct.
3. **Analyses**, dans l'ordre de leur numéro, en lançant les notebooks depuis le dossier `notebooks/` :

| Notebook                                | Données lues                |
| --------------------------------------- | ---------------------------- |
| `02_analyse_univariee_bivariee.ipynb` | `kidney_clean.csv`         |
| `03_analyses_inferentielles.ipynb`    | `kidney_clean.csv`         |
| `04_analyse_multivariée.ipynb`       | `kidney_clean_imputed.csv` |
| `05_analyse_MRC.ipynb`                | `kidney_clean_imputed.csv` |

Le notebook `05_analyse_MRC.ipynb` réalise une classification ascendante
hiérarchique (CAH) descriptive. Trois groupes sont retenus pour distinguer un
profil de référence, une atteinte rénale modérée et une atteinte rénale sévère,
même si la silhouette favorise légèrement `k = 2` (`0,25` contre `0,20` pour
`k = 3`). La classification sert uniquement à contrôler les groupes obtenus ;
elle n'est pas utilisée directement dans le clustering, mais elle a participé à
l'imputation du fichier analysé.

## Avertissements sur les données

- Utiliser `kidney_clean.csv` pour les statistiques descriptives et les tests `ckd` / `notckd`, en indiquant le `n` réel de chaque variable.
- `kidney_clean_imputed.csv` est utilisé pour les analyses multivariées et la CAH. L'imputation guidée par la classe peut accentuer les écarts entre groupes ; les résultats doivent donc être interprétés comme descriptifs.
- La forêt aléatoire, l'exactitude et les AUC du notebook `05_analyse_MRC.ipynb` servent à classer et décrire les variables qui distinguent les groupes. Elles ne constituent pas un modèle prédictif du projet.
- Le fichier imputé ne doit pas être utilisé pour évaluer un modèle prédictif sur des données non vues : la classe a servi à remplir les valeurs manquantes.
- Ne pas ouvrir les CSV par double-clic dans Excel : c'est probablement ainsi que le fichier d'origine a été abîmé (décimaux convertis en dates).
