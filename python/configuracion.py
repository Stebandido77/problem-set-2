"""Parámetros globales del nivel 7 (Python).

Espejo de scripts/00_configuracion.R: misma semilla, mismos folds (leídos de R) y
rutas relativas a la raíz del repositorio.
"""

import os
from pathlib import Path

SEMILLA = 2026
N_FOLDS = 5

RAIZ = Path(__file__).resolve().parents[1]
RUTA_PROCESADOS = RAIZ / "stores" / "procesados"
RUTA_NIVEL7 = RUTA_PROCESADOS / "nivel7"           # caché de variables y predicciones
RUTA_RESULTADOS = RAIZ / "stores" / "resultados"
RUTA_SUBMISSIONS = RAIZ / "stores" / "submissions" / "nivel7"
RUTA_LOGS = RAIZ / "stores" / "logs"
RUTA_TABLAS = RAIZ / "views" / "tablas" / "nivel7"
RUTA_FIGURAS = RAIZ / "views" / "figuras" / "nivel7"

for ruta in (RUTA_NIVEL7, RUTA_LOGS, RUTA_TABLAS, RUTA_FIGURAS):
    ruta.mkdir(parents=True, exist_ok=True)

# Un solo proceso pesado; LightGBM paraleliza con hilos sobre una copia de los datos
N_HILOS = int(os.environ.get("PS2_NUCLEOS", max((os.cpu_count() or 2) - 1, 1)))

# Hiperparámetros de n6_lightgbm (stores/modelos/n6_lightgbm.rds) traducidos de
# bonsai a LightGBM: mtry -> feature_fraction_bynode, sample_size ->
# bagging_fraction con bagging_freq = 1, num_leaves queda en el default (31).
PARAMETROS_N6 = {
    "objective": "binary",
    "num_iterations": 1852,
    "learning_rate": 0.0113973670271091,
    "max_depth": 12,
    "num_leaves": 31,
    "min_data_in_leaf": 171,
    "feature_fraction_bynode": 0.81882502989843498,
    "min_gain_to_split": 7.204294082254051e-10,
    "bagging_fraction": 0.70311201574280857,
    "bagging_freq": 1,
}
UMBRAL_N6 = 0.35500138702806505
F1_HOLDOUT_N6 = 0.7487895186556537

# Los mismos hiperparámetros de n6 traducidos a XGBoost (etapa 14): histogramas con
# crecimiento por hoja (lossguide, max_leaves = num_leaves), misma profundidad, tasa
# y rondas; subsample y colsample_bynode son bagging_fraction y
# feature_fraction_bynode; gamma es min_gain_to_split. XGBoost no tiene
# min_data_in_leaf: min_child_weight (suma de hessianas p(1 - p), ~0,16 con la tasa
# de pobres de 0,2) aproxima las 171 filas por hoja con 171 x 0,16 ~ 27.
PARAMETROS_N6_XGBOOST = {
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "num_boost_round": PARAMETROS_N6["num_iterations"],
    "learning_rate": PARAMETROS_N6["learning_rate"],
    "max_depth": PARAMETROS_N6["max_depth"],
    "max_leaves": PARAMETROS_N6["num_leaves"],
    "min_child_weight": 27,
    "gamma": PARAMETROS_N6["min_gain_to_split"],
    "subsample": PARAMETROS_N6["bagging_fraction"],
    "colsample_bynode": PARAMETROS_N6["feature_fraction_bynode"],
}

# Ciudades (Dominio) con peor F1 de n6_lightgbm en el holdout (etapa 0)
CIUDADES_PEOR_F1 = ["BUCARAMANGA", "MANIZALES", "PEREIRA", "BOGOTA", "IBAGUE"]
