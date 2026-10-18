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

# Ciudades (Dominio) con peor F1 de n6_lightgbm en el holdout (etapa 0)
CIUDADES_PEOR_F1 = ["BUCARAMANGA", "MANIZALES", "PEREIRA", "BOGOTA", "IBAGUE"]
