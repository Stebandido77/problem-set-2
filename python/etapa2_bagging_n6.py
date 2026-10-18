"""Etapa 2: seed bagging de 5 semillas con los hiperparámetros de n6 sobre la base
v2 (a4 + auxiliar laboral + auxiliar total). Es el contrapunto del bagging con
los parámetros de Optuna, que en CV empatan con los de n6.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa2_bagging_n6.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import PARAMETROS_N6, RUTA_TABLAS
from etapa2_optuna import SEMILLAS_BAGGING
from funciones.bases import armar_matriz, cargar_insumos
from funciones.modelado import evaluar_candidato, f1_subgrupos


def main():
    insumos = cargar_insumos(auxiliar_total=True)
    particion = insumos["particion"]
    resultado = evaluar_candidato("n7_n6param_a4_total_bag5", armar_matriz(insumos), particion,
                                  PARAMETROS_N6, list(insumos["auxiliares"].values()),
                                  con_test=True, semillas=SEMILLAS_BAGGING)
    fila = {"candidato": resultado["nombre"], "semillas": len(SEMILLAS_BAGGING),
            "umbral": resultado["umbral"], "f1_cv": resultado["f1_cv"],
            "f1_cv_sd": resultado["f1_cv_sd"], "f1_holdout": resultado["f1_holdout"],
            "precision_holdout": resultado["precision_holdout"],
            "recall_holdout": resultado["recall_holdout"],
            **f1_subgrupos(resultado, insumos["hogar"], particion),
            "tiempo_min": resultado["tiempo_min"]}
    pd.DataFrame([fila]).to_csv(RUTA_TABLAS / "bagging_n6param_a4_total.csv", index=False)
    print(pd.DataFrame([fila]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
