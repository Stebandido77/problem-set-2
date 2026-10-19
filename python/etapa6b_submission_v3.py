"""Etapa 6b: submission de n7_v3_fuentes_bag5 aunque su holdout (0,7792) quedó
bajo el corte orientativo de 0,780 de la etapa 6.

Justificación: frente a n7_n6param_a4_total_bag5 mejora de forma consistente en
CV (+0,005) y en holdout (+0,003), con más precisión (menos filtración) y el
mismo recall (misma subcobertura).

Usa las predicciones de test que guardó la etapa 6 (reajuste sobre el 100% del
train, bagging de 5 semillas) y el umbral elegido en OOF; no reentrena nada.
Escribe stores/submissions/nivel7/n7_v3_final.csv, lo agrega a la bitácora con
f1_kaggle vacía y compara su tasa de pobres en test con la de n7_final.csv.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa6b_submission_v3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import RUTA_NIVEL7, RUTA_SUBMISSIONS
from funciones.bitacora import registrar_log
from funciones.datos import cargar_particion
from funciones.metricas import buscar_umbral_f1, calcular_metricas, metricas_por_fold
from funciones.submission import guardar_submission, validar_submission

CANDIDATO = "n7_v3_fuentes_bag5"
ARCHIVO = "n7_v3_final.csv"
REFERENCIA = "n7_final.csv"


def resultado_desde_predicciones(nombre, verdad):
    """Arma el diccionario que espera guardar_submission() desde las
    predicciones guardadas de un candidato (OOF, holdout y test).

    Argumentos: nombre = candidato; verdad = pobre por id.
    Devuelve: diccionario con nombre, umbral, f1_cv, f1_holdout y prediccion_test.
    """
    predicciones = pd.read_parquet(RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").set_index("id")
    oof = predicciones[predicciones["tipo"] == "oof"]
    holdout = predicciones[predicciones["tipo"] == "holdout"]["probabilidad"]
    umbral, _ = buscar_umbral_f1(verdad.loc[oof.index], oof["probabilidad"])
    cv = metricas_por_fold(verdad.loc[oof.index], oof["probabilidad"], oof["fold"], umbral)
    metricas_holdout = calcular_metricas(verdad.loc[holdout.index].astype(int), (holdout >= umbral).astype(int))
    return {"nombre": nombre, "umbral": umbral, "f1_cv": cv["f1_cv"], "f1_holdout": metricas_holdout["f1"],
            "prediccion_test": predicciones[predicciones["tipo"] == "test"]["probabilidad"]}


def main():
    particion = cargar_particion()
    verdad = particion.set_index("id")["pobre"]
    ids_test = particion.loc[particion["base"] == "test", "id"]
    resultado = resultado_desde_predicciones(CANDIDATO, verdad)
    assert len(resultado["prediccion_test"]) == len(ids_test), "faltan predicciones de test"

    ruta, tasa = guardar_submission(
        resultado, ARCHIVO, ids_test,
        "LightGBM (hiperparámetros de n6), seed bagging de 5 semillas | nivel 3 + v2 + auxiliares "
        "de ingreso laboral y total (v2) y por fuente en dos partes (v3) con OOF anidada | "
        f"umbral {resultado['umbral']:.4f} (OOF) | holdout bajo el corte orientativo de 0,780 "
        "pero mejora a n7_final en CV y holdout")
    tasa_referencia = validar_submission(pd.read_csv(RUTA_SUBMISSIONS / REFERENCIA, dtype={"id": str}), ids_test)
    registrar_log("etapa6b", f"{ruta.name} umbral {resultado['umbral']:.4f} tasa {tasa:.4f}")
    print(f"{ARCHIVO}: umbral {resultado['umbral']:.4f} | F1 CV {resultado['f1_cv']:.4f} | "
          f"holdout {resultado['f1_holdout']:.4f} | tasa de pobres en test {tasa:.4f}")
    print(f"{REFERENCIA}: tasa de pobres en test {tasa_referencia:.4f} | diferencia {tasa - tasa_referencia:+.4f}")


if __name__ == "__main__":
    main()
