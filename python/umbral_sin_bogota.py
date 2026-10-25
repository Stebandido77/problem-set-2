"""Umbral de F1 sin Bogotá para el modelo final del nivel 7.

El test de Kaggle no trae hogares de Bogotá (6,4% del train, con tasa de
pobreza de 9,7%). Se recalcula el umbral óptimo con las predicciones OOF
excluyendo Bogotá y se compara con el umbral actual en el holdout sin Bogotá.
Si el nuevo umbral mejora ese F1, se escribe n7_final_sin_bogota.csv; si no, el
CSV final usa el umbral actual (n7_final.csv).

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/umbral_sin_bogota.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import RUTA_NIVEL7, RUTA_TABLAS
from funciones.bitacora import registrar_log
from funciones.datos import cargar_hogares, cargar_particion
from funciones.metricas import buscar_umbral_f1, calcular_metricas, metricas_por_fold
from funciones.submission import guardar_submission

CANDIDATO = "n7_n6param_a4_total_bag5"
CIUDAD_AUSENTE = "BOGOTA"


def main():
    particion = cargar_particion().set_index("id")
    dominio = cargar_hogares()["dominio"].astype(str)
    predicciones = pd.read_parquet(RUTA_NIVEL7 / f"predicciones_{CANDIDATO}.parquet").set_index("id")
    predicciones["verdad"] = particion["pobre"].reindex(predicciones.index)
    predicciones["sin_bogota"] = dominio.reindex(predicciones.index).ne(CIUDAD_AUSENTE).to_numpy()

    oof = predicciones[predicciones["tipo"] == "oof"]
    holdout = predicciones[predicciones["tipo"] == "holdout"]
    test = predicciones[predicciones["tipo"] == "test"]
    assert not test["sin_bogota"].eq(False).any(), "el test sí trae hogares de Bogotá"

    umbral_actual, _ = buscar_umbral_f1(oof["verdad"], oof["probabilidad"])
    oof_sb = oof[oof["sin_bogota"]]
    umbral_sin_bogota, _ = buscar_umbral_f1(oof_sb["verdad"], oof_sb["probabilidad"])
    holdout_sb = holdout[holdout["sin_bogota"]]

    filas = []
    for etiqueta, umbral in (("actual (OOF completo)", umbral_actual), ("sin Bogotá (OOF sin Bogotá)", umbral_sin_bogota)):
        metricas = calcular_metricas(holdout_sb["verdad"].astype(int), (holdout_sb["probabilidad"] >= umbral).astype(int))
        cv = metricas_por_fold(oof_sb["verdad"], oof_sb["probabilidad"], oof_sb["fold"], umbral)
        filas.append({"umbral_tipo": etiqueta, "umbral": umbral,
                      "f1_cv_sin_bogota": cv["f1_cv"],
                      "f1_holdout_sin_bogota": metricas["f1"],
                      "precision_holdout_sin_bogota": metricas["precision"],
                      "recall_holdout_sin_bogota": metricas["recall"],
                      "tasa_pobres_test": (test["probabilidad"] >= umbral).mean()})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / "umbral_sin_bogota.csv", index=False)
    print(tabla.round(4).to_string(index=False))

    adoptar = filas[1]["f1_holdout_sin_bogota"] > filas[0]["f1_holdout_sin_bogota"]
    umbral = umbral_sin_bogota if adoptar else umbral_actual
    nombre_archivo = "n7_final_sin_bogota.csv" if adoptar else "n7_final.csv"
    fila = filas[1] if adoptar else filas[0]
    resultado = {
        "nombre": CANDIDATO, "umbral": umbral, "prediccion_test": test["probabilidad"],
        "f1_cv": fila["f1_cv_sin_bogota"], "f1_holdout": fila["f1_holdout_sin_bogota"],
    }
    especificacion = (
        "LightGBM (hiperparámetros de n6), seed bagging de 5 semillas | nivel 3 + v2 + auxiliares de "
        f"ingreso laboral y total con OOF anidada | umbral {umbral:.4f} "
        f"({'OOF sin Bogotá' if adoptar else 'OOF completo'}); F1 CV y holdout medidos sin Bogotá")
    ids_test = particion.index[particion["base"] == "test"]
    ruta, tasa = guardar_submission(resultado, nombre_archivo, ids_test, especificacion)
    registrar_log("umbral_sin_bogota", f"{ruta.name} umbral {umbral:.4f} tasa {tasa:.4f}")
    print(f"{'Adoptado' if adoptar else 'No mejora: se queda'} el umbral {umbral:.4f} -> {ruta.name} (tasa test {tasa:.4f})")


if __name__ == "__main__":
    main()
