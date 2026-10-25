"""Etapa 13: mezcla final desde las predicciones guardadas, sin reentrenar nada.

Combina la base n7_a_agregados_bag5 (LightGBM n6 sobre A, bagging de 5 semillas)
con los candidatos menos correlacionados con ella que quedaron de las etapas 10 y
12: A + KNN fuera de fold (n12_a_knn_1s), RealMLP (n10_realmlp_bag5) y DeepSets
con atención (n12_conjuntos_bag3). Los pesos se eligen SOLO en la OOF (simplex de
paso 0,05, F1 con el umbral óptimo de buscar_umbral_f1()) y el umbral también
sale de la OOF; holdout y test reciben la misma combinación (mezclar_por_oof()).

n12_a_knn_1s no tiene predicción de test guardada (la etapa 12 no reajustó sobre
el 100 % porque no pasó la regla), así que:
  n13_mezcla_con_knn  los cuatro candidatos: solo F1 CV y holdout, sin CSV
  n13_mezcla_final    los candidatos con test: F1 CV, holdout y CSV
El CSV se genera aunque la mezcla no cumpla la regla (+0,002 en CV y en holdout
sobre la base), como submission extra; queda validado, en la bitácora y en el
leaderboard del nivel 7. Tablas en views/tablas/nivel7/etapa13_*.csv.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa13_mezcla_final.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from configuracion import RUTA_NIVEL7, RUTA_SUBMISSIONS, RUTA_TABLAS
from etapa4_leaderboard import agregar_filas_extra, armar_fila
from etapa8_cierre import grilla_simplex
from etapa12_enfoques import BASE, MEJORA_MINIMA, fila_tabla, guardar_filas
from funciones.bitacora import registrar_log
from funciones.datos import cargar_particion
from funciones.mezclas import mezclar_por_oof
from funciones.modelado import resultado_desde_predicciones
from funciones.submission import guardar_submission, validar_submission

ETAPA = "etapa13"
PASO_SIMPLEX = 0.05
ALGORITMO_MEZCLA = "Stacking"
MEZCLA_CON_KNN = "n13_mezcla_con_knn"
MEZCLA_FINAL = "n13_mezcla_final"
ARCHIVO_CSV = "n13_mezcla_final.csv"
ARCHIVO_CSV_BASE = "n7_a_agregados_bag5.csv"
# Candidato -> descripción corta; la base va primero para que encabece los pesos
COMPONENTES = {
    BASE: "base: LightGBM n6 sobre A, bagging 5 semillas (etapa 7)",
    "n12_a_knn_1s": "A + tasa de pobreza de los k = (10, 50, 200) vecinos fuera de fold, 1 semilla (etapa 12)",
    "n10_realmlp_bag5": "RealMLP-TD sobre A, bagging 5 semillas (etapa 10)",
    "n12_conjuntos_bag3": "DeepSets con atención, bagging 3 semillas (etapa 12)",
}


def componentes_con_test(resultados):
    """Candidatos (en el orden dado) que tienen predicción de test guardada.

    Argumentos: resultados = {nombre: resultado_desde_predicciones()}.
    Devuelve: lista de nombres.
    """
    return [nombre for nombre, resultado in resultados.items() if "prediccion_test" in resultado]


def grilla_pesos(n):
    """Pesos no negativos que suman 1 en múltiplos de PASO_SIMPLEX para n
    candidatos; incluye a la base sola, así la mezcla nunca pierde en OOF."""
    return grilla_simplex(n, PASO_SIMPLEX)


def texto_pesos(componentes, pesos):
    """'candidato peso + ...' con dos decimales, para descripciones y bitácora."""
    return " + ".join(f"{c} {w:.2f}" for c, w in zip(componentes, pesos))


def mezclar(nombre, componentes, tabla_pesos, verdad):
    """Mezcla ponderada con pesos elegidos solo en OOF y umbral OOF.

    Argumentos: nombre = candidato nuevo; componentes = candidatos a mezclar;
        tabla_pesos = sufijo del CSV con el F1 OOF de cada vector de la grilla;
        verdad = pobre por id.
    Devuelve: (resultado, pesos elegidos).
    """
    inicio = time.time()
    resultado, pesos, tabla = mezclar_por_oof(nombre, componentes, grilla_pesos(len(componentes)), verdad)
    tabla.to_csv(RUTA_TABLAS / f"{ETAPA}_pesos_{tabla_pesos}.csv", index=False)
    resultado["tiempo_min"] = (time.time() - inicio) / 60
    registrar_log(nombre, f"pesos {np.round(pesos, 2)} | F1 OOF {tabla['f1_oof'].max():.4f} | "
                          f"CV {resultado['f1_cv']:.4f} | holdout {resultado['f1_holdout']:.4f}", inicio)
    return resultado, pesos


def fila_mezcla(resultado, verdad, componentes, pesos, base):
    """Fila de la tabla de la etapa para una mezcla (deltas y regla frente a la base)."""
    fila = fila_tabla(resultado, verdad, f"mezcla {texto_pesos(componentes, pesos)} (pesos por OOF)", base,
                      base["oof"]["probabilidad"])
    fila["tiene_test"] = "prediccion_test" in resultado
    return fila


def fila_leaderboard(nombre, componentes, pesos, resultado, archivo, verdad):
    """Fila del leaderboard del nivel 7 de una mezcla (variables: la base A completa)."""
    columnas = json.loads((RUTA_NIVEL7 / "etapa10" / "columnas_holdout.json").read_text(encoding="utf-8"))
    assert isinstance(columnas, list) and all(isinstance(c, str) for c in columnas), "columnas_holdout.json cambió"
    return armar_fila(
        nombre, ALGORITMO_MEZCLA, f"Promedio ponderado de probabilidades: {texto_pesos(componentes, pesos)}",
        columnas, f"pesos en el simplex de paso {PASO_SIMPLEX} por F1 OOF; umbral OOF",
        len(grilla_pesos(len(componentes))), "; ".join(f"peso_{c} = {w:.2f}" for c, w in zip(componentes, pesos)),
        resultado.get("tiempo_min", pd.NA), archivo, verdad)


def comparar_con_csv_base(ruta, ids_test):
    """Vuelve a validar el CSV escrito y cuenta en cuántos hogares difiere del de la base."""
    submission = pd.read_csv(ruta, dtype={"id": str, "pobre": np.int64})
    tasa = validar_submission(submission, ids_test)
    base = pd.read_csv(RUTA_SUBMISSIONS / ARCHIVO_CSV_BASE, dtype={"id": str, "pobre": np.int64})
    clase_base = base.set_index("id")["pobre"].reindex(submission["id"]).to_numpy()
    distintos = int((submission["pobre"].to_numpy() != clase_base).sum())
    return tasa, distintos


def main():
    inicio = time.time()
    registrar_log(ETAPA, "inicio (mezclas desde predicciones guardadas, sin reentrenar)")
    particion = cargar_particion()
    verdad = particion.set_index("id")["pobre"]
    ids_test = particion.loc[particion["base"] == "test", "id"]

    resultados = {nombre: resultado_desde_predicciones(nombre, verdad) for nombre in COMPONENTES}
    base = resultados[BASE]
    filas = []
    for nombre, descripcion in COMPONENTES.items():
        fila = fila_tabla(resultados[nombre], verdad, descripcion, base, base["oof"]["probabilidad"])
        fila["tiene_test"] = "prediccion_test" in resultados[nombre]
        filas.append(fila)
        registrar_log(nombre, f"CV {fila['f1_cv']:.4f} | holdout {fila['f1_holdout']:.4f} | "
                              f"test {'si' if fila['tiene_test'] else 'no'}")

    # 1. Los cuatro candidatos: solo CV y holdout (KNN no tiene test)
    componentes = list(COMPONENTES)
    con_knn, pesos_knn = mezclar(MEZCLA_CON_KNN, componentes, "mezcla_con_knn", verdad)
    filas.append(fila_mezcla(con_knn, verdad, componentes, pesos_knn, base))

    # 2. Los candidatos con test: CSV siempre, como submission extra si no pasa la regla
    con_test = componentes_con_test(resultados)
    assert con_test[0] == BASE and len(con_test) >= 2, "la base y al menos otro candidato deben tener test"
    final, pesos_final = mezclar(MEZCLA_FINAL, con_test, "mezcla_final", verdad)
    fila_final = fila_mezcla(final, verdad, con_test, pesos_final, base)
    filas.append(fila_final)

    especificacion = (f"Promedio ponderado de probabilidades: {texto_pesos(con_test, pesos_final)} | pesos en el "
                      f"simplex de paso {PASO_SIMPLEX} por F1 OOF | umbral {final['umbral']:.4f} (OOF)")
    if not fila_final["pasa_regla"]:
        especificacion += (f" | submission extra: no cumple la regla de +{MEJORA_MINIMA:.3f} en CV y holdout "
                           f"sobre {BASE}")
    ruta, tasa = guardar_submission(final, ARCHIVO_CSV, ids_test, especificacion, algoritmo=ALGORITMO_MEZCLA)
    tasa, distintos = comparar_con_csv_base(ruta, ids_test)
    fila_final["archivo_submission"] = f"nivel7/{ruta.name}"
    registrar_log(ETAPA, f"CSV {ruta.name}: tasa de pobres {tasa:.4f}, {distintos} hogares distintos a "
                         f"{ARCHIVO_CSV_BASE} ({distintos / len(ids_test):.2%})"
                         + ("" if fila_final["pasa_regla"] else "; no cumple la regla, va como submission extra"))

    # 3. Leaderboard del nivel 7 y tabla de la etapa
    agregar_filas_extra([
        fila_leaderboard(MEZCLA_CON_KNN, componentes, pesos_knn, con_knn, None, verdad),
        fila_leaderboard(MEZCLA_FINAL, con_test, pesos_final, final, f"nivel7/{ruta.name}", verdad),
    ])
    tabla = guardar_filas(f"{ETAPA}_candidatos", filas)
    columnas = ["candidato", "f1_cv", "f1_cv_sd", "f1_holdout", "precision_holdout", "recall_holdout",
                "f1_banda_08_12", "delta_cv", "delta_holdout", "pasa_regla", "pearson_oof_con_lightgbm", "tiene_test"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(tabla[columnas].round(4).to_string(index=False))
    registrar_log(ETAPA, f"fin: {MEZCLA_FINAL} CV {final['f1_cv']:.4f} ({final['f1_cv'] - base['f1_cv']:+.4f}) | "
                         f"holdout {final['f1_holdout']:.4f} ({final['f1_holdout'] - base['f1_holdout']:+.4f})", inicio)
    return tabla


if __name__ == "__main__":
    main()
