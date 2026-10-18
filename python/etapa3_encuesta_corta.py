"""Etapa 3b: frontera de la encuesta corta con NSGA-II y Logit.

Candidatas: las variables originales del nivel 3 y de v2 (laborales, perceptor,
hogares pequeños). Los auxiliares de ingreso NO entran: una encuesta corta no
puede apoyarse en un modelo entrenado con el ingreso de la encuesta larga.
Aptitud barata: Logit con 3 folds del 80% (folds 1-3 de R como validación).
Los puntos no dominados se reevalúan con los 5 folds (umbral OOF) y en holdout.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa3_encuesta_corta.py [minutos_ga]
    python/.venv/Scripts/python.exe python/etapa3_encuesta_corta.py graficar   (solo la figura)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from configuracion import RUTA_FIGURAS, RUTA_NIVEL7, RUTA_RESULTADOS, RUTA_TABLAS, SEMILLA
from funciones.bitacora import Presupuesto, registrar_log
from funciones.datos import cargar_hogares, cargar_particion, cargar_personas, preparar_matriz
from funciones.metricas import calcular_metricas
from funciones.seleccion import (MuestreoDensidades, ProblemaEncuesta, evaluar_en_holdout,
                                 mapear_columnas, preparar_division)
from funciones.variables import VARIABLES_NIVEL_3, construir_grupos

BLOQUES_CANDIDATOS = ["nivel_3", "laborales", "perceptor", "pequenos"]
VARIABLES_NIVEL_E = ["n_personas", "dormitorios", "tenencia", "log_arriendo", "dominio",
                     "educ_maxima_adultos", "n_ocupados", "n_menores_5", "prop_subsidiado",
                     "jefe_mujer"]
MEJOR_N7 = "n7_n6param_a4_total_bag5"
TAMANOS_REPORTE = (5, 10, 15)


def frontera_no_dominada(cache):
    """Mejor F1 por número de variables entre todo lo evaluado, sin puntos dominados.

    Devuelve: lista de (n_variables, f1, mascara) ordenada por n_variables.
    """
    mejores = {}
    for llave, f1 in cache.items():
        mascara = np.frombuffer(llave, dtype=bool)
        n = int(mascara.sum())
        if n not in mejores or f1 > mejores[n][0]:
            mejores[n] = (f1, mascara)
    frontera, mejor_f1 = [], -1.0
    for n in sorted(mejores):
        f1, mascara = mejores[n]
        if f1 > mejor_f1:  # con más variables solo entra si mejora el F1
            frontera.append((n, f1, mascara))
            mejor_f1 = f1
    return frontera


def referencias_r():
    """F1 holdout de los modelos de referencia en el leaderboard interno."""
    tabla = pd.read_csv(RUTA_RESULTADOS / "leaderboard_interno.csv").set_index("candidato")
    return {c: (tabla.loc[c, "f1_holdout"], int(tabla.loc[c, "n_variables"]))
            for c in ("nE_logit", "nE_xgboost", "n3_logit")}


def errores_mejor_n7(particion):
    """Falsos negativos y positivos del mejor n7 en holdout, con su umbral OOF."""
    from funciones.metricas import buscar_umbral_f1
    predicciones = pd.read_parquet(RUTA_NIVEL7 / f"predicciones_{MEJOR_N7}.parquet").set_index("id")
    verdad = particion.set_index("id")["pobre"]
    oof = predicciones[predicciones["tipo"] == "oof"]
    umbral, _ = buscar_umbral_f1(verdad.loc[oof.index], oof["probabilidad"])
    holdout = predicciones[predicciones["tipo"] == "holdout"]
    return calcular_metricas(verdad.loc[holdout.index].astype(int),
                             (holdout["probabilidad"] >= umbral).astype(int))


def graficar_frontera(frontera, referencias, f1_n7):
    """F1 holdout vs. número de variables, con líneas de referencia."""
    azul, gris, tinta, tinta_suave = "#2a78d6", "#9aa0a6", "#202124", "#5f6368"
    figura, eje = plt.subplots(figsize=(10, 5.6))
    eje.plot(frontera["n_variables"], frontera["f1_holdout"], color=azul, linewidth=2,
             marker="o", markersize=5, zorder=3, label="Frontera NSGA-II (Logit)")
    lineas = [(f"Mejor n7 (LightGBM + auxiliares, {f1_n7:.3f})", f1_n7)]
    lineas += [(f"{nombre} ({n} var., {f1:.3f})", f1) for nombre, (f1, n) in referencias.items()]
    for texto, valor in lineas:
        eje.axhline(valor, color=gris, linewidth=1, linestyle="--", zorder=1)
        eje.text(frontera["n_variables"].max(), valor + 0.003, texto, ha="right", va="bottom",
                 fontsize=9, color=tinta_suave)
    for _, (f1, n) in referencias.items():
        eje.plot(n, f1, marker="s", markersize=6, color=gris, zorder=2)
    for n in TAMANOS_REPORTE:
        punto = frontera[frontera["n_variables"] <= n].tail(1)
        if len(punto):
            eje.annotate(f"{int(punto['n_variables'].iloc[0])} var.: {punto['f1_holdout'].iloc[0]:.3f}",
                         (punto["n_variables"].iloc[0], punto["f1_holdout"].iloc[0]),
                         textcoords="offset points", xytext=(10, -22), fontsize=9, color=tinta,
                         bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"},
                         arrowprops={"arrowstyle": "-", "color": tinta_suave, "linewidth": 0.8})
    eje.set_xlabel("Número de variables originales en la encuesta", color=tinta)
    eje.set_ylabel("F1 en holdout", color=tinta)
    figura.text(0.01, 0.97, "¿Cuánto F1 cuesta acortar la encuesta?", fontsize=14,
                fontweight="bold", color=tinta, va="top")
    figura.text(0.01, 0.92, "Frontera F1 vs. número de variables (Logit); holdout del 20%, umbral elegido "
                "con OOF. Cuadrados: modelos de referencia", fontsize=10, color=tinta_suave, va="top")
    eje.grid(axis="y", color="#e8eaed", linewidth=0.8)
    eje.spines[["top", "right"]].set_visible(False)
    eje.legend(loc="lower right", frameon=False)
    figura.tight_layout(rect=(0, 0, 1, 0.88))
    for extension in ("pdf", "png"):
        figura.savefig(RUTA_FIGURAS / f"frontera_encuesta_corta.{extension}", dpi=300, facecolor="white")
    plt.close(figura)


def main(minutos_ga=30):
    reloj = Presupuesto(45)
    registrar_log("etapa3b", "inicio")
    particion = cargar_particion()
    tabla = particion.set_index("id")
    hogar = cargar_hogares()
    bloques = construir_grupos(cargar_personas(), hogar)
    variables = pd.concat([bloques[b] for b in BLOQUES_CANDIDATOS], axis=1)
    matriz = preparar_matriz(variables).astype(np.float64)
    mapa = mapear_columnas(variables, matriz)
    nombres = list(variables.columns)
    verdad = tabla["pobre"]

    # Folds de R: los 5 para la reevaluación; los 3 primeros para la aptitud del GA
    divisiones = []
    for k in range(1, 6):
        ids_ent = tabla.index[tabla["fold_id"].isin([g for g in range(1, 6) if g != k])]
        ids_val = tabla.index[tabla["fold_id"].isin([k])]
        divisiones.append(preparar_division(matriz, ids_ent, ids_val, verdad))
    registrar_log("etapa3b", f"{len(nombres)} variables originales, {matriz.shape[1]} columnas")

    semillas = [np.isin(nombres, VARIABLES_NIVEL_E), np.isin(nombres, VARIABLES_NIVEL_3),
                np.ones(len(nombres), dtype=bool)]
    problema = ProblemaEncuesta(divisiones[:3], mapa)
    algoritmo = NSGA2(pop_size=30, sampling=MuestreoDensidades(semillas),
                      crossover=TwoPointCrossover(), mutation=BitflipMutation(prob=1.0, prob_var=2 / len(nombres)),
                      eliminate_duplicates=True)

    def al_terminar_generacion(algoritmo):
        frontera = frontera_no_dominada(problema.cache)
        resumen = " ".join(f"{n}:{max(f for m, f, _ in frontera if m <= n):.3f}"
                           for n in TAMANOS_REPORTE if any(m <= n for m, _, _ in frontera))
        registrar_log("etapa3b", f"gen {algoritmo.n_gen} | {len(problema.cache)} evaluados | {resumen}")

    minimize(problema, algoritmo, ("time", f"00:{int(minutos_ga):02d}:00"), seed=SEMILLA,
             callback=al_terminar_generacion, verbose=False)
    frontera = frontera_no_dominada(problema.cache)
    registrar_log("etapa3b", f"fin del GA: {len(problema.cache)} evaluados, {len(frontera)} en la frontera")

    # Reevaluación: 5 folds para el umbral y holdout con reajuste sobre el 80%
    ids_80 = tabla.index[tabla["fold_id"].notna() & ~tabla["holdout"]]
    division_holdout = preparar_division(matriz, ids_80, tabla.index[tabla["holdout"]], verdad)
    filas = []
    for n, f1_ga, mascara in frontera + [(int(semillas[0].sum()), np.nan, semillas[0])]:
        if reloj.agotado():
            registrar_log("etapa3b", f"tiempo agotado en n = {n}")
            break
        fila = {"n_variables": n, "f1_ga_3folds": f1_ga,
                "variables": ", ".join(np.array(nombres)[mascara]),
                "es_nivel_e": bool((mascara == semillas[0]).all()),
                **evaluar_en_holdout(mascara, division_holdout, divisiones, mapa)}
        filas.append(fila)
    resultados = pd.DataFrame(filas)
    resultados.to_csv(RUTA_TABLAS / "frontera_encuesta_corta.csv", index=False)

    frontera_tabla = resultados[~resultados["es_nivel_e"]].sort_values("n_variables")
    n7 = errores_mejor_n7(particion)
    graficar_frontera(frontera_tabla, referencias_r(), n7["f1"])

    reporte = []
    for n in TAMANOS_REPORTE:
        punto = frontera_tabla[frontera_tabla["n_variables"] <= n].tail(1).iloc[0]
        reporte.append({"tamano": n, "n_variables": punto["n_variables"], "f1_holdout": punto["f1_holdout"],
                        "subcobertura_n": punto["n_fn_holdout"], "filtracion_n": punto["n_fp_holdout"],
                        "dif_subcobertura_vs_n7": punto["n_fn_holdout"] - n7["n_fn"],
                        "dif_filtracion_vs_n7": punto["n_fp_holdout"] - n7["n_fp"],
                        "variables": punto["variables"]})
    reporte.append({"tamano": "mejor n7", "n_variables": np.nan, "f1_holdout": n7["f1"],
                    "subcobertura_n": n7["n_fn"], "filtracion_n": n7["n_fp"]})
    pd.DataFrame(reporte).to_csv(RUTA_TABLAS / "encuesta_corta_5_10_15.csv", index=False)
    registrar_log("etapa3b", f"fin ({reloj.transcurrido():.1f} min)")
    print(resultados.drop(columns="variables").round(4).to_string(index=False))
    print(pd.DataFrame(reporte).round(4).to_string(index=False))


def regraficar():
    """Rehace la figura desde la tabla guardada, sin volver a correr el GA."""
    resultados = pd.read_csv(RUTA_TABLAS / "frontera_encuesta_corta.csv")
    frontera_tabla = resultados[~resultados["es_nivel_e"]].sort_values("n_variables")
    graficar_frontera(frontera_tabla, referencias_r(), errores_mejor_n7(cargar_particion())["f1"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "graficar":
        regraficar()
    else:
        main(float(sys.argv[1]) if len(sys.argv) > 1 else 30)
