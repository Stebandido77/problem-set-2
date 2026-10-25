"""Etapa 8 (cierre del modelado): tuning de los auxiliares de ingreso y promedio
de modelos.

Base de comparación: n7_a_agregados_bag5 (F1 CV 0,7748, holdout 0,7806). Mismos
folds, holdout, umbral OOF y OOF anidada de los auxiliares.

  1. Tuning de los auxiliares de personas (ingreso laboral y total), que hasta
     ahora usaban parámetros fijos: Optuna (TPE) minimizando el RMSE de
     log(1 + ingreso) por persona, con un estudio por grupo de parámetros
     (cv_1 ... cv_5 y 80 para holdout/test) para que ninguna búsqueda vea el
     fold de validación ni el holdout (funciones/tuneo_auxiliar.py). Luego se
     recalcula la OOF anidada de esos dos auxiliares con los parámetros de cada
     grupo y se reentrena el clasificador de A (n6 + bagging de 5 semillas).
     El auxiliar por fuentes (v3) queda con sus parámetros: tunearlo sin fuga
     exige 38 reajustes de ~2,5 min (~95 min) que no caben en el tope de 2 horas.
  2. Promedio de probabilidades y, aparte, de rangos de A, n7_v3, n7_n6param,
     CatBoost B3 y el modelo del paso 1, con pesos en una grilla del simplex
     (paso 0,1) elegidos por F1 OOF; umbral OOF.
  3. Tabla con F1 CV (sd), holdout, precisión, recall y banda de Lp.
  4. CSV solo si un candidato sube al menos 0,002 en CV y en holdout sobre A;
     a lo sumo uno, el de mayor F1 en holdout entre los que cumplen.

Reanudable (estudios en sqlite, ajustes anidados y folds en parciales/) y con el
vigilante de memoria de la etapa 7. Un solo proceso pesado.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa8_cierre.py
"""

import gc
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from configuracion import PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS
from etapa2_optuna import SEMILLAS_BAGGING
from etapa4_leaderboard import (RUTA_LEADERBOARD_EXTRA, anexar_filas_extra, armar_fila, formatear_como_r,
                                texto_hiperparametros)
from etapa7_exploracion import RUTA_ETAPA7, correr_seguro, fila_tabla
from funciones.anidado import cargar_o_calcular_anidado_por_grupo, crear_perezoso
from funciones.auxiliares import crear_auxiliar_personas
from funciones.bases import BLOQUES_A4, armar_matriz, cargar_insumos
from funciones.bitacora import Presupuesto, iniciar_vigilante_memoria, registrar_log
from funciones.datos import cargar_ingreso_personas, cargar_personas
from funciones.metricas import buscar_umbral_f1
from funciones.modelado import (borrar_parciales, crear_ajuste_lightgbm, crear_preparador, evaluar_modelo,
                                guardar_predicciones, resultado_desde_predicciones)
from funciones.submission import guardar_submission
from funciones.tuneo_auxiliar import (definir_grupos_tuneo, grupo_de_contexto, tunear_auxiliar_personas,
                                      verificar_tuneo_sin_fuga)

BASE = "n7_a_agregados_bag5"
MINUTOS_OPTUNA = 45
RUTA_ETAPA8 = RUTA_NIVEL7 / "etapa8"
ALMACENAMIENTO = f"sqlite:///{(RUTA_ETAPA8 / 'optuna_auxiliares.db').as_posix()}"
# auxiliar: (columna objetivo, sufijo de sus variables, nombre de la caché anidada)
AUXILIARES_TUNEO = {"laboral": ("ingreso_laboral", "", "ingreso_personas_tuneado"),
                    "total": ("ingreso_total", "_total", "ingreso_total_personas_tuneado")}
NOMBRE_PASO_1 = "n8_aux_tuneados_bag5"
CANDIDATOS_PROMEDIO = [BASE, "n7_v3_fuentes_bag5", "n7_n6param_a4_total_bag5", "n7_b3_catboost_agr"]
PASO_PESOS = 0.1
MEJORA_MINIMA = 0.002


# ---- 1. Tuning de los auxiliares ----------------------------------------------

def tunear_auxiliares(personas, ingreso, grupos):
    """Corre (o retoma) un estudio por auxiliar y grupo de parámetros.

    Devuelve: {auxiliar: {grupo: salida de tunear_auxiliar_personas()}}; deja la
        tabla etapa8_tuneo_auxiliares.csv.
    """
    verificar_tuneo_sin_fuga()
    grupos_tuneo = definir_grupos_tuneo()
    minutos = MINUTOS_OPTUNA / (len(AUXILIARES_TUNEO) * len(grupos_tuneo))
    salida, filas = {}, []
    for auxiliar, (columna, _, _) in AUXILIARES_TUNEO.items():
        salida[auxiliar] = {}
        for grupo in grupos_tuneo:
            r = tunear_auxiliar_personas(personas, ingreso, grupos, columna, grupo, minutos, ALMACENAMIENTO)
            salida[auxiliar][grupo] = r
            registrar_log("etapa8_tuneo", f"{auxiliar} {grupo}: RMSE {r['rmse_defecto']:.4f} -> "
                                          f"{r['rmse_tuneado']:.4f} ({r['n_trials']} trials)")
            filas.append({"auxiliar": auxiliar, "grupo_parametros": grupo,
                          "folds_busqueda": ",".join(map(str, r["folds_entrenamiento"])),
                          "fold_validacion_busqueda": r["fold_validacion"], "n_trials": r["n_trials"],
                          "minutos": r["minutos"], "rmse_parametros_actuales": r["rmse_defecto"],
                          "rmse_tuneado": r["rmse_tuneado"],
                          "mejora_rmse_pct": 1 - r["rmse_tuneado"] / r["rmse_defecto"],
                          "hiperparametros": texto_hiperparametros(r["parametros"])})
            gc.collect()
    pd.DataFrame(filas).to_csv(RUTA_TABLAS / "etapa8_tuneo_auxiliares.csv", index=False)
    (RUTA_ETAPA8 / "parametros_auxiliares.json").write_text(
        json.dumps({a: {g: r["parametros"] for g, r in s.items()} for a, s in salida.items()}, indent=1),
        encoding="utf-8")
    return salida


def auxiliares_tuneados(personas, ingreso, hogar, grupos, tuneo):
    """OOF anidada de los auxiliares laboral y total con los parámetros de cada grupo.

    Devuelve: {auxiliar: {contexto: DataFrame}}.
    """
    salida = {}
    for auxiliar, (columna, sufijo, nombre) in AUXILIARES_TUNEO.items():
        def crear_ajuste(grupo, columna=columna, sufijo=sufijo, auxiliar=auxiliar):
            parametros = tuneo[auxiliar][grupo]["parametros"]
            return crear_perezoso(lambda: crear_auxiliar_personas(personas, ingreso, hogar, columna, sufijo,
                                                                  parametros))
        salida[auxiliar] = cargar_o_calcular_anidado_por_grupo(nombre, crear_ajuste, grupos, grupo_de_contexto)
        registrar_log("etapa8_anidado", f"{nombre} listo")
        gc.collect()
    return salida


def paso_1(entorno):
    """Tuning, OOF anidada y clasificador de A con los auxiliares tuneados."""
    insumos = entorno["insumos"]
    personas, ingreso = cargar_personas(), cargar_ingreso_personas()
    tuneo = tunear_auxiliares(personas, ingreso, insumos["grupos"])
    tuneados = auxiliares_tuneados(personas, ingreso, insumos["hogar"], insumos["grupos"], tuneo)
    del personas, ingreso
    gc.collect()
    if (RUTA_NIVEL7 / f"predicciones_{NOMBRE_PASO_1}.parquet").exists():
        return resultado_desde_predicciones(NOMBRE_PASO_1, entorno["verdad"])
    dinamicas = [tuneados["laboral"], tuneados["total"], insumos["auxiliares"]["fuentes"]]
    preparar = crear_preparador(entorno["matriz_a"], entorno["particion"], dinamicas)
    return evaluar_modelo(NOMBRE_PASO_1, preparar, crear_ajuste_lightgbm(PARAMETROS_N6, SEMILLAS_BAGGING),
                          entorno["particion"], con_test=True, reanudable=True)


# ---- 2. Promedio de modelos ---------------------------------------------------

def grilla_simplex(n, paso):
    """Pesos no negativos que suman 1, en múltiplos de paso (n modelos)."""
    k = int(round(1 / paso))
    return [np.array(c) / k for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]


def a_rangos(probabilidad):
    """Rango normalizado a (0, 1] dentro de un conjunto de predicciones."""
    return pd.Series(rankdata(probabilidad) / len(probabilidad), index=probabilidad.index)


def promediar(nombre, resultados, verdad, rangos=False):
    """Promedio ponderado de candidatos con los pesos que maximizan el F1 OOF.

    Pesos y umbral salen solo de la OOF; holdout y test reciben la misma mezcla.
    Con rangos=True cada conjunto (OOF, holdout, test) se pasa a rangos antes de
    promediar, así que el umbral es un cuantil.
    Argumentos: nombre = candidato; resultados = {nombre: resultado_desde_predicciones()};
        verdad = pobre por id; rangos = promedio de rangos en vez de probabilidades.
    Devuelve: (resultado, pesos por candidato).
    """
    nombres = list(resultados)
    indice = resultados[nombres[0]]["oof"].index

    def matriz(tipo):
        if tipo == "oof":
            columnas = {n: resultados[n]["oof"]["probabilidad"].reindex(indice) for n in nombres}
        else:
            columnas = {n: resultados[n][tipo].reindex(resultados[nombres[0]][tipo].index) for n in nombres}
        tabla = pd.DataFrame(columnas)
        return tabla.apply(a_rangos) if rangos else tabla

    oof = matriz("oof")
    verdad_oof = verdad.loc[indice].astype(int)
    evaluados = [(w, buscar_umbral_f1(verdad_oof, oof.to_numpy() @ w)[1]) for w in grilla_simplex(len(nombres),
                                                                                               PASO_PESOS)]
    pesos, f1_oof = max(evaluados, key=lambda x: x[1])
    resultado = {"nombre": nombre,
                 "oof": pd.DataFrame({"probabilidad": oof @ pesos,
                                      "fold": resultados[nombres[0]]["oof"]["fold"].reindex(indice)}),
                 "prediccion_holdout": matriz("prediccion_holdout") @ pesos}
    if all("prediccion_test" in r for r in resultados.values()):
        resultado["prediccion_test"] = matriz("prediccion_test") @ pesos
    guardar_predicciones(resultado)
    pesos = dict(zip(nombres, map(float, pesos)))
    registrar_log(nombre, f"pesos {pesos} (F1 OOF {f1_oof:.4f}, {len(evaluados)} combinaciones)")
    return resultado_desde_predicciones(nombre, verdad), pesos


# ---- 3 y 4. Tabla, CSV y leaderboard ------------------------------------------

def cerrar(entorno, resultados, descripciones):
    """Tabla de candidatos, CSV del mejor que cumple la regla y leaderboard."""
    verdad, base = entorno["verdad"], resultados[BASE]
    filas = []
    for nombre, resultado in resultados.items():
        fila = fila_tabla(resultado, verdad, descripciones[nombre]["descripcion"], base)
        fila["cumple_regla_csv"] = bool(fila["delta_cv"] >= MEJORA_MINIMA and fila["delta_holdout"] >= MEJORA_MINIMA)
        filas.append(fila)
    tabla = pd.DataFrame(filas).drop(columns=["supera_base"])
    tabla.to_csv(RUTA_TABLAS / "etapa8_candidatos.csv", index=False)

    cumplen = tabla[tabla["cumple_regla_csv"]].sort_values("f1_holdout", ascending=False)
    elegido = cumplen["candidato"].iloc[0] if len(cumplen) else None
    registrar_log("etapa8", f"candidatos que cumplen la regla: {list(cumplen['candidato'])} -> CSV: {elegido}")
    ids_test = entorno["particion"].loc[entorno["particion"]["base"] == "test", "id"]

    filas_lb = []
    for nombre, descripcion in descripciones.items():
        if not nombre.startswith("n8_"):
            continue
        archivo = None
        if nombre == elegido:
            ruta, tasa = guardar_submission(resultados[nombre], f"{nombre}.csv", ids_test,
                                            descripcion["especificacion"] + " | umbral OOF", algoritmo="Boosting")
            archivo = f"nivel7/{ruta.name}"
            registrar_log("etapa8", f"submission {ruta.name} tasa {tasa:.4f}")
        filas_lb.append(armar_fila(nombre, "Boosting", descripcion["especificacion"], entorno["variables_a"],
                                   descripcion["busqueda"], descripcion["n_configuraciones"],
                                   descripcion["hiperparametros"], descripcion.get("tiempo_min", pd.NA),
                                   archivo, verdad))
    # Se agregan a las filas de la etapa 7 (no las reemplazan)
    extra = pd.read_csv(RUTA_LEADERBOARD_EXTRA, dtype=str, keep_default_na=False)
    nuevas = formatear_como_r(pd.DataFrame(filas_lb)).astype(str).replace({"<NA>": "NA", "nan": "NA"})
    extra = pd.concat([extra[~extra["candidato"].isin(nuevas["candidato"])], nuevas[extra.columns]],
                      ignore_index=True)
    extra.to_csv(RUTA_LEADERBOARD_EXTRA, index=False)
    anexar_filas_extra()
    return tabla, elegido


def main():
    RUTA_ETAPA8.mkdir(parents=True, exist_ok=True)
    reloj = Presupuesto(115)
    registrar_log("etapa8", "inicio")
    iniciar_vigilante_memoria("etapa8")
    insumos = cargar_insumos(auxiliar_total=True, auxiliar_fuentes=True)
    particion = insumos["particion"]
    verdad = particion.set_index("id")["pobre"]
    agregados = pd.read_parquet(RUTA_ETAPA7 / "agregados_personas.parquet")
    entorno = {"insumos": insumos, "particion": particion, "verdad": verdad,
               "matriz_a": armar_matriz(insumos).join(agregados),
               "variables_a": [c for b in BLOQUES_A4 for c in insumos["bloques"][b].columns]
                              + ["aux_ingreso_laboral", "aux_ingreso_total", "aux_ingreso_fuentes"]
                              + list(agregados.columns)}
    del agregados

    resultados = {n: resultado_desde_predicciones(n, verdad) for n in CANDIDATOS_PROMEDIO}
    descripciones = {
        BASE: {"descripcion": "base: A (n7_v3 + 1026 agregados por rol)"},
        "n7_v3_fuentes_bag5": {"descripcion": "n7_v3 (a4 + auxiliares laboral, total y fuentes)"},
        "n7_n6param_a4_total_bag5": {"descripcion": "n7_final (a4 + auxiliares laboral y total)"},
        "n7_b3_catboost_agr": {"descripcion": "CatBoost B3 sobre la base de A"},
    }

    paso1 = correr_seguro("etapa8", "paso 1", lambda: paso_1(entorno))
    if paso1 is not None:
        borrar_parciales(NOMBRE_PASO_1)
        resultados[NOMBRE_PASO_1] = resultado_desde_predicciones(NOMBRE_PASO_1, verdad)
        tuneo = pd.read_csv(RUTA_TABLAS / "etapa8_tuneo_auxiliares.csv")
        descripciones[NOMBRE_PASO_1] = {
            "descripcion": "A con los auxiliares laboral y total tuneados (Optuna por grupo, sin fuga)",
            "especificacion": "LightGBM (n6, bagging 5 semillas) | base de A con los auxiliares de ingreso "
                              "laboral y total tuneados con Optuna (un estudio por fold, sin fuga)",
            "busqueda": (f"auxiliares: Optuna TPE por tiempo ({MINUTOS_OPTUNA} min en 12 estudios), RMSE de "
                         "log(1 + ingreso) en un fold interno; learning_rate [0.03, 0.2] log, num_leaves "
                         "[15, 255] log, min_data_in_leaf [20, 1000] log, feature_fraction [0.4, 1], "
                         "bagging_fraction [0.5, 1], lambda_l2 [1e-3, 10] log; árboles por parada temprana; "
                         "clasificador con hiperparámetros de n6 fijos"),
            "n_configuraciones": int(tuneo["n_trials"].sum()),
            "hiperparametros": texto_hiperparametros(PARAMETROS_N6)}
    registrar_log("etapa8", f"paso 1 {'listo' if paso1 is not None else 'falló'} ({reloj.transcurrido():.1f} min)")

    base_promedio = dict(resultados)
    for nombre, rangos in (("n8_promedio_probabilidades", False), ("n8_promedio_rangos", True)):
        salida = correr_seguro("etapa8", nombre, lambda: promediar(nombre, base_promedio, verdad, rangos))
        if salida is None:
            continue
        resultados[nombre], pesos = salida
        texto_pesos = ", ".join(f"{n} {w:.1f}" for n, w in pesos.items() if w > 0)
        tipo = "rangos" if rangos else "probabilidades"
        descripciones[nombre] = {
            "descripcion": f"promedio de {tipo} ({texto_pesos})",
            "especificacion": f"Promedio ponderado de {tipo} de modelos LightGBM y CatBoost ({texto_pesos})",
            "busqueda": f"pesos en la grilla del simplex (paso {PASO_PESOS}) por F1 OOF; umbral OOF",
            "n_configuraciones": len(grilla_simplex(len(base_promedio), PASO_PESOS)),
            "hiperparametros": "; ".join(f"peso_{n} = {w:.1f}" for n, w in pesos.items())}

    tabla, elegido = cerrar(entorno, resultados, descripciones)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tabla[["candidato", "f1_cv", "f1_cv_sd", "f1_holdout", "precision_holdout", "recall_holdout",
                     "f1_banda_08_12", "delta_cv", "delta_holdout", "cumple_regla_csv"]].round(4))
    registrar_log("etapa8", f"fin: CSV {elegido} ({reloj.transcurrido():.1f} min)")


if __name__ == "__main__":
    main()
