"""Etapa 4: agrega los candidatos del nivel 7 a stores/resultados/leaderboard_interno.csv
(mismas columnas que fila_leaderboard() en R), anota el F1 de Kaggle del
checkpoint en la bitácora y revalida los CSV finales de stores/submissions/nivel7/.

Las métricas se recalculan desde las predicciones guardadas en
stores/procesados/nivel7/ con el mismo protocolo de R (umbral OOF, holdout).
Las filas existentes no se tocan; las del nivel 7 se reemplazan si ya estaban.
Copia aparte en stores/resultados/leaderboard_nivel7.csv, que 08_leaderboard.R
vuelve a unir al leaderboard. Además exporta, para el mejor n7 con submission
(mayor F1 holdout), la importancia de variables y la curva de umbral que
10_mejor_modelo.R reporta sin readRDS.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa4_leaderboard.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import PARAMETROS_N6, RUTA_NIVEL7, RUTA_RESULTADOS, RUTA_SUBMISSIONS, RUTA_TABLAS
from etapa2_optuna import SEMILLAS_BAGGING
from funciones.anidado import definir_contextos
from funciones.bases import armar_matriz, cargar_insumos
from funciones.datos import cargar_hogares, cargar_particion, cargar_personas, imputar_mediana
from funciones.metricas import calcular_curva_umbral
from funciones.modelado import en_grupos, ajustar_lightgbm, armar_contexto, resultado_desde_predicciones
from funciones.submission import RUTA_BITACORA, validar_submission
from funciones.variables import construir_grupos

ETIQUETA_NIVEL = "7. Python: v2 + auxiliares de ingreso"
# Filas que agregan otras etapas (etapa 7 en adelante) sin pasar por CANDIDATOS
RUTA_LEADERBOARD_EXTRA = RUTA_RESULTADOS / "leaderboard_etapa7.csv"
UMBRAL_SOBREAJUSTE = 0.02
F1_KAGGLE = {"nivel7/checkpoint_a4.csv": "0.77"}
AUXILIARES = {"laboral": 4, "total": 4, "hogar": 1, "fuentes": 10}  # columnas que aporta cada auxiliar

# candidato: (bloques, auxiliares, parámetros, semillas, archivo de submission)
CANDIDATOS = {
    "n7_a0_nivel3": (["nivel_3"], [], "n6", 1, None),
    "n7_a1_laborales": (["nivel_3", "laborales"], [], "n6", 1, None),
    "n7_a2_perceptor": (["nivel_3", "laborales", "perceptor"], [], "n6", 1, None),
    "n7_a3_pequenos": (["nivel_3", "laborales", "perceptor", "pequenos"], [], "n6", 1, None),
    "n7_a4_aux_personas": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral"], "n6", 1, None),
    "n7_a5_aux_hogar": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral", "hogar"], "n6", 1, None),
    "n7_a4_aux_personas_fex": (["nivel_3", "laborales", "perceptor", "pequenos", "fex"], ["laboral"], "n6", 1, None),
    "n7_a4_checkpoint": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral"], "n6", 1, "nivel7/checkpoint_a4.csv"),
    "n7_v1_oficio_nativo": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral"], "n6", 1, None),
    "n7_v2_total_reemplaza": (["nivel_3", "laborales", "perceptor", "pequenos"], ["total"], "n6", 1, None),
    "n7_v2_total_suma": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral", "total"], "n6", 1, None),
    "n7_optuna_a4_total": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral", "total"], "optuna", 1, None),
    "n7_optuna_a4_total_bag5": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral", "total"], "optuna", 5, None),
    "n7_n6param_a4_total_bag5": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral", "total"], "n6", 5, "nivel7/n7_final.csv"),
    "n7_v3_solo_fuentes_bag5": (["nivel_3", "laborales", "perceptor", "pequenos"], ["fuentes"], "n6", 5, None),
    "n7_v3_fuentes_bag5": (["nivel_3", "laborales", "perceptor", "pequenos"], ["laboral", "total", "fuentes"], "n6", 5, "nivel7/n7_v3_final.csv"),
}

# Mejor n7 que exporta exportar_mejor_n7() para 10_mejor_modelo.R, elegido a mano.
# n7_v3_fuentes_bag5 tiene más F1 en holdout, pero su auxiliar de fuentes no se
# carga en cargar_insumos() y el deep dive sigue describiendo n7_final hasta que
# el equipo decida el cambio. Con None se aplica la regla automática (mayor F1
# en holdout entre los que tienen submission).
MEJOR_N7_MANUAL = "n7_n6param_a4_total_bag5"


def tiempos_registrados():
    """Minutos por candidato desde las tablas de cada etapa (si existen)."""
    tiempos = {}
    for archivo in ("ablacion_etapa1.csv", "optuna_bagging_a4_total.csv", "bagging_n6param_a4_total.csv",
                    "etapa6_ablacion.csv"):
        ruta = RUTA_TABLAS / archivo
        if ruta.exists():
            tabla = pd.read_csv(ruta)
            tiempos.update(dict(zip(tabla["candidato"], tabla["tiempo_min"])))
    return tiempos


def campos_metricas(nombre, verdad):
    """Métricas de un candidato con el esquema del leaderboard de R, desde sus
    predicciones guardadas (umbral OOF, CV por fold y holdout).

    Argumentos: nombre = candidato; verdad = pobre por id.
    Devuelve: diccionario en el orden de columnas del leaderboard (umbral ... sobreajuste).
    """
    r = resultado_desde_predicciones(nombre, verdad)
    return {
        "umbral": r["umbral"], "f1_cv": r["f1_cv"], "f1_cv_sd": r["f1_cv_sd"],
        "precision_cv": r["precision_cv"], "recall_cv": r["recall_cv"], "accuracy_cv": r["accuracy_cv"],
        "f1_holdout": r["f1_holdout"], "precision_holdout": r["precision_holdout"],
        "recall_holdout": r["recall_holdout"], "accuracy_holdout": r["accuracy_holdout"],
        "filtracion_pct": r["filtracion_holdout"], "filtracion_n": r["n_fp_holdout"],
        "subcobertura_pct": r["subcobertura_holdout"], "subcobertura_n": r["n_fn_holdout"],
        "pobres_holdout": r["n_vp_holdout"] + r["n_fn_holdout"],
        "predichos_pobres_holdout": r["n_vp_holdout"] + r["n_fp_holdout"],
        "brecha_cv_holdout": r["f1_cv"] - r["f1_holdout"],
        "sobreajuste": (r["f1_cv"] - r["f1_holdout"]) > UMBRAL_SOBREAJUSTE,
    }


def armar_fila(nombre, algoritmo, especificacion, variables, busqueda, n_configuraciones,
               hiperparametros, tiempo_min, archivo, verdad):
    """Fila completa del leaderboard (esquema de R) de un candidato del nivel 7.

    Argumentos: nombre = candidato; algoritmo = nombre en la lista del curso;
        especificacion = descripción; variables = lista de predictores;
        busqueda, n_configuraciones, hiperparametros = cómo se eligieron;
        tiempo_min = minutos (o NA); archivo = CSV de submission (o None);
        verdad = pobre por id.
    Devuelve: diccionario con todas las columnas del leaderboard.
    """
    return {
        "nivel": "7", "etiqueta_nivel": ETIQUETA_NIVEL, "candidato": nombre, "algoritmo": algoritmo,
        "especificacion": especificacion, "n_variables": len(variables), "variables": ", ".join(variables),
        "balanceo": "ninguno", "mecanismo_pesos": "sin_pesos", "pesos_aplicados": pd.NA,
        "busqueda": busqueda, "n_configuraciones": n_configuraciones, "hiperparametros": hiperparametros,
        **campos_metricas(nombre, verdad),
        "tiempo_min": tiempo_min, "archivo_submission": archivo if archivo else pd.NA,
    }


def fila_leaderboard(nombre, especificacion, bloques, verdad):
    """Fila del leaderboard de un candidato de CANDIDATOS (LightGBM)."""
    nombres_bloques, auxiliares, tipo_parametros, semillas, archivo = especificacion
    variables = [c for b in nombres_bloques for c in bloques[b].columns]
    variables += [f"aux_ingreso_{a}" for a in auxiliares]
    parametros = leer_parametros(tipo_parametros)
    if tipo_parametros == "n6":
        busqueda = "hiperparámetros de n6_lightgbm fijos"
        n_configuraciones = 1
    else:
        estudio = pd.read_csv(RUTA_TABLAS / "optuna_etapa2_a4_total.csv")
        busqueda = ("Optuna TPE por tiempo (40 min): learning_rate [0.01, 0.15] log, num_leaves [15, 255] log, "
                    "max_depth [4, 16], min_child_samples [20, 500] log, feature_fraction [0.3, 1], "
                    "bagging_fraction [0.5, 1], lambda_l1/l2 [1e-8, 10] log; early stopping en 10% interno")
        n_configuraciones = len(estudio)
    descripcion = (f"Boosting (lightgbm, Python) | {ETIQUETA_NIVEL} ({len(variables)} variables) | "
                   f"bloques: {' + '.join(nombres_bloques)}"
                   + (f" | auxiliares: {', '.join(auxiliares)}" if auxiliares else "")
                   + (f" | seed bagging {semillas} semillas" if semillas > 1 else "")
                   + (" | perceptor_oficio nativo" if nombre == "n7_v1_oficio_nativo" else ""))
    return armar_fila(nombre, "Boosting", descripcion, variables, busqueda, n_configuraciones,
                      texto_hiperparametros(parametros), tiempos_registrados().get(nombre, pd.NA), archivo, verdad)


def texto_hiperparametros(parametros):
    """Hiperparámetros como texto 'nombre = valor; ...' para el leaderboard."""
    return "; ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                     for k, v in parametros.items() if k not in ("objective", "feature_pre_filter"))


def formatear_como_r(filas):
    """Mismo formato que write_csv() de R: lógicos en mayúscula y faltantes como NA.

    Argumentos: filas = DataFrame del leaderboard.
    Devuelve: copia con sobreajuste como TRUE/FALSE.
    """
    filas = filas.copy()
    filas["sobreajuste"] = filas["sobreajuste"].map({True: "TRUE", False: "FALSE"})
    return filas


def anexar_filas_extra():
    """Agrega a leaderboard_nivel7.csv las filas de leaderboard_etapa7.csv (si
    existe), reemplazando las del mismo candidato, y vuelve a unir el leaderboard
    interno. No toca ninguna otra fila.
    """
    ruta_nivel7 = RUTA_RESULTADOS / "leaderboard_nivel7.csv"
    nivel7 = pd.read_csv(ruta_nivel7, dtype=str, keep_default_na=False)
    if RUTA_LEADERBOARD_EXTRA.exists():
        extra = pd.read_csv(RUTA_LEADERBOARD_EXTRA, dtype=str, keep_default_na=False)
        assert list(extra.columns) == list(nivel7.columns), "el esquema de leaderboard_etapa7.csv cambió"
        nivel7 = pd.concat([nivel7[~nivel7["candidato"].isin(extra["candidato"])], extra], ignore_index=True)
        nivel7.to_csv(ruta_nivel7, index=False)

    ruta = RUTA_RESULTADOS / "leaderboard_interno.csv"
    existente = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    assert list(existente.columns) == list(nivel7.columns), "el esquema del leaderboard cambió"
    existente = existente[~existente["candidato"].isin(nivel7["candidato"])]  # solo reemplaza filas n7
    pd.concat([existente, nivel7], ignore_index=True).to_csv(ruta, index=False)


def archivos_finales():
    """Nombres de los CSV finales del nivel 7: los de CANDIDATOS y los de leaderboard_etapa7.csv."""
    finales = {Path(a).name for *_, a in CANDIDATOS.values() if a}
    if RUTA_LEADERBOARD_EXTRA.exists():
        extra = pd.read_csv(RUTA_LEADERBOARD_EXTRA, dtype=str, keep_default_na=False)
        finales |= {Path(a).name for a in extra["archivo_submission"] if a not in ("", "NA")}
    return finales


def leer_parametros(tipo_parametros):
    """Hiperparámetros de LightGBM de un candidato ("n6" u "optuna")."""
    if tipo_parametros == "n6":
        return PARAMETROS_N6
    return json.loads((RUTA_NIVEL7 / "parametros_optuna_a4_total.json").read_text())


def exportar_mejor_n7(nivel7, verdad):
    """Importancia (gain promedio de las semillas, ajuste sobre el 80%) y curva de
    umbral OOF del mejor n7 con submission (MEJOR_N7_MANUAL si no es None), para
    10_mejor_modelo.R.

    Argumentos: nivel7 = filas del leaderboard del nivel 7; verdad = pobre por id.
    Devuelve: nombre del candidato exportado.
    """
    con_submission = nivel7[nivel7["archivo_submission"].notna()]
    nombre = MEJOR_N7_MANUAL or con_submission.sort_values("f1_holdout", ascending=False)["candidato"].iloc[0]
    nombres_bloques, auxiliares, tipo_parametros, semillas, _ = CANDIDATOS[nombre]

    predicciones = pd.read_parquet(RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").set_index("id")
    oof = predicciones[predicciones["tipo"] == "oof"]
    curva = calcular_curva_umbral(verdad.loc[oof.index].astype(int), oof["probabilidad"])
    curva.insert(0, "candidato", nombre)
    curva.to_csv(RUTA_TABLAS / "mejor_n7_curva_umbral.csv", index=False)

    insumos = cargar_insumos(auxiliar_total="total" in auxiliares)
    tabla = insumos["particion"].set_index("id")
    entrenamiento, _ = definir_contextos()["holdout"]
    ids = tabla.index[en_grupos(tabla, entrenamiento)]
    dinamicas = [insumos["auxiliares"][a] for a in auxiliares]
    x = armar_contexto(armar_matriz(insumos, nombres_bloques), dinamicas, "holdout", ids)
    (x,) = imputar_mediana(x)
    ganancia = 0
    for semilla in SEMILLAS_BAGGING[:semillas]:
        modelo = ajustar_lightgbm(x, verdad.loc[ids].astype(int).to_numpy(), leer_parametros(tipo_parametros), semilla)
        ganancia = ganancia + pd.Series(modelo.feature_importance("gain"), index=modelo.feature_name())
    importancia = (ganancia / ganancia.max()).sort_values(ascending=False)
    pd.DataFrame({"candidato": nombre, "variable": importancia.index, "importancia": importancia.to_numpy()})         .to_csv(RUTA_TABLAS / "mejor_n7_importancia.csv", index=False)
    return nombre


def main():
    particion = cargar_particion()
    verdad = particion.set_index("id")["pobre"]
    bloques = construir_grupos(cargar_personas(), cargar_hogares())

    nivel7 = pd.DataFrame([fila_leaderboard(n, e, bloques, verdad) for n, e in CANDIDATOS.items()])
    nivel7 = nivel7.sort_values("f1_holdout", ascending=False)
    print(f"Mejor n7 con submission exportado para 10_mejor_modelo.R: {exportar_mejor_n7(nivel7, verdad)}")
    formatear_como_r(nivel7).to_csv(RUTA_RESULTADOS / "leaderboard_nivel7.csv", index=False, na_rep="NA")
    anexar_filas_extra()

    # F1 público anotado en la bitácora (donde 09/10_*.R lo buscan)
    bitacora = pd.read_csv(RUTA_BITACORA, dtype=str, keep_default_na=False)
    for archivo, f1 in F1_KAGGLE.items():
        bitacora.loc[bitacora["archivo"] == archivo, "f1_kaggle"] = f1
    bitacora.to_csv(RUTA_BITACORA, index=False)

    # Solo CSV finales en stores/submissions/nivel7/, todos revalidados
    ids_test = particion.loc[particion["base"] == "test", "id"]
    finales = archivos_finales()
    for csv in sorted(RUTA_SUBMISSIONS.glob("*.csv")):
        assert csv.name in finales, f"{csv.name} no es un CSV final"
        tasa = validar_submission(pd.read_csv(csv, dtype={"id": str}), ids_test)
        print(f"{csv.name}: válido, tasa de pobres {tasa:.4f}")
    print(nivel7[["candidato", "n_variables", "umbral", "f1_cv", "f1_cv_sd", "f1_holdout",
                  "filtracion_n", "subcobertura_n"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
