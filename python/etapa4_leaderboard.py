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
from funciones.metricas import buscar_umbral_f1, calcular_curva_umbral, calcular_metricas, metricas_por_fold
from funciones.modelado import en_grupos, ajustar_lightgbm, armar_contexto
from funciones.submission import RUTA_BITACORA, validar_submission
from funciones.variables import construir_grupos

ETIQUETA_NIVEL = "7. Python: v2 + auxiliares de ingreso"
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


def fila_leaderboard(nombre, especificacion, bloques, verdad):
    """Fila del leaderboard con el esquema de R, desde las predicciones guardadas."""
    nombres_bloques, auxiliares, tipo_parametros, semillas, archivo = especificacion
    predicciones = pd.read_parquet(RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").set_index("id")
    oof = predicciones[predicciones["tipo"] == "oof"]
    holdout = predicciones[predicciones["tipo"] == "holdout"]
    umbral, _ = buscar_umbral_f1(verdad.loc[oof.index], oof["probabilidad"])
    cv = metricas_por_fold(verdad.loc[oof.index], oof["probabilidad"], oof["fold"], umbral)
    h = calcular_metricas(verdad.loc[holdout.index].astype(int), (holdout["probabilidad"] >= umbral).astype(int))

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
    hiperparametros = "; ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                                for k, v in parametros.items() if k not in ("objective", "feature_pre_filter"))
    descripcion = (f"Boosting (lightgbm, Python) | {ETIQUETA_NIVEL} ({len(variables)} variables) | "
                   f"bloques: {' + '.join(nombres_bloques)}"
                   + (f" | auxiliares: {', '.join(auxiliares)}" if auxiliares else "")
                   + (f" | seed bagging {semillas} semillas" if semillas > 1 else "")
                   + (" | perceptor_oficio nativo" if nombre == "n7_v1_oficio_nativo" else ""))
    return {
        "nivel": "7", "etiqueta_nivel": ETIQUETA_NIVEL, "candidato": nombre, "algoritmo": "Boosting",
        "especificacion": descripcion, "n_variables": len(variables), "variables": ", ".join(variables),
        "balanceo": "ninguno", "mecanismo_pesos": "sin_pesos", "pesos_aplicados": pd.NA,
        "busqueda": busqueda, "n_configuraciones": n_configuraciones, "hiperparametros": hiperparametros,
        "umbral": umbral, "f1_cv": cv["f1_cv"], "f1_cv_sd": cv["f1_cv_sd"],
        "precision_cv": cv["precision_cv"], "recall_cv": cv["recall_cv"], "accuracy_cv": cv["accuracy_cv"],
        "f1_holdout": h["f1"], "precision_holdout": h["precision"], "recall_holdout": h["recall"],
        "accuracy_holdout": h["accuracy"], "filtracion_pct": h["filtracion"], "filtracion_n": h["n_fp"],
        "subcobertura_pct": h["subcobertura"], "subcobertura_n": h["n_fn"],
        "pobres_holdout": h["n_vp"] + h["n_fn"], "predichos_pobres_holdout": h["n_vp"] + h["n_fp"],
        "brecha_cv_holdout": cv["f1_cv"] - h["f1"],
        "sobreajuste": (cv["f1_cv"] - h["f1"]) > UMBRAL_SOBREAJUSTE,
        "tiempo_min": tiempos_registrados().get(nombre, pd.NA),
        "archivo_submission": archivo if archivo else pd.NA,
    }


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
    # Mismo formato que write_csv() de R: lógicos en mayúscula y faltantes como NA
    nivel7["sobreajuste"] = nivel7["sobreajuste"].map({True: "TRUE", False: "FALSE"})
    nivel7.to_csv(RUTA_RESULTADOS / "leaderboard_nivel7.csv", index=False, na_rep="NA")

    ruta = RUTA_RESULTADOS / "leaderboard_interno.csv"
    existente = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    assert list(existente.columns) == list(nivel7.columns), "el esquema del leaderboard cambió"
    existente = existente[~existente["candidato"].isin(nivel7["candidato"])]  # solo reemplaza filas n7
    nuevas = pd.read_csv(RUTA_RESULTADOS / "leaderboard_nivel7.csv", dtype=str, keep_default_na=False)
    pd.concat([existente, nuevas], ignore_index=True).to_csv(ruta, index=False)

    # F1 público anotado en la bitácora (donde 09/10_*.R lo buscan)
    bitacora = pd.read_csv(RUTA_BITACORA, dtype=str, keep_default_na=False)
    for archivo, f1 in F1_KAGGLE.items():
        bitacora.loc[bitacora["archivo"] == archivo, "f1_kaggle"] = f1
    bitacora.to_csv(RUTA_BITACORA, index=False)

    # Solo CSV finales en stores/submissions/nivel7/, todos revalidados
    ids_test = particion.loc[particion["base"] == "test", "id"]
    finales = {Path(a).name for *_, a in CANDIDATOS.values() if a}
    for csv in sorted(RUTA_SUBMISSIONS.glob("*.csv")):
        assert csv.name in finales, f"{csv.name} no es un CSV final"
        tasa = validar_submission(pd.read_csv(csv, dtype={"id": str}), ids_test)
        print(f"{csv.name}: válido, tasa de pobres {tasa:.4f}")
    print(nivel7[["candidato", "n_variables", "umbral", "f1_cv", "f1_cv_sd", "f1_holdout",
                  "filtracion_n", "subcobertura_n"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
