"""Etapa 11: el estrato como objetivo auxiliar y la reconciliación del arriendo
imputado en los auxiliares de ingreso.

Base de comparación de 1 semilla: n7_v2_total_suma (a4 + auxiliares de ingreso
laboral y total, parámetros de n6, semilla 2026; F1 CV 0,7693, holdout 0,7776).

  1. Auxiliar de estrato: un LightGBM multiclase sobre los predictores de hogar de
     a4 (presentes en test) predice Estrato1 (1 a 6, solo train) con la misma OOF
     anidada de los auxiliares de ingreso (verificar_sin_fuga()). Al clasificador
     entran el estrato esperado y P(estrato <= 2).
  2. n11_estrato_1s: la base + las dos variables de estrato, 1 semilla. Si sube
     más de MEJORA_MINIMA en CV y en holdout frente a la base, se corre
     n11_a_estrato_bag5: la base de A (1.324 columnas, matrices de la etapa 10) +
     estrato, bagging de 5 semillas y test; genera el CSV solo si supera a
     n7_a_agregados_bag5 en CV y en holdout.
  3. n11_arriendo_1s: la base con el per cápita de los auxiliares reconciliado con
     el arriendo imputado del DANE (derivar_per_capita(arriendo_imputado=True)),
     1 semilla. La corrección se adopta por defecto solo si no baja ni el F1 CV ni
     el F1 holdout.

Tablas en views/tablas/nivel7/: etapa11_candidatos.csv, etapa11_estrato_auxiliar.csv
y etapa11_arriendo_reconciliacion.csv.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa11_estrato.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from configuracion import PARAMETROS_N6, RUTA_NIVEL7, RUTA_PROCESADOS, RUTA_TABLAS
from etapa2_optuna import SEMILLAS_BAGGING
from funciones.anidado import GRUPO_HOLDOUT, GRUPOS_80
from funciones.auxiliares import ESTRATO_BAJO_MAXIMO, arriendo_imputado_hogar
from funciones.bases import armar_matriz, cargar_insumos, reconciliar_auxiliares
from funciones.bitacora import Presupuesto, iniciar_vigilante_memoria, registrar_log
from funciones.datos import cargar_estrato_hogar
from funciones.modelado import (borrar_parciales, crear_ajuste_lightgbm, evaluar_candidato, evaluar_modelo,
                                resultado_desde_predicciones)
from funciones.submission import guardar_submission

CONTROL = "n7_v2_total_suma"
BASE_A = "n7_a_agregados_bag5"
CANDIDATO_ESTRATO = "n11_estrato_1s"
CANDIDATO_ARRIENDO = "n11_arriendo_1s"
CANDIDATO_BAG5 = "n11_a_estrato_bag5"
# Mejora mínima en F1 CV y holdout (sobre la base de 1 semilla) para pasar al bagging sobre A
MEJORA_MINIMA = 0.003
RUTA_MATRICES_A = RUTA_NIVEL7 / "etapa10" / "matrices"
ARCHIVO_TIEMPOS = RUTA_NIVEL7 / "etapa11_tiempos.json"


def evaluar_o_cargar(nombre, evaluar, verdad):
    """Resultado desde las predicciones guardadas si el candidato ya terminó; si
    no, lo evalúa y anota su tiempo.

    Argumentos: nombre = candidato; evaluar = función sin argumentos que corre
        evaluar_candidato()/evaluar_modelo(); verdad = pobre por id.
    Devuelve: diccionario de resultado con tiempo_min.
    """
    tiempos = pd.read_json(ARCHIVO_TIEMPOS, typ="series") if ARCHIVO_TIEMPOS.exists() else pd.Series(dtype=float)
    if (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
        resultado = resultado_desde_predicciones(nombre, verdad)
        resultado["tiempo_min"] = float(tiempos.get(nombre, np.nan))
        return resultado
    resultado = evaluar()
    tiempos[nombre] = resultado["tiempo_min"]
    tiempos.to_json(ARCHIVO_TIEMPOS, indent=2)
    return resultado


def fila_tabla(resultado, descripcion, base=None):
    """Fila de la tabla de candidatos con las diferencias frente a una base."""
    fila = {"candidato": resultado["nombre"], "descripcion": descripcion, "umbral": resultado["umbral"],
            "f1_cv": resultado["f1_cv"], "f1_cv_sd": resultado.get("f1_cv_sd", np.nan),
            "f1_holdout": resultado["f1_holdout"], "precision_holdout": resultado["precision_holdout"],
            "recall_holdout": resultado["recall_holdout"], "tiempo_min": resultado.get("tiempo_min", np.nan)}
    if base is not None:
        fila["base"] = base["nombre"]
        fila["delta_cv"] = resultado["f1_cv"] - base["f1_cv"]
        fila["delta_holdout"] = resultado["f1_holdout"] - base["f1_holdout"]
    return fila


def calidad_auxiliar_estrato(aux, grupos, estrato, verdad):
    """Qué tan bien predice el auxiliar el estrato y la pobreza, fuera de muestra.

    OOF del 80%: en cada contexto cv_k, las filas del fold k (predichas por un
    auxiliar que no las vio); holdout: las filas del holdout en el contexto holdout.
    Argumentos: aux = {contexto: DataFrame} del auxiliar; grupos = asignar_grupos();
        estrato = cargar_estrato_hogar(); verdad = pobre por id.
    Devuelve: DataFrame con una fila por conjunto.
    """
    partes = {"oof_80": pd.concat([aux[f"cv_{k}"].reindex(grupos.index[grupos == k]) for k in GRUPOS_80]),
              "holdout": aux["holdout"].reindex(grupos.index[grupos == GRUPO_HOLDOUT])}
    filas = []
    for conjunto, tabla in partes.items():
        y_estrato = estrato.reindex(tabla.index)
        y_pobre = verdad.reindex(tabla.index).astype(int)
        filas.append({
            "conjunto": conjunto, "hogares": len(tabla),
            "mae_estrato_esperado": float((tabla["estrato_esperado"] - y_estrato).abs().mean()),
            "spearman_estrato": float(spearmanr(tabla["estrato_esperado"], y_estrato)[0]),
            "auc_estrato_bajo": float(roc_auc_score(y_estrato <= ESTRATO_BAJO_MAXIMO, tabla["p_estrato_bajo"])),
            "auc_pobre_p_estrato_bajo": float(roc_auc_score(y_pobre, tabla["p_estrato_bajo"])),
            "auc_pobre_estrato_esperado": float(roc_auc_score(y_pobre, -tabla["estrato_esperado"])),
            "correlacion_estrato_esperado_p_bajo": float(tabla["estrato_esperado"].corr(tabla["p_estrato_bajo"])),
        })
    return pd.DataFrame(filas)


def reconciliacion_arriendo(hogar):
    """Compara la regla de arriendo_imputado_hogar() con Ingtotugarr - Ingtotug en
    train, por tenencia.

    Argumentos: hogar = cargar_hogares().
    Devuelve: DataFrame por tenencia (más una fila total) con hogares, hogares con
        adición positiva, mediana de la adición, discrepancias de la regla (más de
        un peso de diferencia) y hogares cuya clase cambiaría sin el arriendo.
    """
    ingreso = pd.read_parquet(RUTA_PROCESADOS / "ingreso_hogar_train.parquet").set_index("id")
    delta = ingreso["Ingtotugarr"] - ingreso["Ingtotug"]
    regla = arriendo_imputado_hogar(hogar).reindex(ingreso.index)
    n_ug = hogar.loc[ingreso.index, "personas_ug"].clip(lower=1)
    sin_arriendo = (ingreso["Ingtotug"] / n_ug < ingreso["Lp"]).astype(int)
    tabla = pd.DataFrame({"tenencia": hogar.loc[ingreso.index, "tenencia"].astype(str), "delta": delta,
                          "discrepa": ~np.isclose(delta, regla, atol=1),
                          "cambia_clase": sin_arriendo.ne(ingreso["Pobre"])})
    filas = []
    for tenencia, g in list(tabla.groupby("tenencia", sort=False)) + [("total", tabla)]:
        positiva = g["delta"] > 0.01
        filas.append({"tenencia": tenencia, "hogares": len(g), "con_arriendo_imputado": int(positiva.sum()),
                      "mediana_arriendo_imputado": float(g.loc[positiva, "delta"].median()) if positiva.any() else 0.0,
                      "discrepancias_regla": int(g["discrepa"].sum()),
                      "tasa_discrepancias": float(g["discrepa"].mean()),
                      "cambia_clase_sin_arriendo": int(g["cambia_clase"].sum())})
    return pd.DataFrame(filas)


def crear_preparador_a(estrato):
    """preparar() de la base de A: las matrices por contexto que guardó la etapa 10
    (1.324 columnas ya imputadas) más las dos variables de estrato del contexto.

    Argumentos: estrato = {contexto: DataFrame} del auxiliar de estrato.
    Devuelve: función(contexto) -> (x_ent, y_ent, x_eval).
    """
    def preparar(contexto):
        x_ent = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_ent.parquet")
        y_ent = x_ent.pop("pobre").to_numpy()
        x_eval = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_eval.parquet")
        extra = estrato[contexto].astype(np.float32)
        x_ent = x_ent.join(extra)
        x_eval = x_eval.join(extra)
        assert x_ent[extra.columns].notna().all().all() and x_eval[extra.columns].notna().all().all()
        return x_ent, y_ent, x_eval

    return preparar


def main():
    reloj = Presupuesto(45)
    registrar_log("etapa11", "inicio")
    iniciar_vigilante_memoria("etapa11")

    # El auxiliar de estrato se ajusta aquí (23 ajustes, con caché y verificación de fuga)
    insumos = cargar_insumos(auxiliar_total=True, auxiliar_estrato=True)
    particion, grupos, hogar = insumos["particion"], insumos["grupos"], insumos["hogar"]
    verdad = particion.set_index("id")["pobre"]
    laboral, total, estrato = (insumos["auxiliares"][k] for k in ("laboral", "total", "estrato"))
    matriz = armar_matriz(insumos)
    registrar_log("etapa11", f"insumos y auxiliar de estrato listos ({reloj.transcurrido():.1f} min)")

    calidad = calidad_auxiliar_estrato(estrato, grupos, cargar_estrato_hogar(), verdad)
    calidad.to_csv(RUTA_TABLAS / "etapa11_estrato_auxiliar.csv", index=False)
    print(calidad.round(4).to_string(index=False))

    reconciliacion = reconciliacion_arriendo(hogar)
    reconciliacion.to_csv(RUTA_TABLAS / "etapa11_arriendo_reconciliacion.csv", index=False)
    print(reconciliacion.round(4).to_string(index=False))

    # 1. Base de 1 semilla y los dos candidatos de 1 semilla
    control = evaluar_o_cargar(CONTROL, lambda: evaluar_candidato(
        CONTROL, matriz, particion, PARAMETROS_N6, [laboral, total]), verdad)
    descripciones = {CONTROL: "base: a4 + auxiliares laboral y total (n6, 1 semilla)"}
    resultados = {CONTROL: control}

    resultados[CANDIDATO_ESTRATO] = evaluar_o_cargar(CANDIDATO_ESTRATO, lambda: evaluar_candidato(
        CANDIDATO_ESTRATO, matriz, particion, PARAMETROS_N6, [laboral, total, estrato]), verdad)
    descripciones[CANDIDATO_ESTRATO] = "base + estrato esperado y P(estrato <= 2) del auxiliar de estrato"
    registrar_log("etapa11", f"{CANDIDATO_ESTRATO} listo ({reloj.transcurrido():.1f} min)")

    con_arriendo = reconciliar_auxiliares({"laboral": laboral, "total": total}, hogar, arriendo_imputado=True)
    resultados[CANDIDATO_ARRIENDO] = evaluar_o_cargar(CANDIDATO_ARRIENDO, lambda: evaluar_candidato(
        CANDIDATO_ARRIENDO, matriz, particion, PARAMETROS_N6, [con_arriendo["laboral"], con_arriendo["total"]]), verdad)
    descripciones[CANDIDATO_ARRIENDO] = "base con el per cápita de los auxiliares reconciliado con el arriendo imputado"
    registrar_log("etapa11", f"{CANDIDATO_ARRIENDO} listo ({reloj.transcurrido():.1f} min)")

    filas = [fila_tabla(resultados[n], descripciones[n], None if n == CONTROL else control) for n in resultados]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / "etapa11_candidatos.csv", index=False)

    # 2. Regla de paso al bagging sobre la base de A
    delta_cv = resultados[CANDIDATO_ESTRATO]["f1_cv"] - control["f1_cv"]
    delta_holdout = resultados[CANDIDATO_ESTRATO]["f1_holdout"] - control["f1_holdout"]
    pasa = delta_cv > MEJORA_MINIMA and delta_holdout > MEJORA_MINIMA
    registrar_log("etapa11", f"estrato: delta CV {delta_cv:+.4f}, holdout {delta_holdout:+.4f} -> "
                             f"{'pasa' if pasa else 'no pasa'} a bagging sobre A")
    if pasa:
        base_a = resultado_desde_predicciones(BASE_A, verdad)
        resultados[CANDIDATO_BAG5] = evaluar_o_cargar(CANDIDATO_BAG5, lambda: evaluar_modelo(
            CANDIDATO_BAG5, crear_preparador_a(estrato), crear_ajuste_lightgbm(PARAMETROS_N6, SEMILLAS_BAGGING),
            particion, con_test=True, reanudable=True), verdad)
        borrar_parciales(CANDIDATO_BAG5)
        descripciones[CANDIDATO_BAG5] = "base de A + estrato (n6, bagging 5 semillas)"
        filas.append(fila_tabla(base_a, "referencia: A (n6, bagging 5 semillas)"))
        filas.append(fila_tabla(resultados[CANDIDATO_BAG5], descripciones[CANDIDATO_BAG5], base_a))
        tabla = pd.DataFrame(filas)
        tabla.to_csv(RUTA_TABLAS / "etapa11_candidatos.csv", index=False)
        supera = (resultados[CANDIDATO_BAG5]["f1_cv"] > base_a["f1_cv"]
                  and resultados[CANDIDATO_BAG5]["f1_holdout"] > base_a["f1_holdout"])
        if supera:
            ids_test = particion.loc[particion["base"] == "test", "id"]
            ruta, tasa = guardar_submission(
                resultados[CANDIDATO_BAG5], f"{CANDIDATO_BAG5}.csv", ids_test,
                "LightGBM (n6, bagging 5 semillas) | base de A + estrato esperado y P(estrato <= 2) de un "
                "auxiliar multiclase de Estrato1 con OOF anidada | umbral OOF")
            registrar_log("etapa11", f"CSV {ruta.name} (tasa de pobres {tasa:.3f})")
        else:
            registrar_log("etapa11", f"{CANDIDATO_BAG5} no supera a {BASE_A} en CV y holdout: sin CSV")

    # 3. Veredicto de la corrección del arriendo imputado
    arriendo = resultados[CANDIDATO_ARRIENDO]
    no_empeora = arriendo["f1_cv"] >= control["f1_cv"] and arriendo["f1_holdout"] >= control["f1_holdout"]
    registrar_log("etapa11", f"arriendo imputado: delta CV {arriendo['f1_cv'] - control['f1_cv']:+.4f}, holdout "
                             f"{arriendo['f1_holdout'] - control['f1_holdout']:+.4f} -> "
                             f"{'no empeora: adoptar' if no_empeora else 'empeora: no adoptar'}")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tabla[["candidato", "umbral", "f1_cv", "f1_cv_sd", "f1_holdout", "precision_holdout",
                     "recall_holdout", "delta_cv", "delta_holdout", "tiempo_min"]].round(4).to_string(index=False))
    registrar_log("etapa11", f"fin ({reloj.transcurrido():.1f} min)")


if __name__ == "__main__":
    main()
