"""Etapa 12, experimento 1: AutoGluon (TabularPredictor, presets="best_quality",
eval_metric="f1" con 1 = pobre) sobre dos bases, con el protocolo del nivel 7.

Corre en el entorno aparte python/.venv_autogluon/ (AutoGluon fija versiones de
torch, numpy y pandas distintas a las del entorno principal).

  Corrida A: base casi cruda (variables de hogar + conteos, medias y máximos de
             las variables originales de personas por rol), sin variables
             construidas ni auxiliares; la escribe etapa12_enfoques.py preparar.
  Corrida B: base de A (1.324 columnas con auxiliares de la OOF anidada), leída
             de las matrices de la etapa 10.

Protocolo: un solo ajuste sobre el 80% con bagging por NUESTROS cinco folds
(groups="fold_id"), de modo que la OOF de AutoGluon es OOF sobre los mismos folds
de R; el umbral sale de esa OOF con buscar_umbral_f1(); el holdout se predice con
el ensamble ajustado en el 80% y nunca entra al entrenamiento. En la corrida B
los auxiliares de las filas de entrenamiento son los del contexto holdout
(OOF dentro del 80%): el holdout es limpio y la F1 CV puede ser algo optimista.
Si la corrida pasa la regla (+0,002 en CV y holdout sobre n7_a_agregados_bag5),
reajusta sobre el 100% (holdout como sexto grupo) y predice el test.

Uso (desde la raíz del repo):
    python/.venv_autogluon/Scripts/python.exe python/etapa12_autogluon.py --corrida A --minutos 40
"""

import argparse
import inspect
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import RUTA_NIVEL7, RUTA_TABLAS
from funciones.anidado import GRUPO_HOLDOUT
from funciones.bitacora import registrar_log
from funciones.datos import cargar_particion
from funciones.metricas import buscar_umbral_f1, calcular_metricas, metricas_por_fold
from funciones.modelado import guardar_predicciones, resultado_desde_predicciones

BASE = "n7_a_agregados_bag5"
MEJORA_MINIMA = 0.002
RUTA_ETAPA12 = RUTA_NIVEL7 / "etapa12"
RUTA_MATRICES_A = RUTA_NIVEL7 / "etapa10" / "matrices"
NUCLEOS = int(os.environ.get("PS2_NUCLEOS", 12))
MEMORIA_GB = 7  # memory_limit de fit(): la máquina tiene ~9 GB libres con lo demás abierto
EXCLUIDOS = ["KNN"]  # vecinos sobre cientos de columnas: lento y sin señal
MINUTOS_IMPORTANCIA = 10


def cargar_base(corrida, contexto, particion):
    """Matrices de entrenamiento y evaluación de la corrida en un contexto.

    Argumentos: corrida = "A" (base cruda) o "B" (base de A); contexto =
        "holdout" (80% -> holdout) o "test" (100% -> test); particion =
        cargar_particion() indexada por id.
    Devuelve: (x_ent, y_ent, grupos_ent, x_eval) con los ids en el índice.
    """
    tabla = particion
    if corrida == "B":
        x_ent = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_ent.parquet")
        y_ent = x_ent.pop("pobre").astype(int)
        x_eval = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_eval.parquet")
    else:
        base = pd.read_parquet(RUTA_ETAPA12 / "base_cruda.parquet")
        if contexto == "holdout":
            ids_ent = tabla.index[(tabla["base"] == "train") & ~tabla["holdout"]]
            ids_eval = tabla.index[tabla["holdout"]]
        else:
            ids_ent = tabla.index[tabla["base"] == "train"]
            ids_eval = tabla.index[tabla["base"] == "test"]
        x_ent, x_eval = base.loc[ids_ent], base.loc[ids_eval]
        y_ent = tabla.loc[ids_ent, "pobre"].astype(int)
    grupos = tabla.loc[x_ent.index, "fold_id"].astype("float").fillna(GRUPO_HOLDOUT).astype(int)
    assert not x_ent.index.isin(x_eval.index).any() and y_ent.index.equals(x_ent.index)
    return x_ent, y_ent, grupos, x_eval


def argumentos_soportados(funcion, candidatos):
    """Deja solo los argumentos que existen en la firma de la versión instalada."""
    parametros = inspect.signature(funcion).parameters
    return {k: v for k, v in candidatos.items() if k in parametros}


def ajustar_autogluon(x_ent, y_ent, grupos, ruta, minutos):
    """Ajusta un TabularPredictor con bagging por los grupos dados.

    Argumentos: x_ent, y_ent, grupos = entrenamiento; ruta = carpeta del modelo;
        minutos = límite de tiempo de fit().
    Devuelve: (predictor, DataFrame de entrenamiento tal como lo vio AutoGluon).
    """
    from autogluon.tabular import TabularPredictor

    datos = x_ent.copy()
    datos["pobre"] = y_ent.to_numpy()
    datos["fold_id"] = grupos.to_numpy()
    datos = datos.reset_index(drop=True)
    # groups: columna con el fold de cada hogar; AutoGluon la usa como partición
    # del bagging (num_bag_folds = número de grupos) y no como predictor
    predictor = TabularPredictor(**argumentos_soportados(TabularPredictor.__init__, {
        "label": "pobre", "problem_type": "binary", "eval_metric": "f1", "positive_class": 1,
        "groups": "fold_id", "path": str(ruta), "verbosity": 2}))
    assert predictor._learner.groups == "fold_id", "la versión instalada no acepta groups"
    opciones = argumentos_soportados(predictor.fit, {
        "presets": "best_quality", "time_limit": int(minutos * 60), "num_cpus": NUCLEOS,
        "memory_limit": MEMORIA_GB, "dynamic_stacking": False, "calibrate_decision_threshold": False,
        "verbosity": 2})
    # excluded_model_types llega por **kwargs de fit(), por eso no pasa por el filtro
    predictor.fit(datos, excluded_model_types=EXCLUIDOS, **opciones)
    return predictor, datos


def probabilidad_pobre(tabla):
    """Columna de P(pobre) de una salida de predict_proba()/predict_proba_oof()."""
    if isinstance(tabla, pd.Series):
        return tabla.to_numpy()
    columna = 1 if 1 in tabla.columns else tabla.columns[-1]
    return tabla[columna].to_numpy()


def corte_duro(etapa, minutos):
    """Termina el proceso si se pasa del tope (AutoGluon trata time_limit como orientativo)."""
    def vigilar():
        time.sleep(minutos * 60)
        registrar_log(etapa, f"CORTE DURO: {minutos:.0f} min agotados")
        os._exit(4)
    threading.Thread(target=vigilar, daemon=True).start()


def main(corrida, minutos, minutos_totales):
    inicio = time.time()
    etapa = f"etapa12_autogluon_{corrida}"
    nombre = f"n12_autogluon_{corrida.lower()}"
    RUTA_ETAPA12.mkdir(parents=True, exist_ok=True)
    corte_duro(etapa, minutos_totales)
    registrar_log(etapa, f"inicio: corrida {corrida}, {minutos} min de fit, {NUCLEOS} núcleos")
    particion = cargar_particion().set_index("id")
    verdad = particion["pobre"]

    # 1. Ajuste sobre el 80% con bagging por los folds de R
    x_ent, y_ent, grupos, x_holdout = cargar_base(corrida, "holdout", particion)
    registrar_log(etapa, f"80%: {x_ent.shape[0]} hogares x {x_ent.shape[1]} columnas; holdout {len(x_holdout)}")
    predictor, _ = ajustar_autogluon(x_ent, y_ent, grupos, RUTA_ETAPA12 / f"autogluon_{corrida}", minutos)
    registrar_log(etapa, f"fit listo ({(time.time() - inicio) / 60:.1f} min); mejor modelo {predictor.model_best}")

    # 2. OOF sobre nuestros folds -> umbral; holdout con ese umbral
    oof = pd.DataFrame({"probabilidad": probabilidad_pobre(predictor.predict_proba_oof()),
                        "fold": grupos.to_numpy(), "verdad": y_ent.to_numpy()}, index=x_ent.index)
    umbral, _ = buscar_umbral_f1(oof["verdad"], oof["probabilidad"])
    metricas_cv = metricas_por_fold(oof["verdad"], oof["probabilidad"], oof["fold"], umbral)
    prob_holdout = pd.Series(probabilidad_pobre(predictor.predict_proba(x_holdout)), index=x_holdout.index,
                             name="probabilidad")
    metricas_holdout = calcular_metricas(verdad.loc[prob_holdout.index].astype(int), (prob_holdout >= umbral).astype(int))
    resultado = {"nombre": nombre, "umbral": umbral, **metricas_cv,
                 **{f"{k}_holdout": v for k, v in metricas_holdout.items()},
                 "oof": oof, "prediccion_holdout": prob_holdout}
    guardar_predicciones(resultado)
    registrar_log(etapa, f"F1 CV {metricas_cv['f1_cv']:.4f} | holdout {metricas_holdout['f1']:.4f} | umbral {umbral:.4f}")

    # 3. Qué modelos usa y qué variables pesan
    holdout_con_etiqueta = x_holdout.copy()
    holdout_con_etiqueta["pobre"] = verdad.loc[x_holdout.index].astype(int).to_numpy()
    tablero = predictor.leaderboard(holdout_con_etiqueta.reset_index(drop=True), silent=True)
    tablero.to_csv(RUTA_TABLAS / f"etapa12_autogluon_{corrida}_modelos.csv", index=False)
    registrar_log(etapa, f"leaderboard: {len(tablero)} modelos")
    try:
        importancia = predictor.feature_importance(
            holdout_con_etiqueta.reset_index(drop=True), subsample_size=3000, num_shuffle_sets=1,
            time_limit=MINUTOS_IMPORTANCIA * 60, silent=True)
        importancia.reset_index().rename(columns={"index": "variable"}).to_csv(
            RUTA_TABLAS / f"etapa12_autogluon_{corrida}_importancia.csv", index=False)
        registrar_log(etapa, f"importancia por permutación: {len(importancia)} variables")
    except Exception as error:  # noqa: BLE001 (la importancia es informativa; no detiene la corrida)
        registrar_log(etapa, f"sin importancia: {type(error).__name__}: {error}"[:200])

    # 4. Regla de la etapa: solo si supera a la base en CV y holdout se predice el test
    base = resultado_desde_predicciones(BASE, verdad)
    pasa = (metricas_cv["f1_cv"] >= base["f1_cv"] + MEJORA_MINIMA
            and metricas_holdout["f1"] >= base["f1_holdout"] + MEJORA_MINIMA)
    registrar_log(etapa, f"frente a {BASE}: CV {metricas_cv['f1_cv'] - base['f1_cv']:+.4f}, "
                         f"holdout {metricas_holdout['f1'] - base['f1_holdout']:+.4f} -> {'pasa' if pasa else 'no pasa'}")
    restante = minutos_totales - (time.time() - inicio) / 60 - 10
    if pasa and restante >= 15:
        x_todo, y_todo, grupos_todo, x_test = cargar_base(corrida, "test", particion)
        predictor_test, _ = ajustar_autogluon(x_todo, y_todo, grupos_todo, RUTA_ETAPA12 / f"autogluon_{corrida}_test",
                                              min(minutos, restante))
        resultado["prediccion_test"] = pd.Series(probabilidad_pobre(predictor_test.predict_proba(x_test)),
                                                 index=x_test.index, name="probabilidad")
        guardar_predicciones(resultado)
        registrar_log(etapa, "predicción de test guardada (reajuste sobre el 100%)")
    elif pasa:
        registrar_log(etapa, f"pasa pero no caben los {minutos} min del reajuste sobre el 100%: sin test")
    pd.Series({"minutos": (time.time() - inicio) / 60}).to_json(RUTA_ETAPA12 / f"tiempo_{nombre}.json")
    registrar_log(etapa, f"fin ({(time.time() - inicio) / 60:.1f} min)")


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("--corrida", choices=["A", "B"], required=True)
    analizador.add_argument("--minutos", type=float, default=40, help="límite de tiempo del fit")
    analizador.add_argument("--minutos-totales", type=float, default=None,
                            help="corte duro del proceso (por defecto 2,5 x minutos + 20)")
    args = analizador.parse_args()
    main(args.corrida, args.minutos, args.minutos_totales or 2.5 * args.minutos + 20)
