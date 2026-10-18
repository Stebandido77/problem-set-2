"""Etapa 2: Optuna (TPE) sobre la base ganadora y seed bagging de 5 semillas.

Protocolo de cada trial (un solo proceso, LightGBM con hilos internos):
  - mismos 5 folds del 80%; en cada fold el early stopping usa un 10%
    estratificado del entrenamiento del fold (nunca el fold de validación);
  - la métrica es el F1 OOF con umbral óptimo sobre los 5 folds;
  - poda con MedianPruner sobre el F1 OOF acumulado tras cada fold.
Presupuesto por tiempo (timeout), no por número de trials. Los trials quedan en
stores/procesados/nivel7/optuna_etapa2.db, así que se puede retomar.

Con los mejores parámetros (num_iterations = media de las mejores iteraciones /
0,9, porque el reajuste usa el 100% del entrenamiento del fold) se corre
evaluar_candidato() con 1 semilla y con 5 semillas, ambos con test.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa2_optuna.py --minutos 40 [--nativas] [--auxiliar-total]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from configuracion import N_HILOS, PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from funciones.anidado import definir_contextos
from funciones.bases import armar_matriz, cargar_insumos
from funciones.bitacora import registrar_log
from funciones.datos import imputar_mediana
from funciones.metricas import buscar_umbral_f1
from funciones.modelado import armar_contexto, evaluar_candidato, f1_subgrupos

MAXIMO_ITERACIONES = 4000
PACIENCIA = 100
SEMILLAS_BAGGING = tuple(SEMILLA + i for i in range(5))


def preparar_folds(matriz, dinamicas, particion):
    """Datos de cada fold, imputados una sola vez y reutilizados en todos los trials.

    Devuelve: lista de diccionarios con x/y de ajuste, de parada temprana y de validación.
    """
    tabla = particion.set_index("id")
    contextos = definir_contextos()
    folds = []
    for k in range(1, 6):
        entrenamiento, _ = contextos[f"cv_{k}"]
        ids_ent = tabla.index[tabla["fold_id"].isin(entrenamiento) & ~tabla["holdout"]]
        ids_val = tabla.index[tabla["fold_id"].isin([k])]
        x_ent = armar_contexto(matriz, dinamicas, f"cv_{k}", ids_ent)
        x_val = armar_contexto(matriz, dinamicas, f"cv_{k}", ids_val)
        x_ent, x_val = imputar_mediana(x_ent, x_val)
        y_ent = tabla.loc[ids_ent, "pobre"].astype(int).to_numpy()
        x_ajuste, x_parada, y_ajuste, y_parada = train_test_split(
            x_ent, y_ent, test_size=0.1, stratify=y_ent, random_state=SEMILLA)
        folds.append({"x_ajuste": x_ajuste, "y_ajuste": y_ajuste, "x_parada": x_parada,
                      "y_parada": y_parada, "x_val": x_val,
                      "y_val": tabla.loc[ids_val, "pobre"].astype(int).to_numpy()})
    return folds


def proponer_parametros(trial):
    """Espacio de búsqueda amplio (rangos fijos, muestreo TPE)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def crear_objetivo(folds):
    """Función objetivo de Optuna: F1 OOF con umbral óptimo."""
    def objetivo(trial):
        parametros = {**proponer_parametros(trial), "objective": "binary", "bagging_freq": 1,
                      "num_threads": N_HILOS, "seed": SEMILLA, "verbose": -1,
                      "feature_pre_filter": False}
        verdad, probabilidad, iteraciones = [], [], []
        for k, fold in enumerate(folds):
            modelo = lgb.train(
                parametros, lgb.Dataset(fold["x_ajuste"], label=fold["y_ajuste"]),
                num_boost_round=MAXIMO_ITERACIONES,
                valid_sets=[lgb.Dataset(fold["x_parada"], label=fold["y_parada"])],
                callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)],
            )
            iteraciones.append(modelo.best_iteration)
            verdad.append(fold["y_val"])
            probabilidad.append(modelo.predict(fold["x_val"], num_iteration=modelo.best_iteration))
            _, f1_parcial = buscar_umbral_f1(np.concatenate(verdad), np.concatenate(probabilidad))
            trial.report(f1_parcial, k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("iteraciones", iteraciones)
        _, f1 = buscar_umbral_f1(np.concatenate(verdad), np.concatenate(probabilidad))
        return f1

    return objetivo


def parametros_finales(estudio):
    """Mejores parámetros traducidos al formato de evaluar_candidato()."""
    mejor = estudio.best_trial
    iteraciones = int(round(np.mean(mejor.user_attrs["iteraciones"]) / 0.9))
    return {**mejor.params, "objective": "binary", "bagging_freq": 1,
            "num_iterations": iteraciones, "feature_pre_filter": False}


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--minutos", type=float, default=40)
    argumentos.add_argument("--nativas", action="store_true", help="perceptor_oficio nativo")
    argumentos.add_argument("--auxiliar-total", action="store_true", help="suma el auxiliar total")
    argumentos.add_argument("--solo-evaluar", action="store_true", help="no corre trials nuevos")
    opciones = argumentos.parse_args()

    insumos = cargar_insumos(auxiliar_total=opciones.auxiliar_total)
    matriz = armar_matriz(insumos, nativas=["perceptor_oficio"] if opciones.nativas else ())
    dinamicas = list(insumos["auxiliares"].values())
    particion = insumos["particion"]
    base = "a4" + ("_nativo" if opciones.nativas else "") + ("_total" if opciones.auxiliar_total else "")

    estudio = optuna.create_study(
        study_name=f"etapa2_{base}", direction="maximize", load_if_exists=True,
        storage=f"sqlite:///{(RUTA_NIVEL7 / 'optuna_etapa2.db').as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    if not opciones.solo_evaluar:
        if len(estudio.trials) == 0:
            # Punto de partida: los parámetros de n6 en este espacio
            estudio.enqueue_trial({
                "learning_rate": 0.0114, "num_leaves": 31, "max_depth": 12,
                "min_child_samples": 171, "feature_fraction": 0.82, "bagging_fraction": 0.70,
                "lambda_l1": 1e-8, "lambda_l2": 1e-8})
        folds = preparar_folds(matriz, dinamicas, particion)
        registrar_log("etapa2_optuna", f"inicio {base} ({opciones.minutos:.0f} min)")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inicio = time.time()
        estudio.optimize(
            crear_objetivo(folds), timeout=opciones.minutos * 60, gc_after_trial=True,
            callbacks=[lambda e, t: registrar_log(
                "etapa2_optuna", f"trial {t.number} {t.state.name} "
                f"{t.value if t.value is not None else float('nan'):.4f} mejor {e.best_value:.4f}")])
        registrar_log("etapa2_optuna", f"fin: {len(estudio.trials)} trials", inicio)
        del folds

    estudio.trials_dataframe().to_csv(RUTA_TABLAS / f"optuna_etapa2_{base}.csv", index=False)
    parametros = parametros_finales(estudio)
    (RUTA_NIVEL7 / f"parametros_optuna_{base}.json").write_text(json.dumps(parametros, indent=2))
    registrar_log("etapa2_optuna", f"mejor trial {estudio.best_trial.number}: {estudio.best_value:.4f}")

    filas = []
    for nombre, semillas in ((f"n7_optuna_{base}", (SEMILLA,)), (f"n7_optuna_{base}_bag5", SEMILLAS_BAGGING)):
        resultado = evaluar_candidato(nombre, matriz, particion, parametros, dinamicas,
                                      con_test=True, semillas=semillas)
        filas.append({"candidato": nombre, "semillas": len(semillas), "umbral": resultado["umbral"],
                      "f1_cv": resultado["f1_cv"], "f1_cv_sd": resultado["f1_cv_sd"],
                      "f1_holdout": resultado["f1_holdout"],
                      "precision_holdout": resultado["precision_holdout"],
                      "recall_holdout": resultado["recall_holdout"],
                      **f1_subgrupos(resultado, insumos["hogar"], particion),
                      "tiempo_min": resultado["tiempo_min"]})
        pd.DataFrame(filas).to_csv(RUTA_TABLAS / f"optuna_bagging_{base}.csv", index=False)
    print(json.dumps(parametros, indent=2))
    print(pd.DataFrame(filas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
