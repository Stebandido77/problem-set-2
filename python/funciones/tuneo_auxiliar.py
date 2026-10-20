"""Tuning con Optuna de los auxiliares de ingreso de personas (laboral y total)
sin fuga hacia la OOF anidada (etapa 8).

Los hiperparámetros se eligen por separado para cada grupo de parámetros:
  cv_k  -> se tunea solo con los folds del 80% distintos de k
  80    -> se tunea con los folds 1-5; lo usan los contextos holdout y test
Así, en cv_k ningún paso (ni la búsqueda ni los ajustes) ve el fold k, y el
holdout nunca entra a una búsqueda. Dentro del grupo, la búsqueda entrena con
todos sus folds menos el último y mide el RMSE de log(1 + ingreso) por persona
en ese último fold, con parada temprana para fijar el número de árboles.
"""

import lightgbm as lgb
import numpy as np
import optuna

from configuracion import N_HILOS, SEMILLA
from funciones.anidado import GRUPO_HOLDOUT, GRUPO_TEST, GRUPOS_80, definir_contextos
from funciones.auxiliares import PARAMETROS_AUXILIAR, preparar_personas_auxiliar

MAXIMO_ARBOLES = 1000
PARADA_TEMPRANA = 50


def definir_grupos_tuneo():
    """Folds con que se tunea cada grupo de parámetros.

    Devuelve: diccionario {grupo_parametros: tupla de folds}.
    """
    grupos = {f"cv_{k}": tuple(g for g in GRUPOS_80 if g != k) for k in GRUPOS_80}
    grupos["80"] = GRUPOS_80
    return grupos


def grupo_de_contexto(contexto):
    """Grupo de parámetros que usa un contexto del modelo principal."""
    return contexto if contexto.startswith("cv_") else "80"


def verificar_tuneo_sin_fuga(contextos=None):
    """Falla si la búsqueda de algún contexto usó folds fuera de su entrenamiento
    (en particular, su fold de evaluación, el holdout o el test).

    Devuelve: True si todo está bien (si no, AssertionError).
    """
    contextos = contextos or definir_contextos()
    grupos_tuneo = definir_grupos_tuneo()
    for contexto, (entrenamiento, evaluacion) in contextos.items():
        usados = set(grupos_tuneo[grupo_de_contexto(contexto)])
        assert usados <= set(entrenamiento), f"{contexto}: la búsqueda usa folds fuera de su entrenamiento"
        assert not usados & set(evaluacion), f"{contexto}: la búsqueda vio su evaluación"
        assert not usados & {GRUPO_HOLDOUT, GRUPO_TEST}, f"{contexto}: la búsqueda vio holdout o test"
    return True


def particion_interna(folds):
    """Folds de entrenamiento y de validación de la búsqueda de un grupo.

    Argumentos: folds = definir_grupos_tuneo()[grupo].
    Devuelve: (folds de entrenamiento, fold de validación).
    """
    validacion = max(folds)
    return tuple(f for f in folds if f != validacion), validacion


def espacio_busqueda(trial):
    """Rango de los hiperparámetros del auxiliar (log donde la escala lo pide).

    learning_rate tiene piso de 0,03 y los árboles tope de MAXIMO_ARBOLES para que
    los 38 reajustes anidados quepan en el tiempo de la etapa.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
    }


def tunear_auxiliar_personas(personas, ingreso, grupos, columna, grupo_parametros, minutos, almacenamiento):
    """Estudio de Optuna (TPE) para un auxiliar y un grupo de parámetros.

    Reanudable: el estudio vive en almacenamiento (sqlite) y, al retomarlo, solo
    corre el tiempo que le falta (minutos menos lo que ya duraron sus trials).
    El primer trial son los parámetros actuales (PARAMETROS_AUXILIAR), para
    medir cuánto mejora el error.
    Argumentos: personas = cargar_personas(); ingreso = cargar_ingreso_personas();
        grupos = asignar_grupos(); columna = "ingreso_laboral" o "ingreso_total";
        grupo_parametros = clave de definir_grupos_tuneo(); minutos = presupuesto;
        almacenamiento = URL sqlite de Optuna.
    Devuelve: diccionario con parametros (listos para crear_auxiliar_personas()),
        rmse_defecto, rmse_tuneado, n_trials y minutos usados.
    """
    entrenamiento, validacion = particion_interna(definir_grupos_tuneo()[grupo_parametros])
    grupo_persona = grupos.reindex(personas["id"]).to_numpy()
    filas_ent = np.isin(grupo_persona, entrenamiento)
    filas_val = grupo_persona == validacion
    assert not (filas_ent & filas_val).any()
    objetivo = personas[["id", "Orden"]].merge(ingreso, on=["id", "Orden"], how="left")[columna].to_numpy()
    y = np.log1p(objetivo)
    assert not np.isnan(y[filas_ent | filas_val]).any(), "personas de la búsqueda sin ingreso"
    x = preparar_personas_auxiliar(personas)
    # feature_pre_filter = False: el mismo Dataset sirve para distintos min_data_in_leaf
    datos_ent = lgb.Dataset(x[filas_ent], label=y[filas_ent], params={"feature_pre_filter": False},
                            free_raw_data=False)
    datos_val = lgb.Dataset(x[filas_val], label=y[filas_val], reference=datos_ent, free_raw_data=False)
    del x

    def objetivo_optuna(trial):
        parametros = {"objective": "regression", "metric": "rmse", "bagging_freq": 1, "verbose": -1,
                      "num_threads": N_HILOS, "seed": SEMILLA, **espacio_busqueda(trial)}
        modelo = lgb.train(parametros, datos_ent, num_boost_round=MAXIMO_ARBOLES, valid_sets=[datos_val],
                           callbacks=[lgb.early_stopping(PARADA_TEMPRANA, verbose=False)])
        trial.set_user_attr("num_iterations", int(modelo.best_iteration))
        return float(modelo.best_score["valid_0"]["rmse"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(study_name=f"{columna}_{grupo_parametros}", storage=almacenamiento,
                                  direction="minimize", load_if_exists=True,
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    if not estudio.trials:
        estudio.enqueue_trial({k: PARAMETROS_AUXILIAR[k] for k in
                               ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
                                "bagging_fraction")} | {"lambda_l2": 1e-3})
    completos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    usados = sum((t.datetime_complete - t.datetime_start).total_seconds() for t in completos) / 60
    if minutos - usados > 0:
        estudio.optimize(objetivo_optuna, timeout=(minutos - usados) * 60, gc_after_trial=True)
    completos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    mejor = estudio.best_trial
    parametros = {**PARAMETROS_AUXILIAR, **mejor.params, "num_iterations": mejor.user_attrs["num_iterations"]}
    return {"parametros": parametros, "rmse_defecto": completos[0].value, "rmse_tuneado": mejor.value,
            "n_trials": len(completos), "folds_entrenamiento": list(entrenamiento), "fold_validacion": validacion,
            "minutos": sum((t.datetime_complete - t.datetime_start).total_seconds() for t in completos) / 60}
