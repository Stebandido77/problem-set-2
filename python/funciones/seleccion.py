"""Selección de variables con NSGA-II (pymoo) para la encuesta corta.

El cromosoma es una máscara binaria sobre las variables ORIGINALES: una
categórica entra o sale completa (todas sus dummies). El modelo es un Logit sin
penalización con la receta de R (mediana, dummies, estandarización); la
estandarización por columna no depende de qué otras columnas entren, así que
cada fold se prepara una sola vez y cada individuo solo elige columnas.
Objetivos: maximizar el F1 OOF (umbral óptimo) y minimizar el número de variables.
"""

import time
import warnings

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from configuracion import SEMILLA
from funciones.datos import imputar_mediana
from funciones.metricas import buscar_umbral_f1, calcular_metricas


def mapear_columnas(variables, matriz):
    """Columnas de la matriz en dummies que corresponden a cada variable original.

    Argumentos: variables = DataFrame de variables originales; matriz = preparar_matriz(variables).
    Devuelve: lista (una entrada por variable original) de índices de columna.
    """
    posiciones = {c: i for i, c in enumerate(matriz.columns)}
    mapa = []
    for v in variables.columns:
        if isinstance(variables[v].dtype, pd.CategoricalDtype):
            mapa.append([i for c, i in posiciones.items() if c.startswith(f"{v}_")])
        else:
            mapa.append([posiciones[v]])
    return mapa


def preparar_division(matriz, ids_entrenamiento, ids_evaluacion, verdad):
    """Imputa con la mediana y estandariza con la media/sd del entrenamiento.

    Devuelve: diccionario con x/y de entrenamiento y de evaluación (float64).
    """
    x_ent, x_eval = imputar_mediana(matriz.loc[ids_entrenamiento], matriz.loc[ids_evaluacion])
    media = x_ent.mean()
    desviacion = x_ent.std().replace(0, 1)
    return {
        "x_ent": ((x_ent - media) / desviacion).to_numpy(np.float64),
        "y_ent": verdad.loc[ids_entrenamiento].astype(int).to_numpy(),
        "x_eval": ((x_eval - media) / desviacion).to_numpy(np.float64),
        "y_eval": verdad.loc[ids_evaluacion].astype(int).to_numpy(),
    }


def ajustar_logit(division, columnas):
    """Logit sin penalización sobre un subconjunto de columnas.

    Devuelve: P(pobre) en la parte de evaluación.
    """
    modelo = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=300)  # sin penalización
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo.fit(division["x_ent"][:, columnas], division["y_ent"])
    return modelo.predict_proba(division["x_eval"][:, columnas])[:, 1]


def evaluar_mascara(mascara, divisiones, mapa):
    """F1 OOF con umbral óptimo de un subconjunto de variables originales.

    Devuelve: (f1, umbral).
    """
    columnas = sorted(c for v in np.flatnonzero(mascara) for c in mapa[v])
    verdad = np.concatenate([d["y_eval"] for d in divisiones])
    probabilidad = np.concatenate([ajustar_logit(d, columnas) for d in divisiones])
    umbral, f1 = buscar_umbral_f1(verdad, probabilidad)
    return f1, umbral


class ProblemaEncuesta(ElementwiseProblem):
    """Dos objetivos: -F1 OOF y número de variables. Guarda en caché cada máscara."""

    def __init__(self, divisiones, mapa):
        super().__init__(n_var=len(mapa), n_obj=2, xl=0, xu=1, vtype=bool)
        self.divisiones = divisiones
        self.mapa = mapa
        self.cache = {}

    def _evaluate(self, x, out, *args, **kwargs):
        mascara = np.asarray(x, dtype=bool)
        if not mascara.any():
            out["F"] = [0.0, 0.0]  # sin variables: F1 nulo, dominado por cualquiera útil
            return
        llave = mascara.tobytes()
        if llave not in self.cache:
            self.cache[llave] = evaluar_mascara(mascara, self.divisiones, self.mapa)[0]
        out["F"] = [-self.cache[llave], int(mascara.sum())]


class MuestreoDensidades(Sampling):
    """Población inicial con densidades variadas (de 2 variables a casi todas) más
    individuos semilla (p. ej. las 10 variables de nE_logit)."""

    def __init__(self, semillas=()):
        super().__init__()
        self.semillas = [np.asarray(s, dtype=bool) for s in semillas]

    def _do(self, problem, n_samples, **kwargs):
        generador = np.random.default_rng(SEMILLA)
        densidades = np.geomspace(2 / problem.n_var, 0.9, n_samples - len(self.semillas))
        aleatorios = [generador.random(problem.n_var) < d for d in densidades]
        return np.array(self.semillas + aleatorios, dtype=bool)


def evaluar_en_holdout(mascara, division_holdout, divisiones_oof, mapa):
    """Umbral con OOF de 5 folds y métricas en holdout (reajuste sobre el 80%).

    Devuelve: diccionario con f1_oof, umbral y métricas de holdout.
    """
    f1_oof, umbral = evaluar_mascara(mascara, divisiones_oof, mapa)
    columnas = sorted(c for v in np.flatnonzero(mascara) for c in mapa[v])
    inicio = time.time()
    probabilidad = ajustar_logit(division_holdout, columnas)
    metricas = calcular_metricas(division_holdout["y_eval"], (probabilidad >= umbral).astype(int))
    return {"f1_oof_5folds": f1_oof, "umbral": umbral,
            **{f"{k}_holdout": v for k, v in metricas.items()},
            "segundos_holdout": time.time() - inicio}
