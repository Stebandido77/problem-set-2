"""Métricas de clasificación y búsqueda de umbral, idénticas a las de R
(calcular_metricas(), buscar_umbral_f1() y metricas_por_fold() en scripts/funciones/)."""

import numpy as np
import pandas as pd


def calcular_metricas(verdad, prediccion):
    """F1, precisión, recall y accuracy con 1 = pobre.

    Argumentos: verdad, prediccion = arreglos 0/1.
    Devuelve: diccionario con f1, precision, recall, accuracy, filtracion,
        subcobertura y los conteos de la matriz de confusión.
    """
    verdad = np.asarray(verdad)
    prediccion = np.asarray(prediccion)
    vp = int(np.sum((verdad == 1) & (prediccion == 1)))
    fp = int(np.sum((verdad == 0) & (prediccion == 1)))
    fn = int(np.sum((verdad == 1) & (prediccion == 0)))
    vn = int(np.sum((verdad == 0) & (prediccion == 0)))
    precision = 0.0 if vp + fp == 0 else vp / (vp + fp)
    recall = 0.0 if vp + fn == 0 else vp / (vp + fn)
    f1 = 0.0 if vp == 0 else 2 * vp / (2 * vp + fp + fn)
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "accuracy": (vp + vn) / len(verdad),
        "filtracion": 1 - precision, "subcobertura": 1 - recall,
        "n_vp": vp, "n_fp": fp, "n_fn": fn, "n_vn": vn,
    }


def calcular_curva_umbral(verdad, probabilidad):
    """Métricas en cada umbral de la grilla de R (como $curva de buscar_umbral_f1()).

    Grilla: 0,01 a 0,99 más los percentiles 0,5% a 99,5% de las probabilidades,
    sin umbrales <= 0.
    Argumentos: verdad = arreglo 0/1; probabilidad = P(pobre).
    Devuelve: DataFrame con umbral, f1, precision, recall, accuracy, filtracion,
        subcobertura y los conteos de la matriz de confusión.
    """
    verdad = np.asarray(verdad)
    probabilidad = np.asarray(probabilidad, dtype=float)
    percentiles = np.quantile(probabilidad, np.arange(0.005, 0.9951, 0.005))
    grilla = np.unique(np.concatenate([np.round(np.arange(0.01, 0.9901, 0.01), 2), percentiles]))
    grilla = grilla[grilla > 0]

    prob_pobres = np.sort(probabilidad[verdad == 1])
    prob_no_pobres = np.sort(probabilidad[verdad == 0])
    # Cuántos tienen probabilidad >= u (findInterval con left.open = TRUE en R)
    vp = len(prob_pobres) - np.searchsorted(prob_pobres, grilla, side="left")
    fp = len(prob_no_pobres) - np.searchsorted(prob_no_pobres, grilla, side="left")
    fn = len(prob_pobres) - vp
    vn = len(prob_no_pobres) - fp
    precision = np.where(vp + fp == 0, 0.0, vp / np.maximum(vp + fp, 1))
    recall = np.where(vp + fn == 0, 0.0, vp / np.maximum(vp + fn, 1))
    return pd.DataFrame({
        "umbral": grilla,
        "f1": np.where(vp == 0, 0.0, 2 * vp / np.maximum(2 * vp + fp + fn, 1)),
        "precision": precision, "recall": recall,
        "accuracy": (vp + vn) / len(verdad),
        "filtracion": 1 - precision, "subcobertura": 1 - recall,
        "n_vp": vp, "n_fp": fp, "n_fn": fn, "n_vn": vn,
    })


def buscar_umbral_f1(verdad, probabilidad):
    """Umbral que maximiza el F1 en predicciones fuera de muestra.

    Argumentos: verdad = arreglo 0/1; probabilidad = P(pobre).
    Devuelve: (umbral óptimo, F1 en ese umbral).
    """
    curva = calcular_curva_umbral(verdad, probabilidad)
    mejor = int(np.argmax(curva["f1"].to_numpy()))
    return float(curva["umbral"].iloc[mejor]), float(curva["f1"].iloc[mejor])


def metricas_por_fold(verdad, probabilidad, fold, umbral):
    """Métricas de CV con umbral fijo: media y desviación entre folds.

    Argumentos: verdad, probabilidad, fold = arreglos alineados; umbral = umbral fijo.
    Devuelve: diccionario con f1_cv, f1_cv_sd, precision_cv, recall_cv, accuracy_cv.
    """
    tabla = pd.DataFrame({"verdad": verdad, "probabilidad": probabilidad, "fold": fold})
    por_fold = pd.DataFrame([
        calcular_metricas(g.verdad, (g.probabilidad >= umbral).astype(int))
        for _, g in tabla.groupby("fold")
    ])
    return {
        "f1_cv": por_fold.f1.mean(), "f1_cv_sd": por_fold.f1.std(ddof=1),
        "precision_cv": por_fold.precision.mean(), "recall_cv": por_fold.recall.mean(),
        "accuracy_cv": por_fold.accuracy.mean(),
    }
