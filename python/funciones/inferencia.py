"""Inferencia de la etapa 16: F1 por fold con el umbral de toda la OOF, resumen de
la validación cruzada repetida, comparaciones pareadas con intervalo t, bootstrap
pareado del holdout y efecto mínimo detectable (MDE).

Convenciones:
  - La métrica es la del pipeline: F1 de la clase pobre con el umbral elegido en
    la OOF completa de una partición (buscar_umbral_f1()) y aplicado fijo a cada
    fold; en el holdout, el umbral OOF del candidato ya viene fijado en la clase.
  - "Pareado" quiere decir que la diferencia se calcula dentro de la misma unidad
    (el mismo fold de la misma partición, o la misma remuestra de hogares).
  - MDE = (z_{1-alfa/2} + z_{potencia}) x error estándar, con alfa = 0,05 bilateral
    y potencia 0,80: la diferencia más pequeña que este diseño detectaría con esa
    potencia. En CV se usan cuantiles t con n - 1 grados de libertad.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from configuracion import SEMILLA
from funciones.metricas import buscar_umbral_f1, calcular_metricas

ALFA = 0.05
POTENCIA = 0.80
# Suma de cuantiles normales para el MDE con alfa bilateral 0,05 y potencia 0,80 (~2,80)
MDE_Z = float(norm.ppf(1 - ALFA / 2) + norm.ppf(POTENCIA))
BLOQUE_REPLICAS = 100


def f1_por_fold_con_umbral_oof(oof):
    """Métricas por fold con el umbral elegido en toda la OOF (métrica del pipeline).

    Argumentos: oof = DataFrame con probabilidad, fold y verdad (evaluar_oof()).
    Devuelve: DataFrame con fold, umbral, f1, precision y recall (una fila por fold).
    """
    umbral, _ = buscar_umbral_f1(oof["verdad"], oof["probabilidad"])
    filas = []
    for fold, grupo in oof.groupby("fold", sort=True):
        metricas = calcular_metricas(grupo["verdad"], (grupo["probabilidad"] >= umbral).astype(int))
        filas.append({"fold": int(fold), "umbral": umbral, "f1": metricas["f1"],
                      "precision": metricas["precision"], "recall": metricas["recall"]})
    return pd.DataFrame(filas)


def resumir_cv_repetida(folds):
    """Media, desviación y error estándar del F1 sobre todas las evaluaciones
    (partición x fold) de un candidato.

    Argumentos: folds = DataFrame con particion, fold y f1.
    Devuelve: diccionario con f1_medio, f1_sd, f1_ee y n_evaluaciones.
    """
    f1 = folds["f1"].to_numpy(dtype=float)
    n = len(f1)
    sd = float(np.std(f1, ddof=1)) if n > 1 else np.nan
    return {"f1_medio": float(np.mean(f1)) if n else np.nan, "f1_sd": sd,
            "f1_ee": sd / np.sqrt(n) if n > 1 else np.nan, "n_evaluaciones": int(n)}


def comparar_pareado(folds_candidato, folds_base, razon_prueba_entrenamiento=0.25):
    """Diferencia de F1 candidato - base por fold (misma partición y fold), con
    intervalo t al 95 %, si excluye cero y el efecto mínimo detectable.

    Además del intervalo naive (folds tratados como independientes) reporta el
    corregido por dependencia de Nadeau y Bengio (2003): varianza de la media
    multiplicada por (1/n + n_prueba/n_entrenamiento), que en 5 folds es 1/4.
    Argumentos: folds_candidato, folds_base = DataFrames con particion, fold y f1;
        razon_prueba_entrenamiento = n_prueba / n_entrenamiento de cada fold.
    Devuelve: diccionario con delta_medio, delta_sd, delta_ee, ic_inferior,
        ic_superior, excluye_cero, mde, n (pares usados) y las versiones
        *_corregido del error estándar, el intervalo, excluye_cero y el MDE.
    """
    unido = folds_candidato[["particion", "fold", "f1"]].merge(
        folds_base[["particion", "fold", "f1"]], on=["particion", "fold"], suffixes=("_candidato", "_base"))
    delta = (unido["f1_candidato"] - unido["f1_base"]).to_numpy(dtype=float)
    n = len(delta)
    vacio = {"delta_sd": np.nan, "delta_ee": np.nan, "ic_inferior": np.nan, "ic_superior": np.nan,
             "excluye_cero": False, "mde": np.nan, "delta_ee_corregido": np.nan, "ic_inferior_corregido": np.nan,
             "ic_superior_corregido": np.nan, "excluye_cero_corregido": False, "mde_corregido": np.nan}
    if n < 2:
        return {"delta_medio": float(delta.mean()) if n else np.nan, **vacio, "n": int(n)}
    media, sd = float(delta.mean()), float(delta.std(ddof=1))
    cuantil_ic, cuantil_mde = float(t.ppf(1 - ALFA / 2, n - 1)), float(t.ppf(1 - ALFA / 2, n - 1) + t.ppf(POTENCIA, n - 1))
    salida = {"delta_medio": media, "delta_sd": sd, "n": int(n)}
    for sufijo, ee in (("", sd / np.sqrt(n)), ("_corregido", sd * np.sqrt(1 / n + razon_prueba_entrenamiento))):
        inferior, superior = media - cuantil_ic * ee, media + cuantil_ic * ee
        salida.update({f"delta_ee{sufijo}": ee, f"ic_inferior{sufijo}": inferior, f"ic_superior{sufijo}": superior,
                       f"excluye_cero{sufijo}": bool(inferior > 0 or superior < 0), f"mde{sufijo}": cuantil_mde * ee})
    return salida


def indicadores_f1(verdad, clase):
    """Vectores 0/1 por hogar de verdadero positivo, falso positivo y falso negativo."""
    verdad, clase = np.asarray(verdad, dtype=int), np.asarray(clase, dtype=int)
    return ((verdad == 1) & (clase == 1)).astype(np.int64), ((verdad == 0) & (clase == 1)).astype(np.int64), \
        ((verdad == 1) & (clase == 0)).astype(np.int64)


def f1_desde_conteos(vp, fp, fn):
    """F1 de la clase pobre desde los conteos (0 si no hay verdaderos positivos)."""
    denominador = 2 * vp + fp + fn
    return np.where(denominador > 0, 2 * vp / np.maximum(denominador, 1), 0.0)


def bootstrap_holdout(verdad, clases, base, n_replicas=2000, semilla=SEMILLA):
    """Bootstrap pareado del F1 en el holdout: la misma remuestra de hogares (con
    reemplazo) para todos los candidatos, IC percentil al 95 % del F1 de cada uno
    y de su diferencia con la base, y MDE del holdout.

    Argumentos: verdad = pobre por id (Serie); clases = {candidato: Serie 0/1 por
        id con la clase predicha con su umbral OOF}; base = candidato de
        referencia (debe estar en clases); n_replicas = réplicas; semilla = del
        generador.
    Devuelve: DataFrame con una fila por candidato (en el orden de clases):
        candidato, f1, ic_inferior, ic_superior, delta_observado (diferencia de
        los F1 observados), delta_medio (media bootstrap), delta_sd,
        delta_ic_inferior, delta_ic_superior, delta_excluye_cero, mde_holdout.
    """
    ids = verdad.index
    n = len(ids)
    indicadores = {nombre: indicadores_f1(verdad.to_numpy(), clase.reindex(ids).to_numpy()) for nombre, clase in clases.items()}
    rng = np.random.default_rng(semilla)
    f1_replicas = {nombre: np.empty(n_replicas) for nombre in clases}
    hechas = 0
    while hechas < n_replicas:
        tamano = min(BLOQUE_REPLICAS, n_replicas - hechas)
        posiciones = rng.integers(0, n, size=(tamano, n))
        for nombre, (vp, fp, fn) in indicadores.items():
            f1_replicas[nombre][hechas:hechas + tamano] = f1_desde_conteos(
                vp[posiciones].sum(axis=1), fp[posiciones].sum(axis=1), fn[posiciones].sum(axis=1))
        hechas += tamano
    f1_observado = {nombre: float(f1_desde_conteos(vp.sum(), fp.sum(), fn.sum()))
                    for nombre, (vp, fp, fn) in indicadores.items()}
    filas = []
    for nombre in indicadores:
        delta = f1_replicas[nombre] - f1_replicas[base]
        inferior, superior = (float(np.quantile(delta, q)) for q in (ALFA / 2, 1 - ALFA / 2))
        sd = float(delta.std(ddof=1))
        filas.append({"candidato": nombre, "f1": f1_observado[nombre],
                      "ic_inferior": float(np.quantile(f1_replicas[nombre], ALFA / 2)),
                      "ic_superior": float(np.quantile(f1_replicas[nombre], 1 - ALFA / 2)),
                      "delta_observado": f1_observado[nombre] - f1_observado[base],
                      "delta_medio": float(delta.mean()), "delta_sd": sd, "delta_ic_inferior": inferior,
                      "delta_ic_superior": superior, "delta_excluye_cero": bool(inferior > 0 or superior < 0),
                      "mde_holdout": MDE_Z * sd})
    return pd.DataFrame(filas)
