"""Validación cruzada repetida (etapa 16): particiones nuevas del 80 % que no
tocan el holdout, contextos solo de CV, auxiliares anidados y OOF reanudables
guardados en la carpeta de cada partición.

El protocolo es el de siempre (funciones/anidado.py y funciones/modelado.py),
pero con los grupos de una partición distinta de la de R y sin los contextos
holdout y test: la CV repetida mide solo la OOF del 80 %; el holdout se evalúa
con las predicciones del pipeline original.
"""

import gc
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from configuracion import N_FOLDS, SEMILLA
from funciones.agregados import PREFIJO as PREFIJO_AGREGADOS
from funciones.anidado import (GRUPO_HOLDOUT, GRUPO_TEST, GRUPOS_80, asignar_grupos, guardar_parquet_atomico,
                               predecir_anidado, verificar_sin_fuga)
from funciones.bitacora import registrar_log

# Semillas de las particiones repetidas (distintas de la de R, SEMILLA)
SEMILLAS_PARTICIONES = {1: SEMILLA + 100, 2: SEMILLA + 200, 3: SEMILLA + 300}
# Sufijo de las 10 variables del auxiliar de fuentes (etapa 6)
SUFIJO_FUENTES = "_v3"


def generar_particion_repetida(particion, semilla):
    """Grupo de cada hogar con una partición nueva del 80 % en N_FOLDS folds
    estratificados por pobre; el holdout (6) y el test (0) quedan como en R.

    Argumentos: particion = cargar_particion(); semilla = del StratifiedKFold.
    Devuelve: Serie de enteros indexada por id (como asignar_grupos()).
    """
    grupos = asignar_grupos(particion).copy()
    en_80 = grupos.isin(GRUPOS_80).to_numpy()
    ids_80 = grupos.index[en_80]
    y = particion.set_index("id").loc[ids_80, "pobre"].astype(int).to_numpy()
    nuevos = np.zeros(len(ids_80), dtype=int)
    for k, (_, validacion) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=semilla)
                                        .split(np.zeros(len(y)), y), 1):
        nuevos[validacion] = k
    grupos.loc[ids_80] = nuevos
    return grupos.astype(int)


def definir_contextos_cv():
    """Contextos de la CV repetida: solo cv_k (entrena con los folds != k y evalúa
    en k); sin holdout ni test.

    Devuelve: diccionario {contexto: (grupos_entrenamiento, grupos_evaluacion)}.
    """
    return {f"cv_{k}": (tuple(g for g in GRUPOS_80 if g != k), (k,)) for k in GRUPOS_80}


def anidado_en_carpeta(nombre, ajustar_predecir, grupos, carpeta, contextos):
    """Variables de un auxiliar por contexto con OOF anidada, con caché y parciales
    en la carpeta de una partición (como cargar_o_calcular_anidado(), pero con los
    grupos y contextos que se le pasen).

    Argumentos: nombre = etiqueta del auxiliar; ajustar_predecir = función del
        auxiliar (ids_entrenamiento, ids_prediccion) -> DataFrame por id; grupos =
        generar_particion_repetida(); carpeta = carpeta de la partición; contextos =
        definir_contextos_cv().
    Devuelve: diccionario {contexto: DataFrame indexado por id}, verificado sin fuga.
    """
    ruta = carpeta / f"anidado_{nombre}.parquet"
    ruta_registro = carpeta / f"registro_{nombre}.parquet"
    if ruta.exists() and ruta_registro.exists():
        tabla, registro = pd.read_parquet(ruta), pd.read_parquet(ruta_registro)
    else:
        ruta_parcial = carpeta / "parciales" / nombre
        variables, registro = predecir_anidado(ajustar_predecir, grupos, nombre, contextos, ruta_parcial=ruta_parcial)
        tabla = pd.concat([v.assign(contexto=c) for c, v in variables.items()]).rename_axis("id").reset_index()
        carpeta.mkdir(parents=True, exist_ok=True)
        guardar_parquet_atomico(registro, ruta_registro)
        guardar_parquet_atomico(tabla, ruta)
        shutil.rmtree(ruta_parcial, ignore_errors=True)
    verificar_sin_fuga(registro, grupos, contextos)
    return {c: g.drop(columns="contexto").set_index("id") for c, g in tabla.groupby("contexto")}


def ruta_oof(nombre, carpeta):
    """Parquet con la OOF de un candidato en la carpeta de una partición."""
    return carpeta / "predicciones" / f"{nombre}.parquet"


def guardar_oof(nombre, oof, carpeta):
    """Escribe la OOF (probabilidad, fold, verdad por id) de un candidato."""
    ruta = ruta_oof(nombre, carpeta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    guardar_parquet_atomico(oof.rename_axis("id"), ruta)


def cargar_oof(nombre, carpeta):
    """OOF guardada de un candidato (None si todavía no existe)."""
    ruta = ruta_oof(nombre, carpeta)
    return pd.read_parquet(ruta) if ruta.exists() else None


def evaluar_oof(nombre, preparar, ajustar, verdad, carpeta, contextos=None):
    """OOF de un candidato en los contextos de CV de una partición, reanudable por
    fold (cada fold se guarda al terminar en carpeta/parciales/<nombre>/ y se lee
    si ya existe); al terminar guarda la OOF y borra los parciales.

    Argumentos: nombre = candidato; preparar = función(contexto) -> (x_ent, y_ent,
        x_eval); ajustar = función(x_ent, y_ent, x_eval) -> (probabilidad, importancia
        o None); verdad = pobre por id; carpeta = carpeta de la partición;
        contextos = definir_contextos_cv() por defecto.
    Devuelve: DataFrame indexado por id con probabilidad, fold y verdad.
    """
    inicio = time.time()
    registrar_log(nombre, "inicio OOF")
    ruta_parcial = carpeta / "parciales" / nombre
    partes = []
    for k, contexto in enumerate(contextos or definir_contextos_cv(), 1):
        parcial = ruta_parcial / f"{contexto}.parquet"
        if parcial.exists():
            probabilidad = pd.read_parquet(parcial)["probabilidad"]
            registrar_log(nombre, f"{contexto} desde caché parcial")
        else:
            x_ent, y_ent, x_eval = preparar(contexto)
            valores, _ = ajustar(x_ent, y_ent, x_eval)
            probabilidad = pd.Series(np.asarray(valores, dtype=float), index=x_eval.index, name="probabilidad")
            del x_ent, x_eval
            gc.collect()
            ruta_parcial.mkdir(parents=True, exist_ok=True)
            guardar_parquet_atomico(probabilidad.to_frame(), parcial)
            registrar_log(nombre, f"fin {contexto}", inicio)
        partes.append(pd.DataFrame({"probabilidad": probabilidad, "fold": np.int64(k)}))
    oof = pd.concat(partes)
    oof["verdad"] = verdad.loc[oof.index].astype(np.int64).to_numpy()
    oof.index.name = "id"
    guardar_oof(nombre, oof, carpeta)
    shutil.rmtree(ruta_parcial, ignore_errors=True)
    registrar_log(nombre, "OOF guardada", inicio)
    return oof


def promediar_oof(nombre, nombres, pesos, carpeta):
    """Combinación ponderada de las OOF guardadas de varios candidatos, alineada
    por id con la del primero; se guarda como un candidato más.

    Argumentos: nombre = candidato nuevo; nombres = candidatos a combinar; pesos =
        pesos (suman 1) en el mismo orden; carpeta = carpeta de la partición.
    Devuelve: DataFrame indexado por id con probabilidad, fold y verdad.
    """
    assert abs(sum(pesos) - 1) < 1e-9, "los pesos deben sumar 1"
    partes = [cargar_oof(n, carpeta) for n in nombres]
    assert all(p is not None for p in partes), f"falta la OOF de alguno de {nombres}"
    primera = partes[0]
    probabilidad = sum(w * p["probabilidad"].reindex(primera.index) for w, p in zip(pesos, partes))
    assert probabilidad.notna().all(), "las OOF a combinar no tienen los mismos hogares"
    oof = pd.DataFrame({"probabilidad": probabilidad.to_numpy(), "fold": primera["fold"].to_numpy(),
                        "verdad": primera["verdad"].to_numpy()}, index=primera.index)
    oof.index.name = "id"
    guardar_oof(nombre, oof, carpeta)
    return oof


def columnas_v3(columnas_a):
    """Columnas de la base n7_v3 (a4 + auxiliares laboral, total y fuentes): las de
    A sin los agregados por rol (ag_*), en el mismo orden."""
    return [c for c in columnas_a if not c.startswith(PREFIJO_AGREGADOS)]


def columnas_a4_total(columnas_a):
    """Columnas de la base n7_n6param_a4_total (a4 + auxiliares laboral y total):
    las de n7_v3 sin las 10 variables del auxiliar de fuentes (*_v3)."""
    return [c for c in columnas_v3(columnas_a) if not c.endswith(SUFIJO_FUENTES)]


__all__ = ["SEMILLAS_PARTICIONES", "GRUPO_HOLDOUT", "GRUPO_TEST", "generar_particion_repetida",
           "definir_contextos_cv", "anidado_en_carpeta", "evaluar_oof", "cargar_oof", "guardar_oof",
           "promediar_oof", "columnas_v3", "columnas_a4_total"]
