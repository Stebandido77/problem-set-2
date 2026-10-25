"""Agregación a lo bruto de las variables de personas al hogar (etapa 7, A).

Toma las variables originales de la encuesta de personas presentes en test y
las lleva al hogar por rol (jefe, cónyuge, resto de miembros, todos):
  - numéricas: valor del jefe y del cónyuge; media, máximo, mínimo, suma y
    desviación estándar del resto y de todos;
  - categóricas: indicadora de cada categoría para jefe y cónyuge; conteo y
    proporción de cada categoría en el resto y en todos.
Solo usa predictores (nunca ingresos ni la etiqueta), así que se calcula una vez
para train y test sin fuga. Las columnas casi constantes o repetidas se podan.
"""

import re

import numpy as np
import pandas as pd

# Variables originales de personas: códigos de la encuesta (P...), Oficio y las
# indicadoras de fuerza de trabajo. Quedan fuera las llaves (id, Orden), las
# constantes dentro del hogar (Clase, Dominio, Depto) y los factores de expansión
PATRON_ORIGINALES = r"^(P\d|Pet$|Oc$|Des$|Ina$|Oficio$)"
# Una variable con a lo sumo estos valores distintos se trata como categórica
MAXIMO_NIVELES_CATEGORICA = 10
CATEGORICAS_FORZADAS = ["Oficio"]
# Categorías con menos de esta proporción de personas se juntan en "otros"
PROPORCION_MINIMA_NIVEL = 0.005
# Una columna cuyo valor más común cubre más de esta proporción de hogares se poda
PROPORCION_MAXIMA_MODA = 0.999
PREFIJO = "ag_"
ESTADISTICAS = {"media": "mean", "max": "max", "min": "min", "suma": "sum"}


def variables_originales(personas):
    """Variables de personas que entran a la agregación (56 de las 63 del test).

    Argumentos: personas = cargar_personas().
    Devuelve: (numericas, categoricas), listas de nombres.
    """
    variables = [c for c in personas.columns if re.match(PATRON_ORIGINALES, c)]
    categoricas = [c for c in variables
                   if c in CATEGORICAS_FORZADAS or personas[c].nunique() <= MAXIMO_NIVELES_CATEGORICA]
    numericas = [c for c in variables if c not in categoricas]
    return numericas, categoricas


def asignar_rol(parentesco):
    """Rol de cada persona según P6050: 1 jefe, 2 cónyuge, el resto 'resto'."""
    return np.select([parentesco == 1, parentesco == 2], ["jefe", "conyuge"], "resto")


def abrir_categoricas(personas, categoricas):
    """Indicadoras (0/1) de cada categoría, con NA como categoría propia y las
    categorías raras juntas en 'otros'.

    Argumentos: personas = cargar_personas(); categoricas = nombres.
    Devuelve: DataFrame float32 alineado con personas.
    """
    partes = []
    for variable in categoricas:
        valores = personas[variable].astype("Int64").astype("string").fillna("NA")
        frecuencia = valores.value_counts(normalize=True)
        raras = frecuencia.index[frecuencia < PROPORCION_MINIMA_NIVEL]
        valores = valores.where(~valores.isin(raras), "otros")
        partes.append(pd.get_dummies(valores, prefix=variable, dtype=np.float32))
    return pd.concat(partes, axis=1)


def agregar_por_rol(numericas, indicadoras, ids, rol, hogares):
    """Agregados de un rol, reindexados a todos los hogares.

    Argumentos: numericas, indicadoras = DataFrames por persona; ids = id de hogar
        de cada persona; rol = 'jefe', 'conyuge', 'resto' o 'todos'; hogares =
        índice con todos los hogares.
    Devuelve: DataFrame por hogar con columnas ag_<rol>_<variable>_<estadística>.
    """
    if rol in ("jefe", "conyuge"):
        # Una persona por hogar (21 hogares tienen dos cónyuges: se toma el primero)
        valor = numericas.groupby(ids).first().add_suffix("_valor")
        indicadora = indicadoras.groupby(ids).first().reindex(hogares).fillna(0)
        salida = pd.concat([valor.reindex(hogares), indicadora], axis=1)
    else:
        grupos = numericas.groupby(ids)
        partes = [grupos.agg(funcion).add_suffix(f"_{nombre}") for nombre, funcion in ESTADISTICAS.items()]
        partes.append(grupos.std(ddof=0).add_suffix("_desv"))
        conteo = indicadoras.groupby(ids).sum()
        proporcion = conteo.div(ids.groupby(ids).size(), axis=0)
        partes += [conteo.add_suffix("_conteo"), proporcion.add_suffix("_prop")]
        salida = pd.concat(partes, axis=1).reindex(hogares)
        conteos = [c for c in salida.columns if c.endswith("_conteo")]
        salida[conteos] = salida[conteos].fillna(0)  # sin personas en el rol = conteo 0
    return salida.add_prefix(f"{PREFIJO}{rol}_")


def podar_columnas(agregados):
    """Quita columnas casi constantes y columnas idénticas a otra.

    No usa la etiqueta: solo la distribución de cada columna en train + test.
    Argumentos: agregados = DataFrame por hogar.
    Devuelve: DataFrame podado.
    """
    moda = agregados.apply(lambda c: c.value_counts(dropna=False, normalize=True).iloc[0])
    agregados = agregados.loc[:, moda <= PROPORCION_MAXIMA_MODA]
    huellas = agregados.apply(lambda c: pd.util.hash_pandas_object(c, index=False).sum())
    return agregados.loc[:, ~huellas.duplicated()]


def construir_agregados(personas, hogares):
    """Agregados por rol de todas las variables originales de personas.

    Argumentos: personas = cargar_personas(); hogares = índice de ids de todos
        los hogares (train + test).
    Devuelve: DataFrame float32 indexado por id, con columnas ag_*.
    """
    numericas, categoricas = variables_originales(personas)
    rol = asignar_rol(personas["P6050"].to_numpy())
    x_numerico = personas[numericas].astype(np.float32)
    indicadoras = abrir_categoricas(personas, categoricas)
    ids = personas["id"]
    partes = []
    for nombre_rol in ("jefe", "conyuge", "resto", "todos"):
        filas = np.ones(len(personas), bool) if nombre_rol == "todos" else rol == nombre_rol
        partes.append(agregar_por_rol(x_numerico[filas], indicadoras[filas], ids[filas], nombre_rol, hogares))
    agregados = pd.concat(partes, axis=1).astype(np.float32)
    agregados.index.name = "id"
    return podar_columnas(agregados)
