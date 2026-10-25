"""Carga de las bases que exporta scripts/11_exportar_python.R y réplica de la
receta de R (crear_receta()) para que todos los candidatos del nivel 7 usen el
mismo preprocesamiento."""

import numpy as np
import pandas as pd

from configuracion import RUTA_PROCESADOS


def cargar_particion():
    """Partición de R: id, pobre, fold_id (1-5; NA en holdout y test), holdout, base.

    Devuelve: DataFrame con una fila por hogar (train y test).
    """
    particion = pd.read_parquet(RUTA_PROCESADOS / "particion_nivel7.parquet")
    particion["fold_id"] = particion["fold_id"].astype("Int64")
    return particion


def cargar_hogares():
    """Base de hogar del nivel 3 (train + test) con las columnas crudas extra.

    Devuelve: DataFrame indexado por id, con las categóricas como 'category'.
    """
    hogar = pd.read_parquet(RUTA_PROCESADOS / "hogar_nivel7.parquet").set_index("id")
    for columna in hogar.columns:
        if hogar[columna].dtype == object or str(hogar[columna].dtype) == "string":
            hogar[columna] = hogar[columna].astype("category")
    return hogar


def cargar_personas():
    """Base de personas de train y test (columnas crudas comunes + limpias de R).

    Devuelve: DataFrame con una fila por persona.
    """
    return pd.read_parquet(RUTA_PROCESADOS / "personas_nivel7.parquet")


def cargar_ingreso_personas():
    """Ingreso laboral individual de train (objetivo del modelo auxiliar).

    Suma el ingreso monetario de la primera y segunda actividad, en especie y de
    desocupados/inactivos, más sus versiones imputadas por el DANE (*es). Donde
    hay imputación el valor observado es 0, así que se suman sin contar doble.
    El ingreso total es Ingtot = Ingtotob + Ingtotes: laboral + pensiones,
    arriendos, transferencias y demás fuentes, con sus imputaciones.
    Devuelve: DataFrame con id, Orden, ingreso_laboral e ingreso_total (pesos por mes).
    """
    ingreso = pd.read_parquet(RUTA_PROCESADOS / "ingreso_personas_train.parquet")
    componentes = ["Impa", "Isa", "Ie", "Imdi", "Impaes", "Isaes", "Iees", "Imdies"]
    ingreso["ingreso_laboral"] = ingreso[componentes].fillna(0).sum(axis=1)
    ingreso["ingreso_total"] = ingreso["Ingtot"].fillna(0)
    return ingreso[["id", "Orden", "ingreso_laboral", "ingreso_total"]]


# Pesos por mes por debajo de los cuales "otros" se considera residuo de redondeo
RESIDUO_MINIMO_OTROS = 1000


def cargar_ingreso_fuentes():
    """Ingreso individual de train separado por fuente (objetivo del auxiliar v3).

    laboral = como en cargar_ingreso_personas(); pensiones = Iof2; arriendos =
    Iof6; transferencias = ayudas de otros hogares e instituciones (Iof3h + Iof3i);
    otros = lo que falta para llegar a Ingtot (intereses Iof1, cesantías, primas,
    imputaciones). Las cinco suman Ingtot, salvo que "otros" por debajo de
    RESIDUO_MINIMO_OTROS se lleva a cero: el 22% de sus valores positivos son
    residuos de la resta (centavos), que en log tienen una desviación de ~14 y
    disparan el modelo de dos partes.
    Devuelve: DataFrame con id, Orden y una columna por fuente (pesos por mes).
    """
    ingreso = pd.read_parquet(RUTA_PROCESADOS / "ingreso_personas_train.parquet")
    componentes = ["Impa", "Isa", "Ie", "Imdi", "Impaes", "Isaes", "Iees", "Imdies"]
    fuentes = pd.DataFrame({
        "laboral": ingreso[componentes].fillna(0).sum(axis=1),
        "pensiones": ingreso["Iof2"].fillna(0),
        "arriendos": ingreso["Iof6"].fillna(0),
        "transferencias": ingreso["Iof3h"].fillna(0) + ingreso["Iof3i"].fillna(0),
    })
    otros = ingreso["Ingtot"].fillna(0) - fuentes.sum(axis=1)
    fuentes["otros"] = otros.where(otros >= RESIDUO_MINIMO_OTROS, 0)
    return pd.concat([ingreso[["id", "Orden"]], fuentes], axis=1)


def cargar_ingreso_hogar():
    """Ingreso per cápita de la unidad de gasto (solo train) y su razón con Lp.

    Devuelve: Serie log(Ingpcug / Lp) indexada por id (pobre <=> valor < 0).
    """
    hogar = pd.read_parquet(RUTA_PROCESADOS / "ingreso_hogar_train.parquet").set_index("id")
    return np.log(np.maximum(hogar["Ingpcug"], 1) / hogar["Lp"]).rename("log_ingreso_relativo")


def preparar_matriz(variables, nativas=()):
    """Pasa las categóricas a dummies con los mismos niveles en todas las bases.

    Como step_dummy() + step_novel() en R: el conjunto de columnas es fijo y una
    categoría nueva queda en ceros. La imputación se hace aparte, por ajuste.
    Argumentos: variables = DataFrame de predictores indexado por id;
        nativas = categóricas que se dejan como 'category' para que LightGBM
        las trate de forma nativa en vez de abrirlas en dummies.
    Devuelve: DataFrame con las dummies en float32 y las nativas como 'category'.
    """
    categoricas = [c for c in variables.columns
                   if isinstance(variables[c].dtype, pd.CategoricalDtype) and c not in nativas]
    matriz = pd.get_dummies(variables, columns=categoricas, dummy_na=False, dtype=np.float32)
    matriz.columns = [str(c).replace(" ", "_") for c in matriz.columns]
    numericas = [c for c in matriz.columns if c not in nativas]
    matriz[numericas] = matriz[numericas].astype(np.float32)
    return matriz


def imputar_mediana(entrenamiento, *otras):
    """Imputa NA con la mediana del entrenamiento (step_impute_median en R).

    Las dummies no tienen NA; la moda de las categóricas ya quedó resuelta porque
    un NA categórico es una fila de ceros.
    Argumentos: entrenamiento = matriz de ajuste; otras = matrices a predecir.
    Devuelve: lista con las matrices imputadas, en el mismo orden.
    """
    medianas = entrenamiento.median(skipna=True, numeric_only=True).fillna(0)
    return [m.fillna(medianas) for m in (entrenamiento, *otras)]


def cargar_estrato_hogar():
    """Estrato del hogar (Estrato1 del DANE, 1 a 6; solo train; objetivo del
    auxiliar de la etapa 11, nunca predictor).

    Devuelve: Serie de enteros indexada por id.
    """
    estrato = pd.read_parquet(RUTA_PROCESADOS / "estrato_hogar_train.parquet").set_index("id")["estrato"]
    return estrato.astype(int).rename("estrato")
