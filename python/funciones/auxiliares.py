"""Modelos auxiliares de ingreso (LightGBM de regresión) que alimentan variables
del modelo principal a través de predecir_anidado().

  - Personas: log(1 + ingreso laboral individual) con variables de personas
    presentes en test; se agrega al hogar (total, per cápita sobre Npersug y
    per cápita / Lp).
  - Hogar (directo): log(Ingpcug / Lp) con las variables de hogar del nivel 3 + v2.
Los ingresos solo existen en train y solo se usan como objetivo, nunca como
predictores.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from configuracion import N_HILOS, SEMILLA
from funciones.datos import imputar_mediana

PARAMETROS_AUXILIAR = {
    "objective": "regression",
    "learning_rate": 0.08,
    "num_iterations": 500,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

# Columnas de personas que no entran al auxiliar: llaves y factores de expansión
EXCLUIDAS_PERSONAS = ["id", "Orden", "Fex_c", "Fex_dpto"]
CATEGORICAS_PERSONAS = ["Dominio", "Depto", "P6050", "P6100", "P6240", "P6430",
                        "P6870", "P6920", "Oficio", "posicion", "oficio_grupo"]
# Parentescos fuera de la unidad de gasto: servicio doméstico, pensionistas, trabajadores
FUERA_UNIDAD_GASTO = [6, 7, 8]


def preparar_personas_auxiliar(personas):
    """Matriz de predictores a nivel persona para el auxiliar.

    Argumentos: personas = cargar_personas().
    Devuelve: DataFrame de predictores (categóricas como 'category').
    """
    x = personas.drop(columns=EXCLUIDAS_PERSONAS)
    for columna in CATEGORICAS_PERSONAS:
        x[columna] = x[columna].astype("string").fillna("NA").astype("category")
    return x


def agregar_ingreso_hogar(ids, ingreso, en_unidad_gasto, hogar, sufijo=""):
    """Agrega ingresos individuales al hogar, sobre la unidad de gasto.

    La usan el auxiliar (con ingresos predichos) y el oráculo de la etapa 5 (con
    ingresos observados), para que ambos se agreguen exactamente igual.
    Argumentos: ids = id de hogar de cada persona; ingreso = pesos por mes;
        en_unidad_gasto = booleano por persona; hogar = cargar_hogares();
        sufijo = se agrega al nombre de las variables.
    Devuelve: DataFrame por hogar con ingreso_pred_total, log_ingreso_pred_pc,
        log_ingreso_pred_pc_lp e ingreso_pred_max_lp.
    """
    personas = pd.DataFrame({"id": ids, "ingreso_ug": np.asarray(ingreso) * np.asarray(en_unidad_gasto)})
    por_hogar = personas.groupby("id").agg(
        ingreso_pred_total=("ingreso_ug", "sum"),
        ingreso_pred_max=("ingreso_ug", "max"),
    )
    n_ug = hogar.loc[por_hogar.index, "personas_ug"].clip(lower=1)
    lp = hogar.loc[por_hogar.index, "linea_pobreza"]
    per_capita = por_hogar["ingreso_pred_total"] / n_ug
    por_hogar["log_ingreso_pred_pc"] = np.log1p(per_capita)
    por_hogar["log_ingreso_pred_pc_lp"] = np.log((per_capita + 1) / lp)
    por_hogar["ingreso_pred_max_lp"] = por_hogar["ingreso_pred_max"] / lp
    return por_hogar.drop(columns="ingreso_pred_max").add_suffix(sufijo)


def crear_auxiliar_personas(personas, ingreso, hogar, columna="ingreso_laboral", sufijo=""):
    """Función ajustar_predecir() del auxiliar de personas para predecir_anidado().

    Argumentos: personas = cargar_personas(); ingreso = cargar_ingreso_personas();
        hogar = cargar_hogares() (Npersug y línea de pobreza para agregar);
        columna = ingreso objetivo ("ingreso_laboral" o "ingreso_total");
        sufijo = se agrega al nombre de las variables (para usar dos auxiliares).
    Devuelve: función(ids_entrenamiento, ids_prediccion) -> DataFrame por hogar.
    """
    x = preparar_personas_auxiliar(personas)
    llave = personas[["id", "Orden"]]
    objetivo = llave.merge(ingreso, on=["id", "Orden"], how="left")[columna]
    y = np.log1p(objetivo.to_numpy())
    en_unidad_gasto = ~personas["P6050"].isin(FUERA_UNIDAD_GASTO).to_numpy()
    id_persona = personas["id"]

    def ajustar_predecir(ids_entrenamiento, ids_prediccion):
        # Series.isin usa hash; np.isin sobre ids de texto es cuadrático
        filas_ent = id_persona.isin(ids_entrenamiento).to_numpy()
        filas_pred = id_persona.isin(ids_prediccion).to_numpy()
        assert not np.isnan(y[filas_ent]).any(), "personas de entrenamiento sin ingreso"
        datos = lgb.Dataset(x[filas_ent], label=y[filas_ent], free_raw_data=True)
        modelo = lgb.train({**PARAMETROS_AUXILIAR, "num_threads": N_HILOS, "seed": SEMILLA}, datos)
        ingreso_predicho = np.expm1(modelo.predict(x[filas_pred]))
        return agregar_ingreso_hogar(id_persona.to_numpy()[filas_pred], ingreso_predicho,
                                     en_unidad_gasto[filas_pred], hogar, sufijo)

    return ajustar_predecir


def crear_auxiliar_hogar(matriz, objetivo):
    """Función ajustar_predecir() del auxiliar directo de hogar.

    Argumentos: matriz = predictores de hogar ya en dummies (preparar_matriz());
        objetivo = cargar_ingreso_hogar() (log(Ingpcug / Lp), solo train).
    Devuelve: función(ids_entrenamiento, ids_prediccion) -> DataFrame por hogar.
    """
    def ajustar_predecir(ids_entrenamiento, ids_prediccion):
        x_ent, x_pred = imputar_mediana(matriz.loc[ids_entrenamiento], matriz.loc[ids_prediccion])
        datos = lgb.Dataset(x_ent, label=objetivo.loc[ids_entrenamiento].to_numpy())
        modelo = lgb.train({**PARAMETROS_AUXILIAR, "num_iterations": 600, "num_leaves": 63,
                            "learning_rate": 0.05, "num_threads": N_HILOS, "seed": SEMILLA}, datos)
        return pd.DataFrame({"log_ingreso_relativo_pred": modelo.predict(x_pred)},
                            index=pd.Index(ids_prediccion, name="id"))

    return ajustar_predecir
