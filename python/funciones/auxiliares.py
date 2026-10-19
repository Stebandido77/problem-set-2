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


# ---- Auxiliar v3 (etapa 6) -----------------------------------------------------
FUENTES_V3 = ["laboral", "pensiones", "arriendos", "transferencias", "otros"]
CUANTILES_V3 = (0.1, 0.25, 0.5, 0.75, 0.9)


def agregar_fuentes_hogar(ids, ingresos, en_unidad_gasto, hogar):
    """Agrega al hogar el ingreso esperado de cada fuente y el total.

    Argumentos: ids = id de hogar de cada persona; ingresos = DataFrame por persona
        con una columna por fuente (pesos por mes); en_unidad_gasto = booleano por
        persona; hogar = cargar_hogares().
    Devuelve: DataFrame por hogar con <fuente>_pc_lp_v3 (per cápita / Lp) de cada
        fuente, las variables de agregar_ingreso_hogar() del total (sufijo _v3) y
        n_perceptores_v3 (personas con ingreso esperado mayor a medio Lp).
    """
    ug = np.asarray(en_unidad_gasto)
    por_persona = ingresos.mul(ug, axis=0).assign(id=np.asarray(ids))
    suma = por_persona.groupby("id").sum()
    n_ug = hogar.loc[suma.index, "personas_ug"].clip(lower=1)
    lp = hogar.loc[suma.index, "linea_pobreza"]
    salida = suma.div(n_ug * lp, axis=0).add_suffix("_pc_lp_v3")
    total = ingresos.sum(axis=1).to_numpy()
    salida = salida.join(agregar_ingreso_hogar(ids, total, ug, hogar, "_v3"))
    lp_persona = hogar.loc[np.asarray(ids), "linea_pobreza"].to_numpy()
    perceptor = pd.Series((total * ug > 0.5 * lp_persona).astype(float), index=np.asarray(ids))
    salida["n_perceptores_v3"] = perceptor.groupby(level=0).sum()
    return salida


def crear_auxiliar_fuentes(personas, fuentes, hogar, contexto_hogar):
    """Función ajustar_predecir() del auxiliar v3: dos partes por fuente.

    Para cada fuente, un clasificador da P(ingreso > 0) y una regresión sobre los
    que reciben da E[log ingreso | > 0]; el esperado es P * exp(media) * factor de
    smearing de Duan (media de exp(residuo) en el entrenamiento). A los predictores
    de la persona se les pegan las variables de su hogar (contexto_hogar).
    Argumentos: personas = cargar_personas(); fuentes = cargar_ingreso_fuentes();
        hogar = cargar_hogares(); contexto_hogar = DataFrame de variables de hogar
        indexado por id (sin ingresos).
    Devuelve: función(ids_entrenamiento, ids_prediccion) -> DataFrame por hogar.
    """
    contexto = contexto_hogar.reindex(personas["id"].to_numpy()).add_prefix("hogar_")
    x = pd.concat([preparar_personas_auxiliar(personas).reset_index(drop=True),
                   contexto.reset_index(drop=True)], axis=1)
    objetivo = personas[["id", "Orden"]].merge(fuentes, on=["id", "Orden"], how="left")[FUENTES_V3]
    en_unidad_gasto = ~personas["P6050"].isin(FUERA_UNIDAD_GASTO).to_numpy()
    id_persona = personas["id"]
    base = {**PARAMETROS_AUXILIAR, "num_threads": N_HILOS, "seed": SEMILLA}

    def ajustar_predecir(ids_entrenamiento, ids_prediccion):
        filas_ent = id_persona.isin(ids_entrenamiento).to_numpy()
        filas_pred = id_persona.isin(ids_prediccion).to_numpy()
        assert not (filas_ent & filas_pred).any(), "una persona está en entrenamiento y predicción"
        assert objetivo[filas_ent].notna().all().all(), "personas de entrenamiento sin ingreso"
        x_ent, x_pred = x[filas_ent], x[filas_pred]
        esperado = {}
        for fuente in FUENTES_V3:
            y = objetivo.loc[filas_ent, fuente].to_numpy()
            recibe = y > 0
            clasificador = lgb.train({**base, "objective": "binary"},
                                     lgb.Dataset(x_ent, label=recibe.astype(int)))
            regresion = lgb.train(base, lgb.Dataset(x_ent[recibe], label=np.log(y[recibe])))
            residuo = np.log(y[recibe]) - regresion.predict(x_ent[recibe])
            smearing = np.mean(np.exp(residuo))
            esperado[fuente] = clasificador.predict(x_pred) * np.exp(regresion.predict(x_pred)) * smearing
        return agregar_fuentes_hogar(id_persona.to_numpy()[filas_pred], pd.DataFrame(esperado),
                                     en_unidad_gasto[filas_pred], hogar)

    return ajustar_predecir


def crear_auxiliar_cuantiles(matriz, objetivo):
    """Función ajustar_predecir() de la regresión por cuantiles de log(Ingpcug / Lp).

    Un LightGBM con pérdida cuantil por cada cuantil de CUANTILES_V3; además de
    los cuantiles devuelve el ancho intercuartil y cuántos cuantiles quedan bajo la
    línea (una aproximación de P(pobre) desde la distribución del ingreso).
    Argumentos: matriz = predictores de hogar en dummies; objetivo =
        cargar_ingreso_hogar() (solo train).
    Devuelve: función(ids_entrenamiento, ids_prediccion) -> DataFrame por hogar.
    """
    def ajustar_predecir(ids_entrenamiento, ids_prediccion):
        x_ent, x_pred = imputar_mediana(matriz.loc[ids_entrenamiento], matriz.loc[ids_prediccion])
        datos = lgb.Dataset(x_ent, label=objetivo.loc[ids_entrenamiento].to_numpy(), free_raw_data=False)
        salida = pd.DataFrame(index=pd.Index(ids_prediccion, name="id"))
        for cuantil in CUANTILES_V3:
            modelo = lgb.train({**PARAMETROS_AUXILIAR, "objective": "quantile", "alpha": cuantil,
                                "num_iterations": 400, "num_leaves": 63, "learning_rate": 0.05,
                                "num_threads": N_HILOS, "seed": SEMILLA}, datos)
            salida[f"cuantil_{int(cuantil * 100)}_v3"] = modelo.predict(x_pred)
        # Los cuantiles se ajustan por separado y pueden cruzarse: se ordenan
        salida[:] = np.sort(salida.to_numpy(), axis=1)
        salida["ancho_intercuartil_v3"] = salida["cuantil_75_v3"] - salida["cuantil_25_v3"]
        salida["cuantiles_bajo_linea_v3"] = (salida.filter(like="cuantil_") < 0).sum(axis=1)
        return salida

    return ajustar_predecir
