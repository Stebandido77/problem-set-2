"""Modelos auxiliares de ingreso (LightGBM de regresión) que alimentan variables
del modelo principal a través de predecir_anidado().

  - Personas: log(1 + ingreso laboral individual) con variables de personas
    presentes en test; se agrega al hogar (total, per cápita sobre Npersug y
    per cápita / Lp).
  - Hogar (directo): log(Ingpcug / Lp) con las variables de hogar del nivel 3 + v2.
  - Estrato (etapa 11): el estrato del hogar (Estrato1, 1 a 6) como objetivo
    multiclase; al modelo principal entran el estrato esperado y P(estrato <= 2).
Los ingresos y el estrato solo existen en train y solo se usan como objetivo,
nunca como predictores.
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


# Arriendo imputado del DANE (Ingtotugarr - Ingtotug), verificado contra train en el
# diagnóstico forense: el arriendo estimado (P5130) de las viviendas propias pagadas
# y en usufructo, y en las propias que se están pagando el estimado menos la cuota
# de amortización (P5100) cuando queda positivo. Arrendatarios, posesión sin título
# y otras tenencias no reciben imputación. La regla reproduce la diferencia exacta en
# el 98,4% de los hogares de train (2.694 discrepancias de 164.960).
TENENCIAS_ARRIENDO_COMPLETO = ("propia_pagada", "usufructo")
TENENCIA_PAGANDO = "propia_pagando"
# Si True, agregar_ingreso_hogar() suma el arriendo imputado al ingreso de la unidad
# de gasto antes de dividir, como hace Ingpcug. Queda en False porque en la etapa 11
# la reconciliación no mejoró el clasificador (F1 CV +0,0000, holdout -0,0016 frente
# a la base de 1 semilla); se conserva como opción documentada y comprobada.
ARRIENDO_IMPUTADO_POR_DEFECTO = False


def arriendo_imputado_hogar(hogar):
    """Arriendo imputado que el DANE suma al ingreso de la unidad de gasto.

    Ingpcug, y con él la etiqueta, se calcula con Ingtotugarr = Ingtotug +
    arriendo imputado; los auxiliares suman ingresos personales, así que sin este
    término su per cápita no es comparable con la línea para los propietarios.
    Argumentos: hogar = cargar_hogares() (tenencia, arriendo y P5100).
    Devuelve: Serie por id en pesos por mes (0 donde no aplica).
    """
    tenencia = hogar["tenencia"].astype("string")
    arriendo = hogar["arriendo"].fillna(0)
    cuota = hogar["P5100"].fillna(0)
    imputado = pd.Series(0.0, index=hogar.index)
    imputado[tenencia.isin(TENENCIAS_ARRIENDO_COMPLETO).to_numpy()] = arriendo
    pagando = (tenencia == TENENCIA_PAGANDO).to_numpy()
    imputado[pagando] = (arriendo - cuota).clip(lower=0)
    return imputado.rename("arriendo_imputado")


def derivar_per_capita(por_hogar, hogar, sufijo="", arriendo_imputado=ARRIENDO_IMPUTADO_POR_DEFECTO):
    """Variables per cápita del ingreso agregado del hogar, a partir del total.

    Argumentos: por_hogar = DataFrame indexado por id con ingreso_pred_total<sufijo>;
        hogar = cargar_hogares(); sufijo = el de las variables; arriendo_imputado =
        si True suma al total el arriendo imputado del hogar antes de dividir
        (misma construcción que Ingpcug); si False, solo los ingresos personales.
    Devuelve: copia de por_hogar con log_ingreso_pred_pc<sufijo> (log(1 + per
        cápita)) y log_ingreso_pred_pc_lp<sufijo> (log frente a la línea) recalculadas.
    """
    total = por_hogar[f"ingreso_pred_total{sufijo}"]
    if arriendo_imputado:
        total = total + arriendo_imputado_hogar(hogar).reindex(por_hogar.index).fillna(0)
    n_ug = hogar.loc[por_hogar.index, "personas_ug"].clip(lower=1)
    lp = hogar.loc[por_hogar.index, "linea_pobreza"]
    per_capita = total / n_ug
    salida = por_hogar.copy()
    salida[f"log_ingreso_pred_pc{sufijo}"] = np.log1p(per_capita)
    salida[f"log_ingreso_pred_pc_lp{sufijo}"] = np.log((per_capita + 1) / lp)
    return salida


def agregar_ingreso_hogar(ids, ingreso, en_unidad_gasto, hogar, sufijo="",
                          arriendo_imputado=ARRIENDO_IMPUTADO_POR_DEFECTO):
    """Agrega ingresos individuales al hogar, sobre la unidad de gasto.

    La usan el auxiliar (con ingresos predichos) y el oráculo de la etapa 5 (con
    ingresos observados), para que ambos se agreguen exactamente igual.
    Argumentos: ids = id de hogar de cada persona; ingreso = pesos por mes;
        en_unidad_gasto = booleano por persona; hogar = cargar_hogares();
        sufijo = se agrega al nombre de las variables; arriendo_imputado = si
        True, el per cápita incluye el arriendo imputado del hogar (como Ingpcug).
    Devuelve: DataFrame por hogar con ingreso_pred_total (solo ingresos
        personales), log_ingreso_pred_pc, log_ingreso_pred_pc_lp e ingreso_pred_max_lp.
    """
    personas = pd.DataFrame({"id": ids, "ingreso_ug": np.asarray(ingreso) * np.asarray(en_unidad_gasto)})
    por_hogar = personas.groupby("id").agg(
        ingreso_pred_total=("ingreso_ug", "sum"),
        ingreso_pred_max=("ingreso_ug", "max"),
    )
    por_hogar = derivar_per_capita(por_hogar, hogar, "", arriendo_imputado)
    lp = hogar.loc[por_hogar.index, "linea_pobreza"]
    por_hogar["ingreso_pred_max_lp"] = por_hogar["ingreso_pred_max"] / lp
    return por_hogar.drop(columns="ingreso_pred_max").add_suffix(sufijo)


def crear_auxiliar_personas(personas, ingreso, hogar, columna="ingreso_laboral", sufijo="",
                            parametros=PARAMETROS_AUXILIAR):
    """Función ajustar_predecir() del auxiliar de personas para predecir_anidado().

    Argumentos: personas = cargar_personas(); ingreso = cargar_ingreso_personas();
        hogar = cargar_hogares() (Npersug y línea de pobreza para agregar);
        columna = ingreso objetivo ("ingreso_laboral" o "ingreso_total");
        sufijo = se agrega al nombre de las variables (para usar dos auxiliares);
        parametros = hiperparámetros de LightGBM (los tuneados de la etapa 8 o
        PARAMETROS_AUXILIAR por defecto).
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
        modelo = lgb.train({**parametros, "num_threads": N_HILOS, "seed": SEMILLA}, datos)
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


# ---- Auxiliar de estrato (etapa 11) ---------------------------------------------
ESTRATOS = (1, 2, 3, 4, 5, 6)
# P(estrato <= 2): los dos estratos con más pobreza en train (41% y 22%)
ESTRATO_BAJO_MAXIMO = 2
PARAMETROS_ESTRATO = {
    "objective": "multiclass",
    "num_class": len(ESTRATOS),
    "learning_rate": 0.08,
    "num_iterations": 200,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


def resumir_probabilidades_estrato(probabilidades, ids):
    """Estrato esperado y P(estrato <= ESTRATO_BAJO_MAXIMO) desde las probabilidades
    por clase.

    Argumentos: probabilidades = matriz (hogares x estratos, en el orden de
        ESTRATOS); ids = id de cada fila.
    Devuelve: DataFrame por hogar con estrato_esperado y p_estrato_bajo.
    """
    valores = np.array(ESTRATOS, dtype=float)
    probabilidades = np.asarray(probabilidades)
    return pd.DataFrame({
        "estrato_esperado": probabilidades @ valores,
        "p_estrato_bajo": probabilidades[:, valores <= ESTRATO_BAJO_MAXIMO].sum(axis=1),
    }, index=pd.Index(ids, name="id"))


def crear_auxiliar_estrato(matriz, estrato, parametros=PARAMETROS_ESTRATO):
    """Función ajustar_predecir() del auxiliar de estrato para predecir_anidado().

    Un LightGBM multiclase predice el estrato del hogar (1 a 6, solo train) con
    los predictores de hogar presentes en test; del vector de probabilidades
    salen el estrato esperado y P(estrato <= 2). El estrato nunca entra como
    predictor y los hogares a predecir nunca están en el entrenamiento.
    Argumentos: matriz = predictores de hogar en dummies (armar_matriz());
        estrato = cargar_estrato_hogar(); parametros = hiperparámetros de LightGBM.
    Devuelve: función(ids_entrenamiento, ids_prediccion) -> DataFrame por hogar.
    """
    clase = estrato.map({e: i for i, e in enumerate(ESTRATOS)})

    def ajustar_predecir(ids_entrenamiento, ids_prediccion):
        x_ent, x_pred = imputar_mediana(matriz.loc[ids_entrenamiento], matriz.loc[ids_prediccion])
        y = clase.reindex(ids_entrenamiento)
        assert y.notna().all(), "hogares de entrenamiento sin estrato"
        datos = lgb.Dataset(x_ent, label=y.astype(int).to_numpy())
        modelo = lgb.train({**parametros, "num_threads": N_HILOS, "seed": SEMILLA}, datos)
        return resumir_probabilidades_estrato(modelo.predict(x_pred), ids_prediccion)

    return ajustar_predecir
