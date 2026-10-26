"""Etapa 18: combinación del modelo limpio (limpio/, etapa 17) con el mejor modelo
del nivel 7, con el protocolo de la etapa 16.

Candidatos:
  n7_a_agregados_bag5      base (OOF repetidas de la etapa 16, holdout del pipeline)
  n13_mezcla_final         referencia (ídem)
  n18_limpio               LightGBM limpio de la etapa 17 (74 variables + columna
                           auxiliar de ingreso anidada), exportado por
                           limpio/07_exportar_etapa18.R
  n18_union_bag5           base de A + variables del modelo limpio que A no tiene
                           (incluida la columna auxiliar, anidada por contexto),
                           LightGBM n6 con bagging de 5 semillas
  n18_a_corregida_bag5     base de A con los códigos 9 (sí/no) y 98/99 pasados a
                           NA antes de agregar, más el mes derivado de Lp
  n18_mezcla_a_limpio      n7_a + peso w x limpio
  n18_mezcla_n13_limpio    n13 + peso w x limpio
Los pesos de las mezclas salen SOLO de las OOF de la CV repetida: para evaluar el
fold k de la partición r se elige w en los otros 4 folds de esa partición
(ajuste cruzado, el fold evaluado nunca elige su peso); el peso final (holdout y
test) maximiza el F1 OOF medio de las 3 particiones.

Protocolo (etapa 16): 3 particiones x 5 folds = 15 evaluaciones con el umbral de
toda la OOF de cada partición, diferencia pareada con IC t contra la base,
holdout con bootstrap pareado de 2.000 réplicas. Umbral del holdout: el del OOF
de la partición original para los candidatos que lo tienen (base, n13, limpio y
mezclas); el promedio de los umbrales de las 3 particiones para los LightGBM
nuevos (union y corregida), que no se reentrenan en la partición original para
no pasar el tope de tiempo.

CSV solo si el IC pareado de la CV repetida excluye cero con delta positivo y el
delta del holdout no es negativo; como máximo uno (el de mayor delta CV),
reajustado sobre el 100 % del train y validado.

Uso (desde la raíz del repo; un solo proceso pesado, tope de 4 h):
    python/.venv/Scripts/python.exe python/etapa18_combinacion.py todo [--prueba]
Desacoplado (PowerShell):
    Start-Process -FilePath python\\.venv\\Scripts\\python.exe -ArgumentList python\\etapa18_combinacion.py,todo
      -WorkingDirectory . -WindowStyle Hidden -RedirectStandardOutput stores\\logs\\etapa18_salida.txt
      -RedirectStandardError stores\\logs\\etapa18_error.txt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from configuracion import PARAMETROS_N6, RAIZ, RUTA_LOGS, RUTA_NIVEL7, RUTA_TABLAS
from etapa2_optuna import SEMILLAS_BAGGING
from etapa6_auxiliar_v3 import f1_banda
from funciones import bitacora
from funciones.agregados import PREFIJO as PREFIJO_AGREGADOS
from funciones.agregados import construir_agregados
from funciones.anidado import guardar_parquet_atomico, verificar_sin_fuga
from funciones.bitacora import (CODIGO_SALIDA_MEMORIA, CODIGO_SALIDA_TIEMPO, Presupuesto, iniciar_corte_duro,
                                iniciar_vigilante_memoria, registrar_log)
from funciones.datos import cargar_hogares, cargar_ingreso_hogar, cargar_particion, cargar_personas, imputar_mediana
from funciones.inferencia import bootstrap_holdout, comparar_pareado, f1_por_fold_con_umbral_oof, resumir_cv_repetida
from funciones.metricas import buscar_umbral_f1, calcular_metricas
from funciones.mezclas import combinar_predicciones
from funciones.modelado import crear_ajuste_lightgbm, guardar_predicciones, resultado_desde_predicciones
from funciones.redes import correlaciones_con
from funciones.repeticion import cargar_oof, definir_contextos_cv, evaluar_oof
from funciones.submission import guardar_submission, validar_submission

BASE = "n7_a_agregados_bag5"
REFERENCIA_N13 = "n13_mezcla_final"
RUTA_ETAPA16 = RUTA_NIVEL7 / "etapa16"
RUTA_MATRICES_ORIGINALES = RUTA_NIVEL7 / "etapa10" / "matrices"
RUTA_LIMPIO = RAIZ / "limpio" / "salidas" / "etapa18"
# Hoja de la OOF repetida de la etapa 16 de cada referencia
HOJAS_ETAPA16 = {BASE: "a_bag5", REFERENCIA_N13: "mezcla_final"}

# Variables del modelo limpio que A ya tiene como dummies (mismo concepto)
LIMPIO_YA_EN_A = ["dominio", "tenencia", "oficio_jefe"]
LIMPIO_A_DUMMIES = "posicion_jefe"
CORRELACION_DUPLICADA = 0.999
FILAS_MUESTRA_CORRELACION = 40000
COLUMNA_AUX_LIMPIO = "l_ingreso_aux"

# A corregida: variables de personas con 9 = "no sabe" (diccionario del DANE) y
# grado con 99; vivienda con 98/99 en cuartos y dormitorios
CODIGO_9_PERSONAS = ["P6090", "P6100", "P6210", "P6510", "P6545", "P6580", "P6585s1", "P6585s2", "P6585s3",
                     "P6585s4", "P6590", "P6600", "P6610", "P6620", "P7500s2", "P7500s3", "P7510s1", "P7510s2",
                     "P7510s3", "P7510s5", "P7510s6", "P7510s7"]
CODIGO_99_PERSONAS = ["P6210s1"]
COLUMNAS_VIVIENDA = ["cuartos", "dormitorios", "personas_por_cuarto", "personas_por_dormitorio",
                     "cuartos_por_persona", "dormitorios_por_persona"]

VARIANTES_LGB = {"union": "union_bag5", "corregida": "a_corregida_bag5"}
PASO_PESO, PESO_MAXIMO = 0.05, 0.50
TOLERANCIA_BRECHA_RMSE = 0.02

MINUTOS_TOPE = 240
RESERVA_CIERRE_MIN = 20       # el entrenamiento se corta antes para dejar el cierre
MINUTOS_TEST_LGB = 12         # reajuste bag5 sobre el 100 % si el candidato del CSV es un LightGBM
RELANZAMIENTOS = 8
ESPERA_RELANZAMIENTO_S = 60
HILOS = 14

PRUEBA = False
PREFIJO = "n18_"
CARPETA = RUTA_NIVEL7 / "etapa18"
TABLAS = "etapa18"
PARTICIONES = (1, 2, 3)
PARAMETROS = {"lgb": PARAMETROS_N6, "semillas": SEMILLAS_BAGGING, "replicas": 2000}


def configurar_prueba():
    """Miniatura: partición 1, 20 iteraciones, 2 semillas, 50 réplicas, carpeta
    etapa18_prueba, candidatos n18p_*, tablas etapa18p_*, tope de 30 min y sin CSV."""
    global PRUEBA, PREFIJO, CARPETA, TABLAS, PARTICIONES, MINUTOS_TOPE, RESERVA_CIERRE_MIN
    PRUEBA, PREFIJO, CARPETA, TABLAS, PARTICIONES = True, "n18p_", RUTA_NIVEL7 / "etapa18_prueba", "etapa18p", (1,)
    MINUTOS_TOPE, RESERVA_CIERRE_MIN = 30, 5
    PARAMETROS.update({"lgb": {**PARAMETROS_N6, "num_iterations": 20}, "semillas": SEMILLAS_BAGGING[:2],
                       "replicas": 50})


# ---- Rutas y nombres --------------------------------------------------------------

def carpeta_particion(r):
    """Carpeta de la etapa con las OOF de la partición repetida r."""
    return CARPETA / f"particion_r{r}"


def nombre(hoja, r=None):
    """Nombre de un candidato de la etapa (con la partición si es una OOF repetida)."""
    return f"{PREFIJO}r{r}_{hoja}" if r is not None else f"{PREFIJO}{hoja}"


def ruta_inicio():
    """Archivo con la hora de arranque: el reloj sigue contando tras un relanzamiento."""
    return CARPETA / "inicio.txt"


# ---- Variables del modelo limpio --------------------------------------------------

def cargar_variables_limpio():
    """Variables de la base limpia (train + test) listas para unir a A: sin las
    categóricas que A ya tiene en dummies, posicion_jefe en dummies y prefijo l_.

    Devuelve: DataFrame float32 indexado por id.
    """
    variables = pd.read_parquet(RUTA_LIMPIO / "variables_limpio.parquet").set_index("id")
    variables = variables.drop(columns=LIMPIO_YA_EN_A)
    dummies = pd.get_dummies(variables.pop(LIMPIO_A_DUMMIES).astype(int), prefix=LIMPIO_A_DUMMIES,
                             dtype=np.float32)
    return pd.concat([variables, dummies], axis=1).add_prefix("l_").astype(np.float32)


def elegir_columnas_limpio(limpio, x_a):
    """Columnas del modelo limpio que A no tiene: se descarta la que tenga
    |correlación| > CORRELACION_DUPLICADA con alguna columna de A (en una muestra
    del 80 %; no usa la etiqueta). Caché en columnas_limpio.json.

    Argumentos: limpio = cargar_variables_limpio(); x_a = matriz de A de un contexto.
    Devuelve: (lista de columnas que entran, {descartada: columna de A equivalente}).
    """
    ruta = CARPETA / "columnas_limpio.json"
    if ruta.exists():
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        return datos["entran"], datos["descartadas"]
    muestra = x_a.sample(min(FILAS_MUESTRA_CORRELACION, len(x_a)), random_state=0)
    a = muestra.to_numpy(np.float64)
    l = limpio.reindex(muestra.index)
    l = l.fillna(l.median()).to_numpy(np.float64)

    def estandarizar(m):
        desvio = m.std(0)
        return (m - m.mean(0)) / np.where(desvio > 0, desvio, 1), desvio > 0

    za, va = estandarizar(a)
    zl, vl = estandarizar(l)
    correlacion = np.abs(zl.T @ za / len(za)) * vl[:, None] * va[None, :]
    entran, descartadas = [], {}
    for i, columna in enumerate(limpio.columns):
        j = int(np.argmax(correlacion[i]))
        if correlacion[i, j] > CORRELACION_DUPLICADA:
            descartadas[columna] = muestra.columns[j]
        else:
            entran.append(columna)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"entran": entran, "descartadas": descartadas}, indent=1), encoding="utf-8")
    return entran, descartadas


def cargar_aux_limpio():
    """Columna auxiliar anidada del modelo limpio por (partición, contexto).

    Devuelve: {(particion, contexto): Serie indexada por id}.
    """
    tabla = pd.read_parquet(RUTA_LIMPIO / "aux_limpio_contextos.parquet")
    return {(int(p), c): g.set_index("id")[COLUMNA_AUX_LIMPIO].astype(np.float32)
            for (p, c), g in tabla.groupby(["particion", "contexto"])}


# ---- A corregida ------------------------------------------------------------------

def corregir_personas(personas):
    """Pasa a NA los 9 de las variables sí/no y el 99 del grado (copia)."""
    p = personas.copy()
    for columna in CODIGO_9_PERSONAS:
        if columna in p.columns:
            p.loc[p[columna] == 9, columna] = np.nan
    for columna in CODIGO_99_PERSONAS:
        p.loc[p[columna] == 99, columna] = np.nan
    return p


def agregados_corregidos():
    """Agregados por rol (como la etapa 7) recalculados con las personas
    corregidas. No usan la etiqueta: se calculan una vez para train y test.
    Caché en agregados_corregidos.parquet.

    Devuelve: DataFrame float32 indexado por id con columnas ag_*.
    """
    ruta = CARPETA / "agregados_corregidos.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    personas = corregir_personas(cargar_personas())
    hogares = pd.Index(cargar_particion()["id"], name="id")
    agregados = construir_agregados(personas, hogares)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    guardar_parquet_atomico(agregados, ruta)
    return agregados


def vivienda_corregida():
    """Cuartos, dormitorios y sus razones con 98/99 pasados a NA.

    Devuelve: DataFrame indexado por id con COLUMNAS_VIVIENDA.
    """
    hogar = cargar_hogares()
    cuartos = hogar["cuartos"].where(~hogar["cuartos"].isin([98, 99]))
    dormitorios = hogar["dormitorios"].where(~hogar["dormitorios"].isin([98, 99]))
    n_ug = hogar["personas_ug"].clip(lower=1)
    return pd.DataFrame({
        "cuartos": cuartos, "dormitorios": dormitorios,
        "personas_por_cuarto": hogar["n_personas"] / cuartos.clip(lower=1),
        "personas_por_dormitorio": hogar["n_personas"] / dormitorios.clip(lower=1),
        "cuartos_por_persona": cuartos / n_ug, "dormitorios_por_persona": dormitorios / n_ug,
    }).astype(np.float32)


# ---- Preparadores de matrices -----------------------------------------------------

def ruta_matrices(r):
    """Matrices de A por contexto: las de la etapa 16 (r = 1-3) o las originales
    de la etapa 10 (r = 0: contextos holdout y test)."""
    return RUTA_MATRICES_ORIGINALES if r == 0 else RUTA_ETAPA16 / f"particion_r{r}" / "matrices"


def crear_insumos_variante(variante):
    """Bloques que la variante agrega o reemplaza en las matrices de A.

    Devuelve: diccionario con lo que usa crear_preparador().
    """
    if variante == "union":
        limpio = cargar_variables_limpio()
        x_a = pd.read_parquet(ruta_matrices(PARTICIONES[0]) / "cv_1_eval.parquet")
        entran, descartadas = elegir_columnas_limpio(limpio, x_a)
        del x_a
        return {"limpio": limpio[entran], "aux": cargar_aux_limpio(), "descartadas": descartadas}
    mes = pd.read_parquet(RUTA_LIMPIO / "variables_limpio.parquet", columns=["id", "mes"]).set_index("id")
    return {"agregados": agregados_corregidos(), "vivienda": vivienda_corregida(),
            "mes": mes["mes"].astype(np.float32)}


def crear_preparador(variante, r, insumos):
    """preparar(contexto) -> (x_ent, y_ent, x_eval) de una variante sobre las
    matrices de A de la partición r (0 = original). Los bloques nuevos se imputan
    con la mediana del entrenamiento del contexto, como el resto de A.

    Argumentos: variante = 'union' o 'corregida'; r = partición; insumos =
    crear_insumos_variante().
    """
    def preparar(contexto):
        carpeta = ruta_matrices(r)
        x_ent = pd.read_parquet(carpeta / f"{contexto}_ent.parquet")
        y_ent = x_ent.pop("pobre").to_numpy()
        x_eval = pd.read_parquet(carpeta / f"{contexto}_eval.parquet")
        if variante == "union":
            aux = insumos["aux"][(r, contexto)]
            nuevo = insumos["limpio"].join(aux.rename(COLUMNA_AUX_LIMPIO), how="left")
            assert aux.reindex(x_ent.index).notna().all() and aux.reindex(x_eval.index).notna().all(), \
                f"{contexto} r{r}: columna auxiliar limpia incompleta"
        else:
            x_ent = x_ent.drop(columns=[c for c in x_ent.columns if c.startswith(PREFIJO_AGREGADOS)])
            x_eval = x_eval.drop(columns=[c for c in x_eval.columns if c.startswith(PREFIJO_AGREGADOS)])
            x_ent, x_eval = x_ent.drop(columns=COLUMNAS_VIVIENDA), x_eval.drop(columns=COLUMNAS_VIVIENDA)
            nuevo = pd.concat([insumos["vivienda"], insumos["mes"].rename("mes"), insumos["agregados"]], axis=1)
        nuevo_ent, nuevo_eval = imputar_mediana(nuevo.reindex(x_ent.index), nuevo.reindex(x_eval.index))
        return pd.concat([x_ent, nuevo_ent], axis=1), y_ent, pd.concat([x_eval, nuevo_eval], axis=1)

    return preparar


# ---- Entrenamiento ----------------------------------------------------------------

def ruta_holdout_lgb(hoja):
    """Probabilidades del holdout (y del test) de un LightGBM nuevo."""
    return CARPETA / "holdout" / f"{nombre(hoja)}.parquet"


def ajustar_contexto_original(variante, contexto, insumos, ruta):
    """Ajusta la variante en un contexto de la partición original (holdout o test)
    y guarda las probabilidades (reanudable: si existe, no reajusta)."""
    if ruta.exists():
        return pd.read_parquet(ruta)["probabilidad"]
    x_ent, y_ent, x_eval = crear_preparador(variante, 0, insumos)(contexto)
    probabilidad, _ = crear_ajuste_lightgbm(PARAMETROS["lgb"], PARAMETROS["semillas"])(x_ent, y_ent, x_eval)
    serie = pd.Series(probabilidad, index=x_eval.index, name="probabilidad")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    guardar_parquet_atomico(serie.to_frame(), ruta)
    return serie


def entrenar():
    """Trabajo pesado: OOF de las variantes LightGBM en las particiones repetidas
    y su holdout, en orden de prioridad (unión primero). Reanudable por fold."""
    verdad = cargar_particion().set_index("id")["pobre"]
    for variante, hoja in VARIANTES_LGB.items():
        insumos = crear_insumos_variante(variante)
        registrar_log(TABLAS, f"{variante}: insumos listos")
        for r in PARTICIONES:
            candidato = nombre(hoja, r)
            if cargar_oof(candidato, carpeta_particion(r)) is not None:
                registrar_log(candidato, "OOF desde caché")
                continue
            evaluar_oof(candidato, crear_preparador(variante, r, insumos),
                        crear_ajuste_lightgbm(PARAMETROS["lgb"], PARAMETROS["semillas"]), verdad,
                        carpeta_particion(r))
        inicio = time.time()
        ajustar_contexto_original(variante, "holdout", insumos, ruta_holdout_lgb(hoja))
        registrar_log(nombre(hoja), "holdout listo", inicio)
        del insumos


# ---- Insumos del reporte -----------------------------------------------------------

def oof_referencia(candidato, r):
    """OOF repetida de la etapa 16 de la base o de n13 en la partición r."""
    oof = cargar_oof(f"n16_r{r}_{HOJAS_ETAPA16[candidato]}", RUTA_ETAPA16 / f"particion_r{r}")
    assert oof is not None, f"falta la OOF de la etapa 16 de {candidato} en r{r}"
    return oof


def oof_limpio(r, verdad):
    """OOF repetida del modelo limpio en la partición r, con los folds de la
    etapa 16 (se verifica que coincidan)."""
    tabla = pd.read_parquet(RUTA_LIMPIO / "oof_limpio_repetida.parquet")
    tabla = tabla[tabla["particion"] == r].set_index("id")
    grupos = pd.read_parquet(RUTA_ETAPA16 / f"particion_r{r}" / "grupos.parquet")["grupo"]
    assert (grupos.reindex(tabla.index).to_numpy() == tabla["fold"].to_numpy()).all(), \
        f"r{r}: los folds del modelo limpio no son los de la etapa 16"
    return pd.DataFrame({"probabilidad": tabla["probabilidad"], "fold": tabla["fold"].astype(np.int64),
                         "verdad": verdad.loc[tabla.index].astype(np.int64)})


def grilla_pesos():
    """Pesos del modelo limpio en la mezcla: 0, 0,05, ..., PESO_MAXIMO."""
    return np.round(np.arange(0, PESO_MAXIMO + 1e-9, PASO_PESO), 2)


def f1_mezcla(oof_principal, oof_secundario, peso, filas=None):
    """F1 OOF (umbral óptimo en esas filas) de (1 - peso) x principal + peso x secundario."""
    p = (1 - peso) * oof_principal["probabilidad"] + peso * oof_secundario["probabilidad"].reindex(oof_principal.index)
    y = oof_principal["verdad"]
    if filas is not None:
        p, y = p[filas], y[filas]
    return buscar_umbral_f1(y.to_numpy(), p.to_numpy())[1]


def mezcla_ajuste_cruzado(oof_principal, oof_secundario):
    """OOF de la mezcla con peso elegido por ajuste cruzado: las filas del fold k
    usan el peso que maximiza el F1 en los otros 4 folds.

    Devuelve: (OOF de la mezcla, {fold: peso}).
    """
    pesos = {}
    probabilidad = pd.Series(np.nan, index=oof_principal.index)
    secundario = oof_secundario["probabilidad"].reindex(oof_principal.index)
    for k in sorted(oof_principal["fold"].unique()):
        otros = (oof_principal["fold"] != k).to_numpy()
        f1 = [f1_mezcla(oof_principal, oof_secundario, w, otros) for w in grilla_pesos()]
        pesos[int(k)] = float(grilla_pesos()[int(np.argmax(f1))])
        filas = oof_principal["fold"] == k
        probabilidad[filas] = (1 - pesos[int(k)]) * oof_principal.loc[filas, "probabilidad"] + \
            pesos[int(k)] * secundario[filas]
    return oof_principal.assign(probabilidad=probabilidad), pesos


def peso_final(oofs_principal, oofs_secundario):
    """Peso para holdout y test: maximiza el F1 OOF medio de las particiones.

    Devuelve: (peso, DataFrame con el F1 medio de cada peso de la grilla).
    """
    filas = [{"peso_limpio": w, "f1_oof_medio": float(np.mean([f1_mezcla(oofs_principal[r], oofs_secundario[r], w)
                                                              for r in oofs_principal]))}
             for w in grilla_pesos()]
    tabla = pd.DataFrame(filas)
    return float(tabla.loc[tabla["f1_oof_medio"].idxmax(), "peso_limpio"]), tabla


def resultado_limpio(verdad):
    """Copia las predicciones del modelo limpio (OOF original, holdout y test) al
    formato del pipeline y las devuelve como resultado_desde_predicciones()."""
    destino = RUTA_NIVEL7 / f"predicciones_{nombre('limpio')}.parquet"
    shutil.copyfile(RUTA_LIMPIO / "predicciones_n18_limpio.parquet", destino)
    return resultado_desde_predicciones(nombre("limpio"), verdad)


# ---- Tests de fuga ----------------------------------------------------------------

def tests_fuga(verdad):
    """Tests de fuga de lo que usa ingresos: auxiliares de A (registro de la
    etapa 16), columna auxiliar limpia (ids por contexto y brecha de RMSE
    entrenamiento/evaluación) y orden de las mezclas (pesos solo con OOF).

    Devuelve: DataFrame (prueba, valor, pasa).
    """
    filas = []
    contextos = definir_contextos_cv()
    for r in PARTICIONES:
        grupos = pd.read_parquet(RUTA_ETAPA16 / f"particion_r{r}" / "grupos.parquet")["grupo"]
        for auxiliar in ("ingreso_personas", "ingreso_total_personas", "ingreso_fuentes_v3"):
            registro = pd.read_parquet(RUTA_ETAPA16 / f"particion_r{r}" / f"registro_{auxiliar}.parquet")
            try:
                pasa = verificar_sin_fuga(registro, grupos, contextos)
            except AssertionError as error:
                registrar_log(TABLAS, f"FUGA {auxiliar} r{r}: {error}")
                pasa = False
            filas.append({"prueba": f"A: auxiliar {auxiliar} anidado sin fuga (r{r})", "valor": np.nan, "pasa": pasa})
    aux = cargar_aux_limpio()
    z = cargar_ingreso_hogar()   # log(Ingpcug / Lp); el limpio usa log(1 + Ingpcug / Lp)
    z_limpio = np.log1p(np.exp(z))
    holdout_ids = set(cargar_particion().query("holdout")["id"])
    for (r, contexto), serie in aux.items():
        if contexto == "test":
            continue
        if r == 0:
            evaluacion = serie.index.isin(holdout_ids) if contexto == "holdout" else None
            if evaluacion is None:
                grupos_r = cargar_particion().set_index("id")["fold_id"]
                evaluacion = (grupos_r.reindex(serie.index) == int(contexto[-1])).to_numpy()
        else:
            grupos = pd.read_parquet(RUTA_ETAPA16 / f"particion_r{r}" / "grupos.parquet")["grupo"]
            evaluacion = (grupos.reindex(serie.index) == int(contexto[-1])).to_numpy()
        error = (serie - z_limpio.reindex(serie.index)).to_numpy()
        rmse_ent, rmse_eval = np.sqrt(np.mean(error[~evaluacion] ** 2)), np.sqrt(np.mean(error[evaluacion] ** 2))
        filas.append({"prueba": f"limpio: RMSE entrenamiento >= evaluación ({contexto}, r{r})",
                      "valor": (rmse_ent - rmse_eval) / rmse_eval,
                      "pasa": (rmse_ent - rmse_eval) / rmse_eval > -TOLERANCIA_BRECHA_RMSE})
    ids_train = set(cargar_particion().query("base == 'train'")["id"])
    filas.append({"prueba": "limpio: el contexto holdout cubre exactamente el train (80 % OOF + holdout)",
                  "valor": np.nan, "pasa": set(aux[(0, "holdout")].index) == ids_train})
    return pd.DataFrame(filas)


# ---- Reporte y CSV ------------------------------------------------------------------

def correlacion_media(oofs, oofs_base):
    """Pearson y Spearman medios (sobre particiones) de una OOF con la de la base."""
    tablas = [correlaciones_con(oofs_base[r]["probabilidad"], {"x": oofs[r]["probabilidad"]}) for r in oofs]
    tabla = pd.concat(tablas)
    return float(tabla["pearson_con_lightgbm"].mean()), float(tabla["spearman_con_lightgbm"].mean())


def reporte(reloj):
    """Mezclas, tabla de la etapa, tests de fuga y, si algún candidato cumple la
    regla, el CSV (máximo uno)."""
    verdad = cargar_particion().set_index("id")["pobre"]
    oofs = {BASE: {r: oof_referencia(BASE, r) for r in PARTICIONES},
            REFERENCIA_N13: {r: oof_referencia(REFERENCIA_N13, r) for r in PARTICIONES},
            nombre("limpio"): {r: oof_limpio(r, verdad) for r in PARTICIONES}}
    resultados = {BASE: resultado_desde_predicciones(BASE, verdad),
                  REFERENCIA_N13: resultado_desde_predicciones(REFERENCIA_N13, verdad),
                  nombre("limpio"): resultado_limpio(verdad)}
    reglas_umbral = {BASE: "OOF original", REFERENCIA_N13: "OOF original", nombre("limpio"): "OOF original"}

    # Mezclas: pesos solo con las OOF repetidas (ajuste cruzado para evaluar)
    filas_pesos = []
    for principal, hoja in ((BASE, "mezcla_a_limpio"), (REFERENCIA_N13, "mezcla_n13_limpio")):
        candidato = nombre(hoja)
        oofs[candidato] = {}
        for r in PARTICIONES:
            oofs[candidato][r], pesos = mezcla_ajuste_cruzado(oofs[principal][r], oofs[nombre("limpio")][r])
            filas_pesos += [{"mezcla": candidato, "particion": r, "fold": k, "peso_limpio": w}
                            for k, w in pesos.items()]
        w, tabla_pesos = peso_final(oofs[principal], oofs[nombre("limpio")])
        tabla_pesos.assign(mezcla=candidato).to_csv(RUTA_TABLAS / f"{TABLAS}_{hoja}_grilla.csv", index=False)
        guardar_predicciones(combinar_predicciones(candidato, [resultados[principal], resultados[nombre("limpio")]],
                                                   np.array([1 - w, w])))
        resultados[candidato] = resultado_desde_predicciones(candidato, verdad)
        resultados[candidato]["peso_limpio"] = w
        reglas_umbral[candidato] = "OOF original"
        registrar_log(TABLAS, f"{candidato}: peso final del limpio {w:.2f}")
    pd.DataFrame(filas_pesos).to_csv(RUTA_TABLAS / f"{TABLAS}_pesos_por_fold.csv", index=False)

    # LightGBM nuevos: OOF repetidas y holdout con el umbral medio de las particiones
    incompletos = []
    for variante, hoja in VARIANTES_LGB.items():
        candidato = nombre(hoja)
        propios = {r: cargar_oof(nombre(hoja, r), carpeta_particion(r)) for r in PARTICIONES}
        propios = {r: o for r, o in propios.items() if o is not None}
        if not propios:
            incompletos.append(candidato)
            continue
        oofs[candidato] = propios
        umbral = float(np.mean([buscar_umbral_f1(o["verdad"], o["probabilidad"])[0] for o in propios.values()]))
        if ruta_holdout_lgb(hoja).exists():
            holdout = pd.read_parquet(ruta_holdout_lgb(hoja))["probabilidad"]
            metricas = calcular_metricas(verdad.loc[holdout.index].astype(int), (holdout >= umbral).astype(int))
            resultados[candidato] = {"nombre": candidato, "umbral": umbral, "prediccion_holdout": holdout,
                                     **{f"{k}_holdout": v for k, v in metricas.items()}}
            reglas_umbral[candidato] = "media de las particiones"
        if len(propios) < len(PARTICIONES) or candidato not in resultados:
            incompletos.append(candidato)

    # Folds, pareadas contra la base y bootstrap del holdout
    folds = pd.concat([f1_por_fold_con_umbral_oof(o).assign(candidato=c, particion=r)
                       for c, por_r in oofs.items() for r, o in por_r.items()], ignore_index=True)
    folds.to_csv(RUTA_TABLAS / f"{TABLAS}_folds.csv", index=False)
    folds_base = folds[folds["candidato"] == BASE]
    ids_holdout = resultados[BASE]["prediccion_holdout"].index
    clases = {c: (r["prediccion_holdout"].reindex(ids_holdout) >= r["umbral"]).astype(int)
              for c, r in resultados.items()}
    bootstrap = bootstrap_holdout(verdad.loc[ids_holdout].astype(int), clases, BASE,
                                  n_replicas=PARAMETROS["replicas"]).set_index("candidato")
    n_esperadas = 5 * len(PARTICIONES)
    filas = []
    for candidato in oofs:
        propios = folds[folds["candidato"] == candidato]
        resumen = resumir_cv_repetida(propios)
        pareado = comparar_pareado(propios, folds_base)
        resultado = resultados.get(candidato)
        boot = bootstrap.loc[candidato] if candidato in bootstrap.index else None
        pearson, spearman = correlacion_media(oofs[candidato], oofs[BASE])
        fila = {
            "candidato": candidato, "n_evaluaciones": resumen["n_evaluaciones"],
            "f1_cv_repetida": resumen["f1_medio"], "f1_cv_repetida_ee": resumen["f1_ee"],
            "delta_cv": pareado["delta_medio"], "delta_cv_ic_inferior": pareado["ic_inferior"],
            "delta_cv_ic_superior": pareado["ic_superior"], "delta_cv_excluye_cero": pareado["excluye_cero"],
            "delta_cv_ic_inferior_corregido": pareado["ic_inferior_corregido"],
            "delta_cv_ic_superior_corregido": pareado["ic_superior_corregido"],
            "umbral_holdout": resultado["umbral"] if resultado else np.nan,
            "regla_umbral": reglas_umbral.get(candidato, ""),
            "f1_holdout": boot["f1"] if boot is not None else np.nan,
            "holdout_ic_inferior": boot["ic_inferior"] if boot is not None else np.nan,
            "holdout_ic_superior": boot["ic_superior"] if boot is not None else np.nan,
            "delta_holdout": boot["delta_observado"] if boot is not None else np.nan,
            "delta_holdout_ic_inferior": boot["delta_ic_inferior"] if boot is not None else np.nan,
            "delta_holdout_ic_superior": boot["delta_ic_superior"] if boot is not None else np.nan,
            "precision_holdout": resultado["precision_holdout"] if resultado else np.nan,
            "recall_holdout": resultado["recall_holdout"] if resultado else np.nan,
            "f1_banda_08_12": (f1_banda(verdad, resultado["prediccion_holdout"], resultado["umbral"])["f1_banda_08_12"]
                               if resultado else np.nan),
            "pearson_oof_con_lightgbm": pearson, "spearman_oof_con_lightgbm": spearman,
            "peso_limpio": resultado.get("peso_limpio", np.nan) if resultado else np.nan,
            "completo": candidato not in incompletos and resumen["n_evaluaciones"] == n_esperadas,
        }
        fila["cumple_regla_csv"] = bool(candidato not in (BASE, REFERENCIA_N13) and fila["completo"]
                                        and fila["delta_cv"] > 0 and fila["delta_cv_excluye_cero"]
                                        and fila["delta_holdout"] >= 0)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / f"{TABLAS}_resumen.csv", index=False)
    fuga = tests_fuga(verdad)
    fuga.to_csv(RUTA_TABLAS / f"{TABLAS}_fuga.csv", index=False)
    registrar_log(TABLAS, f"tests de fuga: {int(fuga['pasa'].sum())}/{len(fuga)} pasan")
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print(tabla.round(4).to_string(index=False))

    elegibles = tabla[tabla["cumple_regla_csv"]].sort_values("delta_cv", ascending=False)
    if PRUEBA or elegibles.empty or not fuga["pasa"].all():
        registrar_log(TABLAS, f"sin CSV: {'miniatura' if PRUEBA else 'ningún candidato cumple la regla o falla un test de fuga'}")
        return tabla
    generar_csv(elegibles.iloc[0], resultados[elegibles.iloc[0]["candidato"]], tabla, reloj)
    return tabla


def generar_csv(fila, resultado, tabla, reloj):
    """CSV de Kaggle del candidato elegido, reajustado sobre el 100 % del train y
    validado, con su fila en la bitácora."""
    candidato = fila["candidato"]
    particion = cargar_particion()
    ids_test = particion.loc[particion["base"] == "test", "id"]
    if "prediccion_test" not in resultado:
        variante = next(v for v, h in VARIANTES_LGB.items() if nombre(h) == candidato)
        if reloj.restante() < MINUTOS_TEST_LGB:
            registrar_log(TABLAS, f"{candidato}: no queda tiempo para el reajuste sobre el 100 %; sin CSV")
            return
        ruta = CARPETA / "holdout" / f"{candidato}_test.parquet"
        resultado["prediccion_test"] = ajustar_contexto_original(variante, "test", crear_insumos_variante(variante),
                                                                 ruta)
    resultado["f1_cv"] = fila["f1_cv_repetida"]
    resultado["f1_holdout"] = fila["f1_holdout"]
    especificacion = (f"etapa 18 | F1 CV repetida {fila['f1_cv_repetida']:.4f} (delta {fila['delta_cv']:+.4f} "
                      f"[{fila['delta_cv_ic_inferior']:+.4f}; {fila['delta_cv_ic_superior']:+.4f}]) | holdout "
                      f"{fila['f1_holdout']:.4f} | umbral {resultado['umbral']:.4f} ({fila['regla_umbral']})")
    if not np.isnan(fila["peso_limpio"]):
        especificacion += f" | peso del modelo limpio {fila['peso_limpio']:.2f}"
    algoritmo = "Stacking" if "mezcla" in candidato else "Boosting"
    ruta, tasa = guardar_submission(resultado, f"{candidato}.csv", ids_test, especificacion, algoritmo=algoritmo)
    submission = pd.read_csv(ruta)
    validar_submission(submission, ids_test)
    registrar_log(TABLAS, f"CSV {ruta.name}: tasa de pobres en test {tasa:.4f}")


# ---- Orquestación -------------------------------------------------------------

def correr_subproceso(comando):
    """Corre una etapa del flujo en un subproceso con los hilos de la etapa."""
    argumentos = [sys.executable, str(Path(__file__).resolve()), comando] + (["--prueba"] if PRUEBA else [])
    entorno = {**os.environ, "PS2_NUCLEOS": str(HILOS)}
    with open(RUTA_LOGS / f"{TABLAS}_{comando}_salida.txt", "a", encoding="utf-8") as salida, \
            open(RUTA_LOGS / f"{TABLAS}_{comando}_error.txt", "a", encoding="utf-8") as error:
        return subprocess.run(argumentos, stdout=salida, stderr=error, cwd=RAIZ, env=entorno).returncode


def todo():
    """Entrenamiento (con relanzamiento si sale por memoria y corte duro), luego
    el reporte, y la marca fin.txt."""
    CARPETA.mkdir(parents=True, exist_ok=True)
    (CARPETA / "fin.txt").unlink(missing_ok=True)
    reloj = Presupuesto(MINUTOS_TOPE, ruta_inicio())
    registrar_log(TABLAS, f"inicio ({reloj.transcurrido():.1f} min ya transcurridos)")
    for intento in range(1, RELANZAMIENTOS + 2):
        codigo = correr_subproceso("entrenar")
        registrar_log(TABLAS, f"entrenar: código {codigo} (intento {intento}, {reloj.transcurrido():.1f} min)")
        if codigo != CODIGO_SALIDA_MEMORIA:
            break
        time.sleep(ESPERA_RELANZAMIENTO_S)
    codigo = correr_subproceso("reporte")
    registrar_log(TABLAS, f"reporte: código {codigo} ({reloj.transcurrido():.1f} min)")
    (CARPETA / "fin.txt").write_text(f"{datetime.now().isoformat()} reporte {codigo}", encoding="utf-8")


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("comando", choices=["todo", "entrenar", "reporte"])
    analizador.add_argument("--prueba", action="store_true", help="miniatura para probar el flujo")
    args = analizador.parse_args()
    if args.prueba:
        configurar_prueba()
    CARPETA.mkdir(parents=True, exist_ok=True)
    bitacora.ARCHIVOS_LOG_EXTRA.append(RUTA_LOGS / f"{TABLAS}.log")
    bitacora.LIMITE_PROCESO_GB = 9.0
    if args.comando == "todo":
        todo()
    elif args.comando == "entrenar":
        iniciar_vigilante_memoria(TABLAS)
        iniciar_corte_duro(TABLAS, Presupuesto(MINUTOS_TOPE - RESERVA_CIERRE_MIN, ruta_inicio()), threading.Event())
        entrenar()
    else:
        reporte(Presupuesto(MINUTOS_TOPE, ruta_inicio()))
