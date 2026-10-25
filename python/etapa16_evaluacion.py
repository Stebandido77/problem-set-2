"""Etapa 16: evaluación definitiva de las hipótesis del nivel 7.

Fase 1 (precisión): validación cruzada repetida sobre el 80 % con tres
particiones nuevas de 5 folds (semillas distintas de la de R), auxiliares de
ingreso recalculados con OOF anidada en cada partición (sin reusar cachés de otra),
bootstrap pareado del holdout de siempre (2.000 réplicas, misma remuestra para
todos los candidatos) y comparaciones pareadas contra n7_a_agregados_bag5 con
IC 95 %. Una mejora cuenta como real solo si el IC pareado excluye cero en la
CV repetida Y en el holdout. Se reporta el efecto mínimo detectable del diseño.

Candidatos (tal como fueron definidos; las mezclas con sus pesos fijos):
  n7_a_agregados_bag5, n13_mezcla_final, n7_v3_fuentes_bag5,
  n7_n6param_a4_total_bag5, A + KNN (5 semillas), mezcla LGB + RealMLP, CatBoost B3.

Trabajos (cada uno un subproceso reanudable; `fase1` los orquesta con máximo dos a
la vez, vigilante de memoria con salida limpia si el sistema baja de 2 GB libres y
relanzamiento del que salga por memoria):
  matrices      grupos, auxiliares anidados y matrices de A por contexto cv_k
  arboles       LightGBM n6: base A, n7_v3, a4_total y A + KNN fuera de fold
  catboost      CatBoost B3 (categóricas nativas) sobre A
  redes         RealMLP (5 semillas) y DeepSets (3 semillas), sus bags y las mezclas
  knn_original  A + KNN con 5 semillas en la partición original (OOF + holdout)
  reporte       tablas de la fase: folds, pareadas, bootstrap, resumen y tiempos

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa16_evaluacion.py fase1 [--prueba]
    python/.venv/Scripts/python.exe python/etapa16_evaluacion.py trabajo <tipo> [--particion r] [--prueba]
Desacoplado de la terminal (PowerShell):
    Start-Process -FilePath python\\.venv\\Scripts\\python.exe -ArgumentList python\\etapa16_evaluacion.py,fase1
      -WorkingDirectory . -WindowStyle Hidden -RedirectStandardOutput stores\\logs\\etapa16_fase1_salida.txt
      -RedirectStandardError stores\\logs\\etapa16_fase1_error.txt
--prueba corre la miniatura (una partición, 20 iteraciones, 2 semillas, 50
réplicas) en etapa16_prueba/ con candidatos n16p_* y tablas etapa16p_*.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import psutil

from configuracion import PARAMETROS_N6, RAIZ, RUTA_LOGS, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa2_optuna import SEMILLAS_BAGGING
from funciones import auxiliares as modulo_auxiliares
from funciones import bitacora
from funciones.anidado import crear_perezoso, guardar_parquet_atomico
from funciones.auxiliares import PARAMETROS_AUXILIAR, crear_auxiliar_fuentes, crear_auxiliar_personas
from funciones.bases import BLOQUES_A4, armar_matriz, reconciliar_auxiliares
from funciones.bitacora import CODIGO_SALIDA_MEMORIA, iniciar_vigilante_memoria, registrar_log
from funciones.datos import (cargar_hogares, cargar_ingreso_fuentes, cargar_ingreso_personas, cargar_particion,
                             cargar_personas, imputar_mediana)
from funciones.inferencia import bootstrap_holdout, comparar_pareado, f1_por_fold_con_umbral_oof, resumir_cv_repetida
from funciones.modelado import (armar_contexto, borrar_parciales, crear_ajuste_lightgbm, evaluar_modelo,
                                resultado_desde_predicciones)
from funciones.repeticion import (SEMILLAS_PARTICIONES, anidado_en_carpeta, cargar_oof, columnas_a4_total,
                                  columnas_v3, definir_contextos_cv, evaluar_oof, generar_particion_repetida,
                                  promediar_oof)
from funciones.variables import construir_grupos

BASE = "n7_a_agregados_bag5"
# Candidato de la spec -> (hoja de la OOF repetida, candidato con las predicciones
# del pipeline para el holdout)
CANDIDATOS = {
    "n7_a_agregados_bag5": ("a_bag5", "n7_a_agregados_bag5"),
    "n13_mezcla_final": ("mezcla_final", "n13_mezcla_final"),
    "n7_v3_fuentes_bag5": ("v3_fuentes_bag5", "n7_v3_fuentes_bag5"),
    "n7_n6param_a4_total_bag5": ("a4_total_bag5", "n7_n6param_a4_total_bag5"),
    "A + KNN (bag5)": ("a_knn_bag5", "n16_a_knn_bag5"),
    "mezcla LGB + RealMLP": ("mezcla_lgb_realmlp", "n10_mezcla_lgb_realmlp"),
    "CatBoost B3": ("catboost_b3", "n7_b3_catboost_agr"),
}
# Mezclas con pesos fijos (los elegidos en OOF en las etapas 10 y 13)
MEZCLAS = {
    "mezcla_lgb_realmlp": (("a_bag5", 0.85), ("realmlp_bag5", 0.15)),
    "mezcla_final": (("a_bag5", 0.80), ("realmlp_bag5", 0.15), ("conjuntos_bag3", 0.05)),
}
AUXILIARES_A = ("laboral", "total", "fuentes")
RUTA_MATRICES_ORIGINALES = RUTA_NIVEL7 / "etapa10" / "matrices"
RUTA_AGREGADOS = RUTA_NIVEL7 / "etapa7" / "agregados_personas.parquet"

PRUEBA = False
PREFIJO = "n16_"
CARPETA = RUTA_NIVEL7 / "etapa16"
TABLAS = "etapa16"
PARTICIONES = (1, 2, 3)
FASE = "fase1"
TIPOS_TRABAJO = ("matrices", "arboles", "catboost", "redes", "knn_original", "reporte")

# Orquestación: dos procesos pesados como máximo, cada uno con los núcleos físicos
MAXIMO_PROCESOS = 2
HILOS_POR_PROCESO = 8
MINIMO_LIBRE_SEGUNDO_GB = 5.0   # el segundo proceso solo arranca con esta memoria libre
LIMITE_PROCESO_GB = 7.5
RELANZAMIENTOS = 8
ESPERA_RELANZAMIENTO_S = 60
INTERVALO_SONDEO_S = 15

# Parámetros de los candidatos (configurar_prueba() los achica)
PARAMETROS = {
    "lgb": PARAMETROS_N6,
    "auxiliar": PARAMETROS_AUXILIAR,
    "catboost": None,            # PARAMETROS_CATBOOST de la etapa 7 (se lee al usarlo)
    "realmlp_extra": {},         # se funde con la configuración de la etapa 10
    "conjuntos_extra": {},       # se funde con PARAMETROS_CONJUNTOS
    "semillas_bagging": SEMILLAS_BAGGING,
    "semillas_realmlp": 5,
    "semillas_conjuntos": 3,
    "replicas": 2000,
}


def configurar_prueba():
    """Miniatura para probar el flujo completo: una partición, candidatos n16p_*,
    carpeta etapa16_prueba, tablas etapa16p_*, 20 iteraciones en los boosting y
    los auxiliares, una época en las redes, 2 semillas y 50 réplicas."""
    global PRUEBA, PREFIJO, CARPETA, TABLAS, PARTICIONES
    PRUEBA, PREFIJO, CARPETA, TABLAS, PARTICIONES = True, "n16p_", RUTA_NIVEL7 / "etapa16_prueba", "etapa16p", (1,)
    from etapa7_exploracion import PARAMETROS_CATBOOST  # noqa: PLC0415
    modulo_auxiliares.PARAMETROS_AUXILIAR = {**PARAMETROS_AUXILIAR, "num_iterations": 20}
    PARAMETROS.update({
        "lgb": {**PARAMETROS_N6, "num_iterations": 20}, "auxiliar": modulo_auxiliares.PARAMETROS_AUXILIAR,
        "catboost": {**PARAMETROS_CATBOOST, "iterations": 20}, "realmlp_extra": {"n_epochs": 1},
        "conjuntos_extra": {"max_epocas": 1}, "semillas_bagging": SEMILLAS_BAGGING[:2], "semillas_realmlp": 2,
        "semillas_conjuntos": 2, "replicas": 50,
    })


# ---- Rutas y nombres --------------------------------------------------------------

def carpeta_particion(r):
    """Carpeta de una partición repetida (grupos, anidados, matrices, OOF)."""
    return CARPETA / f"particion_r{r}"


def ruta_matrices(r):
    """Carpeta con las matrices de A por contexto de una partición."""
    return carpeta_particion(r) / "matrices"


def nombre_candidato(r, hoja):
    """Nombre de la OOF repetida de un candidato en la partición r."""
    return f"{PREFIJO}r{r}_{hoja}"


def ruta_log_fase():
    """Log propio de la fase (además de nivel7.log)."""
    return RUTA_LOGS / f"{TABLAS}_{FASE}.log"


def nombre_knn_original():
    """Candidato A + KNN con 5 semillas en la partición original (n16p_ en la miniatura)."""
    return f"{PREFIJO}a_knn_bag5"


def escribir_json_atomico(ruta, datos):
    """Escribe un JSON en un temporal y lo renombra, para que un corte a mitad de
    escritura nunca deje un archivo truncado con el nombre definitivo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(datos, indent=1), encoding="utf-8")
    temporal.replace(ruta)


def leer_json_tolerante(ruta):
    """Contenido de un JSON, o None si no existe o está roto (se avisa en el log)."""
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        registrar_log("etapa16", f"se ignora {ruta.name}: {type(error).__name__}")
        return None


def anotar_tiempo(nombre, minutos):
    """Minutos de un candidato (un archivo por candidato: varios procesos escriben a la vez)."""
    escribir_json_atomico(CARPETA / "tiempos" / f"{nombre}.json", {"minutos": minutos})


def leer_tiempos():
    """{candidato: minutos} de todos los candidatos anotados (se saltan los archivos rotos)."""
    carpeta = CARPETA / "tiempos"
    if not carpeta.exists():
        return {}
    tiempos = {}
    for ruta in sorted(carpeta.glob("*.json")):
        datos = leer_json_tolerante(ruta)
        if datos is not None and "minutos" in datos:
            tiempos[ruta.stem] = datos["minutos"]
    return tiempos


def evaluar_o_cargar_oof(nombre, carpeta, evaluar):
    """OOF guardada del candidato si ya terminó (y limpia parciales huérfanos); si
    no, la evalúa y anota los minutos."""
    oof = cargar_oof(nombre, carpeta)
    if oof is not None:
        registrar_log(nombre, "OOF desde caché")
        shutil.rmtree(carpeta / "parciales" / nombre, ignore_errors=True)
        return oof
    inicio = time.time()
    oof = evaluar()
    anotar_tiempo(nombre, (time.time() - inicio) / 60)
    gc.collect()
    return oof


def columnas_sin_indice(ruta):
    """Columnas de un parquet escrito por pandas, sin las del índice (p. ej. id),
    leyendo solo el esquema."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    esquema = pq.read_schema(ruta)
    indices = set((esquema.pandas_metadata or {}).get("index_columns", []))
    return [c for c in esquema.names if c not in indices]


def columnas_a():
    """Columnas de la base de A (sin pobre), en el orden de las matrices de la etapa 10."""
    return columnas_sin_indice(RUTA_MATRICES_ORIGINALES / "cv_1_eval.parquet")


# ---- matrices: grupos, auxiliares anidados y matrices de A por partición ----------

def grupos_particion(r):
    """Grupos de la partición r (caché en particion_r<r>/grupos.parquet)."""
    ruta = carpeta_particion(r) / "grupos.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)["grupo"]
    grupos = generar_particion_repetida(cargar_particion(), SEMILLAS_PARTICIONES[r])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    guardar_parquet_atomico(grupos.rename("grupo").to_frame().rename_axis("id"), ruta)
    return grupos


def auxiliares_particion(r, grupos, hogar, personas, bloques):
    """Auxiliares de ingreso (laboral, total y fuentes v3) de una partición con OOF
    anidada en sus contextos cv_k, con la misma construcción que cargar_insumos()."""
    carpeta, contextos = carpeta_particion(r), definir_contextos_cv()
    ingreso = cargar_ingreso_personas()
    parametros = PARAMETROS["auxiliar"]
    contexto_hogar = pd.concat([bloques["nivel_3"], bloques["laborales"]], axis=1)
    auxiliares = {
        "laboral": anidado_en_carpeta("ingreso_personas", crear_perezoso(
            lambda: crear_auxiliar_personas(personas, ingreso, hogar, parametros=parametros)),
            grupos, carpeta, contextos),
        "total": anidado_en_carpeta("ingreso_total_personas", crear_perezoso(
            lambda: crear_auxiliar_personas(personas, ingreso, hogar, "ingreso_total", "_total", parametros=parametros)),
            grupos, carpeta, contextos),
        "fuentes": anidado_en_carpeta("ingreso_fuentes_v3", crear_perezoso(
            lambda: crear_auxiliar_fuentes(personas, cargar_ingreso_fuentes(), hogar, contexto_hogar)),
            grupos, carpeta, contextos),
    }
    return reconciliar_auxiliares(auxiliares, hogar, arriendo_imputado=False)


def construir_matrices_particion(r, entorno):
    """Escribe las matrices de A (imputadas, con pobre en la de entrenamiento) de los
    cinco contextos cv_k de la partición r, con las mismas columnas de la etapa 10."""
    etapa, verdad = entorno["etapa"], entorno["verdad"]
    carpeta, contextos = ruta_matrices(r), definir_contextos_cv()
    pendientes = [c for c in contextos if not all((carpeta / f"{c}_{p}.parquet").exists() for p in ("ent", "eval"))]
    if not pendientes:
        registrar_log(etapa, f"matrices de la partición {r} ya en disco")
        return
    grupos = grupos_particion(r)
    hogar, personas = cargar_hogares(), cargar_personas()
    bloques = construir_grupos(personas, hogar)
    auxiliares = auxiliares_particion(r, grupos, hogar, personas, bloques)
    registrar_log(etapa, f"auxiliares de la partición {r} listos")
    del personas
    gc.collect()
    matriz = armar_matriz({"bloques": bloques}).join(pd.read_parquet(RUTA_AGREGADOS))
    dinamicas = [auxiliares[a] for a in AUXILIARES_A]
    esperadas = columnas_a()
    carpeta.mkdir(parents=True, exist_ok=True)
    for contexto in pendientes:
        entrenamiento, evaluacion = contextos[contexto]
        ids_ent, ids_eval = grupos.index[grupos.isin(entrenamiento)], grupos.index[grupos.isin(evaluacion)]
        x_ent = armar_contexto(matriz, dinamicas, contexto, ids_ent)
        x_eval = armar_contexto(matriz, dinamicas, contexto, ids_eval)
        x_ent, x_eval = imputar_mediana(x_ent, x_eval)
        assert list(x_ent.columns) == esperadas and list(x_eval.columns) == esperadas, \
            f"{contexto}: las columnas no coinciden con las matrices de la etapa 10"
        guardar_parquet_atomico(x_ent.assign(pobre=verdad.loc[ids_ent].astype(np.int64).to_numpy()),
                                carpeta / f"{contexto}_ent.parquet")
        guardar_parquet_atomico(x_eval, carpeta / f"{contexto}_eval.parquet")
        registrar_log(etapa, f"matrices {contexto} de la partición {r}: {x_ent.shape[0]} x {x_ent.shape[1]}")
        del x_ent, x_eval
        gc.collect()


def matrices(entorno):
    """Trabajo `matrices`: grupos, auxiliares anidados y matrices de A de la partición."""
    inicio = time.time()
    construir_matrices_particion(entorno["particion_r"], entorno)
    if time.time() - inicio > 60:
        anotar_tiempo(f"matrices_r{entorno['particion_r']}", (time.time() - inicio) / 60)
    return True


def crear_preparador_particion(r, extras=(), columnas=None):
    """preparar() de la base de A de una partición: lee las matrices del contexto
    (opcionalmente solo unas columnas, en ese orden) y les pega las variables por
    contexto de `extras` ({contexto: DataFrame}, p. ej. KNN fuera de fold)."""
    def preparar(contexto):
        carpeta = ruta_matrices(r)
        x_ent = pd.read_parquet(carpeta / f"{contexto}_ent.parquet",
                                columns=None if columnas is None else list(columnas) + ["pobre"])
        y_ent = x_ent.pop("pobre").to_numpy()
        x_eval = pd.read_parquet(carpeta / f"{contexto}_eval.parquet", columns=columnas)
        if columnas is not None:
            assert list(x_ent.columns) == list(columnas), f"{contexto}: el parquet no respetó el orden de columnas"
        for variables in extras:
            extra = variables[contexto].astype(np.float32)
            x_ent, x_eval = x_ent.join(extra), x_eval.join(extra)
            assert x_ent[extra.columns].notna().all().all() and x_eval[extra.columns].notna().all().all(), \
                f"{contexto}: variables extra con NA"
        return x_ent, y_ent, x_eval

    return preparar


# ---- arboles: LightGBM n6 sobre A, n7_v3, a4_total y A + KNN -----------------------

def variables_knn_particion(r, etapa):
    """{contexto: DataFrame} de las variables KNN fuera de fold de una partición
    (las 60 variables de más gain de la etapa 7; caché por contexto)."""
    from etapa12_enfoques import variables_knn_contexto, variables_top_gain  # noqa: PLC0415
    columnas = variables_top_gain()
    carpeta = carpeta_particion(r) / "knn"
    carpeta.mkdir(parents=True, exist_ok=True)
    salida = {}
    for contexto in definir_contextos_cv():
        ruta = carpeta / f"{contexto}.parquet"
        if not ruta.exists():
            inicio = time.time()
            guardar_parquet_atomico(variables_knn_contexto(contexto, columnas, ruta_matrices=ruta_matrices(r)), ruta)
            registrar_log(etapa, f"knn {contexto} de la partición {r} ({time.time() - inicio:.0f} s)")
        salida[contexto] = pd.read_parquet(ruta)
    return salida


N_AGREGADOS, N_VARIABLES_FUENTES = 1026, 10  # columnas ag_* de A y variables del auxiliar v3


def arboles(entorno):
    """Trabajo `arboles`: LightGBM n6 con bagging sobre A, n7_v3, a4_total y A + KNN
    en los cinco contextos de la partición."""
    r, etapa, verdad = entorno["particion_r"], entorno["etapa"], entorno["verdad"]
    carpeta = carpeta_particion(r)
    todas = columnas_a()
    v3, a4_total = columnas_v3(todas), columnas_a4_total(todas)
    assert len(todas) - len(v3) == N_AGREGADOS and len(v3) - len(a4_total) == N_VARIABLES_FUENTES, \
        "las bases n7_v3 y a4_total no se recuperan de las columnas de A"
    bases = (("a_bag5", None, ()), ("v3_fuentes_bag5", v3, ()), ("a4_total_bag5", a4_total, ()))
    for hoja, columnas, extras in bases:
        nombre = nombre_candidato(r, hoja)
        evaluar_o_cargar_oof(nombre, carpeta, lambda nombre=nombre, columnas=columnas, extras=extras: evaluar_oof(
            nombre, crear_preparador_particion(r, extras, columnas),
            crear_ajuste_lightgbm(PARAMETROS["lgb"], PARAMETROS["semillas_bagging"]), verdad, carpeta))
    nombre = nombre_candidato(r, "a_knn_bag5")

    def evaluar_knn():
        extras = variables_knn_particion(r, etapa)
        return evaluar_oof(nombre, crear_preparador_particion(r, [extras]),
                           crear_ajuste_lightgbm(PARAMETROS["lgb"], PARAMETROS["semillas_bagging"]), verdad, carpeta)

    evaluar_o_cargar_oof(nombre, carpeta, evaluar_knn)
    return True


# ---- catboost: B3 con categóricas nativas sobre A ---------------------------------

def crear_preparador_grupos(matriz, grupos, dinamicas):
    """preparar() de una matriz fija (con categóricas nativas) más los auxiliares
    por contexto, para los grupos de una partición (como crear_preparador())."""
    contextos = definir_contextos_cv()
    verdad = cargar_particion().set_index("id")["pobre"]

    def preparar(contexto):
        entrenamiento, evaluacion = contextos[contexto]
        ids_ent, ids_eval = grupos.index[grupos.isin(entrenamiento)], grupos.index[grupos.isin(evaluacion)]
        x_ent = armar_contexto(matriz, dinamicas, contexto, ids_ent)
        x_eval = armar_contexto(matriz, dinamicas, contexto, ids_eval)
        x_ent, x_eval = imputar_mediana(x_ent, x_eval)
        return x_ent, verdad.loc[ids_ent].astype(int).to_numpy(), x_eval

    return preparar


def catboost(entorno):
    """Trabajo `catboost`: CatBoost B3 (categóricas nativas) sobre A en la partición."""
    from etapa7_exploracion import PARAMETROS_CATBOOST, crear_ajuste_catboost  # noqa: PLC0415
    r, etapa, verdad = entorno["particion_r"], entorno["etapa"], entorno["verdad"]
    carpeta = carpeta_particion(r)
    nombre = nombre_candidato(r, "catboost_b3")
    if cargar_oof(nombre, carpeta) is not None:
        registrar_log(nombre, "OOF desde caché")
        return True
    grupos = grupos_particion(r)
    hogar, personas = cargar_hogares(), cargar_personas()
    bloques = construir_grupos(personas, hogar)
    auxiliares = auxiliares_particion(r, grupos, hogar, personas, bloques)
    del personas
    gc.collect()
    categoricas = [c for b in BLOQUES_A4 for c in bloques[b].columns
                   if isinstance(bloques[b][c].dtype, pd.CategoricalDtype)]
    matriz = armar_matriz({"bloques": bloques}, BLOQUES_A4, nativas=categoricas).join(pd.read_parquet(RUTA_AGREGADOS))
    preparar = crear_preparador_grupos(matriz, grupos, [auxiliares[a] for a in AUXILIARES_A])
    parametros = PARAMETROS["catboost"] or PARAMETROS_CATBOOST
    registrar_log(etapa, f"CatBoost B3 partición {r}: {matriz.shape[1]} columnas fijas, {len(categoricas)} categóricas")
    evaluar_o_cargar_oof(nombre, carpeta, lambda: evaluar_oof(
        nombre, preparar, crear_ajuste_catboost(parametros, categoricas), verdad, carpeta))
    return True


# ---- redes: RealMLP, DeepSets, bags y mezclas fijas --------------------------------

def crear_preparador_ids_particion(grupos):
    """preparar() que solo entrega los ids de cada contexto cv_k de una partición
    (la red de conjuntos busca los datos en su estructura)."""
    contextos = definir_contextos_cv()
    verdad = cargar_particion().set_index("id")["pobre"]

    def preparar(contexto):
        entrenamiento, evaluacion = contextos[contexto]
        ids_ent, ids_eval = grupos.index[grupos.isin(entrenamiento)], grupos.index[grupos.isin(evaluacion)]
        return pd.DataFrame(index=ids_ent), verdad.loc[ids_ent].astype(int).to_numpy(), pd.DataFrame(index=ids_eval)

    return preparar


def redes(entorno):
    """Trabajo `redes`: RealMLP y DeepSets por semilla, sus bags y las mezclas fijas."""
    from etapa12_enfoques import construir_conjuntos  # noqa: PLC0415
    from etapa14_busquedas import CONFIGURACION_REALMLP_ETAPA10, MODO  # noqa: PLC0415
    from funciones.conjuntos import PARAMETROS_CONJUNTOS, crear_ajuste_conjuntos  # noqa: PLC0415
    from funciones.redes import crear_ajuste_realmlp  # noqa: PLC0415
    r, etapa, verdad = entorno["particion_r"], entorno["etapa"], entorno["verdad"]
    carpeta = carpeta_particion(r)

    # 1. RealMLP con la configuración de la etapa 10, una semilla por candidato
    parametros = {**CONFIGURACION_REALMLP_ETAPA10, **MODO["fijos_realmlp"], **PARAMETROS["realmlp_extra"]}
    preparar = crear_preparador_particion(r)
    semillas_realmlp = []
    for k in range(PARAMETROS["semillas_realmlp"]):
        nombre = nombre_candidato(r, f"realmlp_s{k}")
        evaluar_o_cargar_oof(nombre, carpeta, lambda nombre=nombre, k=k: evaluar_oof(
            nombre, preparar, crear_ajuste_realmlp(parametros, SEMILLA + k), verdad, carpeta))
        semillas_realmlp.append(nombre)
    bag_realmlp = nombre_candidato(r, "realmlp_bag5")
    promediar_oof(bag_realmlp, semillas_realmlp, [1 / len(semillas_realmlp)] * len(semillas_realmlp), carpeta)
    registrar_log(etapa, f"{bag_realmlp}: promedio de {len(semillas_realmlp)} semillas")

    # 2. DeepSets con atención (etapa 12)
    conjuntos = construir_conjuntos(cargar_hogares())
    preparar_ids = crear_preparador_ids_particion(grupos_particion(r))
    parametros_conjuntos = {**PARAMETROS_CONJUNTOS, **PARAMETROS["conjuntos_extra"]}
    semillas_conjuntos = []
    for k in range(PARAMETROS["semillas_conjuntos"]):
        nombre = nombre_candidato(r, f"conjuntos_s{k}")
        evaluar_o_cargar_oof(nombre, carpeta, lambda nombre=nombre, k=k: evaluar_oof(
            nombre, preparar_ids, crear_ajuste_conjuntos(conjuntos, parametros_conjuntos, SEMILLA + k), verdad,
            carpeta))
        semillas_conjuntos.append(nombre)
    bag_conjuntos = nombre_candidato(r, "conjuntos_bag3")
    promediar_oof(bag_conjuntos, semillas_conjuntos, [1 / len(semillas_conjuntos)] * len(semillas_conjuntos), carpeta)
    registrar_log(etapa, f"{bag_conjuntos}: promedio de {len(semillas_conjuntos)} semillas")
    del conjuntos
    gc.collect()

    # 3. Mezclas con los pesos fijos de las etapas 10 y 13
    for hoja, componentes in MEZCLAS.items():
        nombres = [nombre_candidato(r, h) for h, _ in componentes]
        if all(cargar_oof(n, carpeta) is not None for n in nombres):
            promediar_oof(nombre_candidato(r, hoja), nombres, [w for _, w in componentes], carpeta)
            registrar_log(etapa, f"{nombre_candidato(r, hoja)}: pesos {[w for _, w in componentes]}")
        else:
            registrar_log(etapa, f"{hoja}: falta alguna OOF de {nombres}; la mezcla se arma en el reporte")
    return True


# ---- knn_original: A + KNN con 5 semillas en la partición original ----------------

def knn_original(entorno):
    """Trabajo `knn_original`: A + KNN fuera de fold con 5 semillas en la partición
    original (OOF y holdout, sin test), para tener su holdout comparable."""
    from etapa12_enfoques import contextos_sin_test, crear_preparador_a, variables_knn  # noqa: PLC0415
    from funciones.bitacora import Presupuesto  # noqa: PLC0415
    etapa, particion, verdad = entorno["etapa"], entorno["particion"], entorno["verdad"]
    nombre = nombre_knn_original()
    if (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
        registrar_log(nombre, "predicciones desde caché")
        return True
    extras = variables_knn(list(contextos_sin_test()), etapa, Presupuesto(10**6))
    inicio = time.time()
    evaluar_modelo(nombre, crear_preparador_a([extras]),
                   crear_ajuste_lightgbm(PARAMETROS["lgb"], PARAMETROS["semillas_bagging"]), particion,
                   con_test=False, reanudable=True)
    borrar_parciales(nombre)
    anotar_tiempo(nombre, (time.time() - inicio) / 60)
    return True


# ---- reporte: tablas de la fase 1 ------------------------------------------------

# Candidatos (hojas) cuyos minutos suman los de cada hoja de la spec
COMPONENTES_TIEMPO = {
    "a_bag5": ["a_bag5"], "v3_fuentes_bag5": ["v3_fuentes_bag5"], "a4_total_bag5": ["a4_total_bag5"],
    "a_knn_bag5": ["a_knn_bag5"], "catboost_b3": ["catboost_b3"],
    "realmlp_bag5": [f"realmlp_s{k}" for k in range(5)], "conjuntos_bag3": [f"conjuntos_s{k}" for k in range(3)],
}
COMPONENTES_TIEMPO["mezcla_lgb_realmlp"] = COMPONENTES_TIEMPO["a_bag5"] + COMPONENTES_TIEMPO["realmlp_bag5"]
COMPONENTES_TIEMPO["mezcla_final"] = COMPONENTES_TIEMPO["mezcla_lgb_realmlp"] + COMPONENTES_TIEMPO["conjuntos_bag3"]


def minutos_candidato(hoja, tiempos):
    """Minutos de entrenamiento de un candidato en la CV repetida: la suma de sus
    componentes en todas las particiones (NaN si no hay ninguno anotado)."""
    valores = [tiempos[nombre_candidato(r, h)] for r in PARTICIONES for h in COMPONENTES_TIEMPO[hoja]
               if nombre_candidato(r, h) in tiempos]
    return float(sum(valores)) if valores else np.nan


def mezclas_pendientes(r):
    """Arma en el reporte las mezclas fijas cuya OOF falte pero tengan sus piezas."""
    carpeta = carpeta_particion(r)
    for hoja, componentes in MEZCLAS.items():
        nombres = [nombre_candidato(r, h) for h, _ in componentes]
        if cargar_oof(nombre_candidato(r, hoja), carpeta) is None and all(cargar_oof(n, carpeta) is not None
                                                                           for n in nombres):
            promediar_oof(nombre_candidato(r, hoja), nombres, [w for _, w in componentes], carpeta)


def folds_repetidos(etapa):
    """Tabla (candidato, particion, fold, umbral, f1, precision, recall) de todas las
    OOF repetidas disponibles."""
    partes = []
    for r in PARTICIONES:
        mezclas_pendientes(r)
        for nombre_spec, (hoja, _) in CANDIDATOS.items():
            oof = cargar_oof(nombre_candidato(r, hoja), carpeta_particion(r))
            if oof is None:
                registrar_log(etapa, f"falta la OOF de {nombre_candidato(r, hoja)}")
                continue
            partes.append(f1_por_fold_con_umbral_oof(oof).assign(candidato=nombre_spec, particion=r))
    if not partes:
        return pd.DataFrame(columns=["candidato", "particion", "fold", "umbral", "f1", "precision", "recall"])
    tabla = pd.concat(partes, ignore_index=True)
    return tabla[["candidato", "particion", "fold", "umbral", "f1", "precision", "recall"]]


def resultados_holdout(verdad, etapa):
    """{candidato de la spec: resultado_desde_predicciones()} de los que tienen
    predicciones del pipeline (umbral OOF original y holdout)."""
    salida = {}
    for nombre_spec, (_, nombre_holdout) in CANDIDATOS.items():
        if nombre_spec == "A + KNN (bag5)":
            nombre_holdout = nombre_knn_original()
        if (RUTA_NIVEL7 / f"predicciones_{nombre_holdout}.parquet").exists():
            salida[nombre_spec] = resultado_desde_predicciones(nombre_holdout, verdad)
        else:
            registrar_log(etapa, f"sin predicciones del pipeline para {nombre_spec} ({nombre_holdout})")
    return salida


def clases_holdout(resultados, ids_holdout):
    """Clase predicha (0/1) de cada candidato en el holdout con su umbral OOF,
    alineada a ids_holdout; exige que todos tengan exactamente esos hogares.

    Argumentos: resultados = {candidato: resultado_desde_predicciones()};
        ids_holdout = índice de hogares del holdout.
    Devuelve: {candidato: Serie 0/1 indexada por ids_holdout}.
    """
    clases = {}
    for nombre, resultado in resultados.items():
        probabilidad = resultado["prediccion_holdout"]
        assert len(probabilidad) == len(ids_holdout) and set(probabilidad.index) == set(ids_holdout), \
            f"{nombre}: su holdout no tiene los mismos hogares que la base"
        clases[nombre] = (probabilidad.reindex(ids_holdout) >= resultado["umbral"]).astype(int)
    return clases


def es_mejora_real(fila, n_esperadas, base):
    """Regla de la etapa: un candidato distinto de la base, con todas sus
    evaluaciones, cuya diferencia pareada es positiva y excluye cero en la CV
    repetida y en el holdout."""
    return bool(fila["candidato"] != base and fila["n_evaluaciones"] == n_esperadas
                and fila["delta_cv_excluye_cero"] and fila["delta_cv"] > 0
                and fila["delta_holdout_excluye_cero"] and fila["delta_holdout"] > 0)


def reporte(entorno):
    """Trabajo `reporte`: tablas de la fase 1 (folds, pareadas, bootstrap, resumen y
    tiempos) y la lista de mejoras reales según la regla de la etapa."""
    etapa, verdad = entorno["etapa"], entorno["verdad"]
    folds = folds_repetidos(etapa)
    folds.to_csv(RUTA_TABLAS / f"{TABLAS}_{FASE}_folds.csv", index=False)
    folds_base = folds[folds["candidato"] == BASE]
    resultados = resultados_holdout(verdad, etapa)
    if BASE not in resultados:
        raise RuntimeError(f"faltan las predicciones de la base {BASE}")
    ids_holdout = resultados[BASE]["prediccion_holdout"].index
    clases = clases_holdout(resultados, ids_holdout)
    bootstrap = bootstrap_holdout(verdad.loc[ids_holdout].astype(int), clases, BASE,
                                  n_replicas=PARAMETROS["replicas"], semilla=SEMILLA)
    bootstrap.to_csv(RUTA_TABLAS / f"{TABLAS}_{FASE}_bootstrap.csv", index=False)
    bootstrap = bootstrap.set_index("candidato")
    tiempos = leer_tiempos()
    n_esperadas = 5 * len(PARTICIONES)

    filas_resumen, filas_pareadas = [], []
    for nombre_spec, (hoja, _) in CANDIDATOS.items():
        propios = folds[folds["candidato"] == nombre_spec]
        resumen = resumir_cv_repetida(propios) if len(propios) else {"f1_medio": np.nan, "f1_sd": np.nan,
                                                                        "f1_ee": np.nan, "n_evaluaciones": 0}
        pareado = comparar_pareado(propios, folds_base) if len(propios) and len(folds_base) else {}
        filas_pareadas.append({"candidato": nombre_spec, "base": BASE, **pareado})
        resultado = resultados.get(nombre_spec)
        boot = bootstrap.loc[nombre_spec] if nombre_spec in bootstrap.index else None
        minutos = minutos_candidato(hoja, tiempos)
        fila = {
            "candidato": nombre_spec, "hoja": hoja, "n_evaluaciones": resumen["n_evaluaciones"],
            "completo": resumen["n_evaluaciones"] == n_esperadas,
            "f1_cv_repetida": resumen["f1_medio"], "f1_cv_repetida_ee": resumen["f1_ee"],
            "f1_cv_repetida_sd": resumen["f1_sd"],
            "f1_oof_original": resultado["f1_cv"] if resultado else np.nan,
            "umbral_oof_original": resultado["umbral"] if resultado else np.nan,
            "f1_holdout": boot["f1"] if boot is not None else np.nan,
            "holdout_ic_inferior": boot["ic_inferior"] if boot is not None else np.nan,
            "holdout_ic_superior": boot["ic_superior"] if boot is not None else np.nan,
            "precision_holdout": resultado["precision_holdout"] if resultado else np.nan,
            "recall_holdout": resultado["recall_holdout"] if resultado else np.nan,
            "delta_cv": pareado.get("delta_medio", np.nan), "delta_cv_ic_inferior": pareado.get("ic_inferior", np.nan),
            "delta_cv_ic_superior": pareado.get("ic_superior", np.nan),
            "delta_cv_excluye_cero": pareado.get("excluye_cero", False), "mde_cv": pareado.get("mde", np.nan),
            "delta_cv_ee_corregido": pareado.get("delta_ee_corregido", np.nan),
            "delta_cv_ic_inferior_corregido": pareado.get("ic_inferior_corregido", np.nan),
            "delta_cv_ic_superior_corregido": pareado.get("ic_superior_corregido", np.nan),
            "delta_cv_excluye_cero_corregido": pareado.get("excluye_cero_corregido", False),
            "mde_cv_corregido": pareado.get("mde_corregido", np.nan),
            "delta_holdout": boot["delta_observado"] if boot is not None else np.nan,
            "delta_holdout_medio_bootstrap": boot["delta_medio"] if boot is not None else np.nan,
            "delta_holdout_ic_inferior": boot["delta_ic_inferior"] if boot is not None else np.nan,
            "delta_holdout_ic_superior": boot["delta_ic_superior"] if boot is not None else np.nan,
            "delta_holdout_excluye_cero": bool(boot["delta_excluye_cero"]) if boot is not None else False,
            "mde_holdout": boot["mde_holdout"] if boot is not None else np.nan,
            "minutos_cv_repetida": minutos,
        }
        fila["mejora_real"] = es_mejora_real(fila, n_esperadas, BASE)
        filas_resumen.append(fila)
    pd.DataFrame(filas_pareadas).to_csv(RUTA_TABLAS / f"{TABLAS}_{FASE}_pareadas.csv", index=False)
    resumen = pd.DataFrame(filas_resumen)
    resumen.to_csv(RUTA_TABLAS / f"{TABLAS}_{FASE}_resumen.csv", index=False)
    pd.DataFrame([{"candidato": k, "minutos": v} for k, v in sorted(tiempos.items())]).to_csv(
        RUTA_TABLAS / f"{TABLAS}_{FASE}_tiempos.csv", index=False)
    columnas = ["candidato", "n_evaluaciones", "completo", "f1_cv_repetida", "f1_cv_repetida_ee", "f1_oof_original",
                "f1_holdout", "holdout_ic_inferior", "holdout_ic_superior", "delta_cv", "delta_cv_ic_inferior",
                "delta_cv_ic_superior", "delta_cv_ic_inferior_corregido", "delta_cv_ic_superior_corregido",
                "delta_holdout", "delta_holdout_ic_inferior", "delta_holdout_ic_superior", "mde_cv",
                "mde_cv_corregido", "mde_holdout", "mejora_real", "minutos_cv_repetida"]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(resumen[columnas].round(4).to_string(index=False))
    reales = resumen.loc[resumen["mejora_real"], "candidato"].tolist()
    incompletos = resumen.loc[~resumen["completo"], "candidato"].tolist()
    registrar_log(etapa, f"mejoras reales (IC pareado excluye cero en CV repetida y holdout): {reales or 'ninguna'}"
                         + (f"; incompletos: {incompletos}" if incompletos else ""))
    return resumen


# ---- Orquestación -------------------------------------------------------------

TRABAJOS = {"matrices": matrices, "arboles": arboles, "catboost": catboost, "redes": redes,
            "knn_original": knn_original, "reporte": reporte}


def planificar_fase1(particiones):
    """Trabajos de la fase 1 con sus dependencias, en orden de prioridad: las
    matrices de la primera partición y el KNN original primero; luego, por
    partición, árboles, redes y CatBoost tras sus matrices; el reporte al final."""
    trabajos = []
    for i, r in enumerate(particiones):
        trabajos.append({"nombre": f"matrices_r{r}", "tipo": "matrices", "particion": r, "depende_de": []})
        if i == 0:
            trabajos.append({"nombre": "knn_original", "tipo": "knn_original", "particion": None, "depende_de": []})
        for tipo in ("arboles", "redes", "catboost"):
            trabajos.append({"nombre": f"{tipo}_r{r}", "tipo": tipo, "particion": r, "depende_de": [f"matrices_r{r}"]})
    trabajos.append({"nombre": "reporte", "tipo": "reporte", "particion": None,
                     "depende_de": [t["nombre"] for t in trabajos]})
    return trabajos


def elegir_lanzables(trabajos, estado, activos, libre_gb, maximo=MAXIMO_PROCESOS,
                     minimo_libre_segundo=MINIMO_LIBRE_SEGUNDO_GB):
    """Trabajos que pueden arrancar ahora: pendientes con todas sus dependencias
    terminadas, en el orden del plan, sin pasar de `maximo` procesos; un segundo
    proceso solo si hay al menos `minimo_libre_segundo` GB libres; nada mientras
    corra un trabajo en modo 'solo' (relanzado tras salir por memoria)."""
    if any(estado.get(a) == "solo" for a in activos):
        return []
    cupos = maximo - len(activos)
    if cupos <= 0 or (activos and libre_gb < minimo_libre_segundo):
        return []
    listos = [t["nombre"] for t in trabajos
              if estado.get(t["nombre"], "pendiente") == "pendiente" and t["nombre"] not in activos
              and all(estado.get(d) == "terminado" for d in t["depende_de"])]
    elegidos = listos[:cupos]
    if len(elegidos) > 1 and libre_gb < minimo_libre_segundo:
        elegidos = elegidos[:1]
    return elegidos


def memoria_libre_gb():
    """Memoria disponible del sistema en GB."""
    return psutil.virtual_memory().available / 1024**3


def estado_para_reanudar(estado):
    """Estado de los trabajos al reanudar fase1: lo que estaba corriendo, en modo
    solo o fallido vuelve a pendiente (todo lo terminado está en disco y se
    retoma desde los parciales); lo demás se conserva."""
    return {nombre: ("pendiente" if valor in ("corriendo", "solo", "fallido") else valor)
            for nombre, valor in estado.items()}


def lanzar_trabajo(trabajo):
    """Subproceso de un trabajo con PS2_NUCLEOS = HILOS_POR_PROCESO y salida en stores/logs/."""
    comando = [sys.executable, str(Path(__file__).resolve()), "trabajo", trabajo["tipo"]]
    if trabajo["particion"] is not None:
        comando += ["--particion", str(trabajo["particion"])]
    if PRUEBA:
        comando.append("--prueba")
    entorno = {**os.environ, "PS2_NUCLEOS": str(HILOS_POR_PROCESO)}
    salida = open(RUTA_LOGS / f"{TABLAS}_{trabajo['nombre']}_salida.txt", "a", encoding="utf-8")
    error = open(RUTA_LOGS / f"{TABLAS}_{trabajo['nombre']}_error.txt", "a", encoding="utf-8")
    proceso = subprocess.Popen(comando, stdout=salida, stderr=error, cwd=RAIZ, env=entorno)
    return proceso, (salida, error)


def fase1():
    """Orquesta la fase 1: lanza los trabajos del plan en subprocesos (máximo dos a
    la vez), relanza en modo solo los que salen por memoria, guarda el estado en
    etapa16/fase1/estado.json y escribe fin.txt al terminar (el reporte corre
    aunque algún trabajo haya fallado, con lo que haya)."""
    etapa = f"{TABLAS}_{FASE}"
    carpeta = CARPETA / FASE
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_estado = carpeta / "estado.json"
    previo = leer_json_tolerante(ruta_estado) or {}
    estado = estado_para_reanudar(previo)
    intentos = {}
    trabajos = planificar_fase1(PARTICIONES)
    inicio = time.time()
    registrar_log(etapa, f"inicio: {len(trabajos)} trabajos, particiones {PARTICIONES}, máximo {MAXIMO_PROCESOS} procesos")
    reintentos = [n for n, v in previo.items() if v != estado.get(n)]
    if reintentos:
        registrar_log(etapa, f"se reintentan: {reintentos}")
    activos = {}

    def guardar_estado():
        escribir_json_atomico(ruta_estado, estado)

    def lanzar(nombre, solo=False):
        trabajo = next(t for t in trabajos if t["nombre"] == nombre)
        activos[nombre] = lanzar_trabajo(trabajo)
        estado[nombre] = "solo" if solo else "corriendo"
        intentos[nombre] = intentos.get(nombre, 0) + 1
        registrar_log(etapa, f"{nombre}: lanzado (intento {intentos[nombre]}{', solo' if solo else ''}; "
                             f"{memoria_libre_gb():.1f} GB libres)")
        guardar_estado()

    while True:
        for nombre, (proceso, archivos) in list(activos.items()):
            codigo = proceso.poll()
            if codigo is None:
                continue
            for archivo in archivos:
                archivo.close()
            del activos[nombre]
            if codigo == 0:
                estado[nombre] = "terminado"
            elif codigo == CODIGO_SALIDA_MEMORIA and intentos[nombre] <= RELANZAMIENTOS:
                estado[nombre] = "pendiente_solo"
            else:
                estado[nombre] = "fallido"
            registrar_log(etapa, f"{nombre}: código {codigo} -> {estado[nombre]} ({(time.time() - inicio) / 60:.1f} min)")
            guardar_estado()
        pendientes_solo = [n for n, e in estado.items() if e == "pendiente_solo"]
        if pendientes_solo and not activos:
            time.sleep(ESPERA_RELANZAMIENTO_S)
            lanzar(pendientes_solo[0], solo=True)
            continue
        if not pendientes_solo:
            for nombre in elegir_lanzables([t for t in trabajos if t["nombre"] != "reporte"], estado, list(activos),
                                           memoria_libre_gb()):
                lanzar(nombre)
        if not activos and not pendientes_solo and not elegir_lanzables(
                [t for t in trabajos if t["nombre"] != "reporte"], estado, [], np.inf):
            break
        time.sleep(INTERVALO_SONDEO_S)

    fallidos = [n for n, e in estado.items() if e == "fallido"]
    if fallidos:
        registrar_log(etapa, f"trabajos fallidos: {fallidos}; el reporte corre con lo que haya")
    lanzar("reporte")
    proceso, archivos = activos.pop("reporte")
    codigo = proceso.wait()
    for archivo in archivos:
        archivo.close()
    estado["reporte"] = "terminado" if codigo == 0 else "fallido"
    guardar_estado()
    registrar_log(etapa, f"fin ({(time.time() - inicio) / 60:.1f} min); reporte código {codigo}")
    (carpeta / "fin.txt").write_text(datetime.now().isoformat(), encoding="utf-8")


def correr_trabajo(tipo, particion_r):
    """Un trabajo en este proceso: vigilante de memoria, log de la fase y código
    de salida 1 si falla."""
    from etapa12_enfoques import correr_seguro  # noqa: PLC0415
    etapa = f"{TABLAS}_{tipo}" + (f"_r{particion_r}" if particion_r is not None else "")
    bitacora.LIMITE_PROCESO_GB = LIMITE_PROCESO_GB
    iniciar_vigilante_memoria(etapa)
    particion = cargar_particion()
    entorno = {"particion": particion, "verdad": particion.set_index("id")["pobre"], "etapa": etapa,
               "particion_r": particion_r}
    inicio = time.time()
    registrar_log(etapa, "inicio")
    salida = correr_seguro(etapa, tipo, lambda: TRABAJOS[tipo](entorno))
    registrar_log(etapa, f"fin ({(time.time() - inicio) / 60:.1f} min){'' if salida is not None else ' con falla'}")
    return 0 if salida is not None else 1


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("comando", choices=["fase1", "trabajo"])
    analizador.add_argument("tipo", nargs="?", choices=TIPOS_TRABAJO, help="tipo de trabajo (con `trabajo`)")
    analizador.add_argument("--particion", type=int, default=None, help="partición repetida (1, 2 o 3)")
    analizador.add_argument("--prueba", action="store_true", help="miniatura para probar el flujo")
    args = analizador.parse_args()
    if args.prueba:
        configurar_prueba()
    CARPETA.mkdir(parents=True, exist_ok=True)
    bitacora.ARCHIVOS_LOG_EXTRA.append(ruta_log_fase())
    if args.comando == "fase1":
        fase1()
    else:
        if args.tipo is None:
            analizador.error("`trabajo` necesita el tipo")
        sys.exit(correr_trabajo(args.tipo, args.particion))
