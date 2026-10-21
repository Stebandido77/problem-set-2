"""Etapa 12: enfoques no probados sobre la base de A, con el protocolo del nivel 7.

Base de comparación: n7_a_agregados_bag5 (F1 CV 0,7748, holdout 0,7806). Mismos
folds y holdout de R, umbral elegido en OOF con buscar_umbral_f1(), auxiliares con
OOF anidada y verificar_sin_fuga(), un solo proceso pesado, guardado reanudable y
corte duro por tiempo en cada experimento.

Cada candidato nuevo se compara primero con 1 semilla contra la referencia de A
de 1 semilla (n12_a_1s: mismo LightGBM n6 y las mismas matrices de la etapa 10);
solo si mejora al menos MEJORA_MINIMA en CV y en holdout se repite con bagging de
5 semillas y contexto test, y solo entonces puede generar CSV (regla de siempre:
+0,002 en CV y en holdout sobre la base bag5).

Experimentos (cada uno con su tope de minutos):
  preparar        base casi cruda para AutoGluon (etapa12_autogluon.py) y la
                  referencia de 1 semilla
  distribucional  LightGBM por cuantiles de log(Ingpcug / Lp) (alfas 0,05 a 0,95)
                  con OOF anidada: P(Ingpcug < Lp) implícita, mediana y ancho
                  intercuantil; clasificación directa y como variables de A
  deepsets        cada hogar como conjunto de personas: codificador por persona,
                  agregación invariante a permutaciones (suma, media, máximo y
                  atención) y cabeza de clasificación; solo y en mezcla con A
  gpboost         boosting con efecto aleatorio por dominio sobre la base de A
  knn             tasa de pobreza de los k vecinos más cercanos (k = 10, 50, 200)
                  en las 60 variables de más gain, fuera de fold, como variables de A
  cierre          tabla final, correlación OOF con LightGBM, CSV y bitácora si
                  algún candidato pasa la regla

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa12_enfoques.py <experimento> [--minutos N]
"""

import argparse
import gc
import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from configuracion import PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa2_optuna import SEMILLAS_BAGGING
from etapa6_auxiliar_v3 import f1_banda
from funciones.agregados import PATRON_ORIGINALES
from funciones.anidado import asignar_grupos, definir_contextos, guardar_parquet_atomico, predecir_anidado, verificar_sin_fuga
from funciones.bitacora import Presupuesto, iniciar_corte_duro, iniciar_vigilante_memoria, registrar_log
from funciones.datos import cargar_hogares, cargar_particion, cargar_personas
from funciones.mezclas import mezclar_por_oof
from funciones.modelado import (borrar_parciales, crear_ajuste_lightgbm, evaluar_modelo,
                                resultado_desde_predicciones)
from funciones.redes import correlaciones_con
from funciones.submission import guardar_submission

BASE = "n7_a_agregados_bag5"
REFERENCIA_1S = "n12_a_1s"
MEJORA_MINIMA = 0.002
RUTA_ETAPA12 = RUTA_NIVEL7 / "etapa12"
RUTA_MATRICES_A = RUTA_NIVEL7 / "etapa10" / "matrices"
ARCHIVO_TIEMPOS = RUTA_ETAPA12 / "tiempos.json"
MINUTOS = {"preparar": 20, "distribucional": 60, "deepsets": 60, "gpboost": 30, "knn": 20, "cierre": 15}
RESERVA_CIERRE_MIN = 3


# ---- Utilidades comunes -------------------------------------------------------

def correr_seguro(etapa, nombre, funcion):
    """Ejecuta funcion(); si falla, anota el error en el log y devuelve None."""
    try:
        return funcion()
    except Exception as error:  # noqa: BLE001 (cualquier falla se registra y se sigue)
        registrar_log(etapa, f"FALLA {nombre}: {type(error).__name__}: {error}"[:300])
        traceback.print_exc()
        return None


def leer_tiempos():
    return json.loads(ARCHIVO_TIEMPOS.read_text(encoding="utf-8")) if ARCHIVO_TIEMPOS.exists() else {}


def anotar_tiempo(nombre, minutos):
    tiempos = leer_tiempos()
    tiempos[nombre] = minutos
    RUTA_ETAPA12.mkdir(parents=True, exist_ok=True)
    ARCHIVO_TIEMPOS.write_text(json.dumps(tiempos, indent=2), encoding="utf-8")


def evaluar_o_cargar(nombre, evaluar, verdad):
    """Resultado desde las predicciones guardadas si el candidato ya terminó; si
    no, lo evalúa y anota su tiempo."""
    if (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
        resultado = resultado_desde_predicciones(nombre, verdad)
        resultado["tiempo_min"] = leer_tiempos().get(nombre, np.nan)
        return resultado
    resultado = evaluar()
    anotar_tiempo(nombre, resultado["tiempo_min"])
    return resultado


def crear_preparador_a(extras=()):
    """preparar() de la base de A: las matrices por contexto de la etapa 10 (1.324
    columnas ya imputadas) más las variables por contexto de `extras`.

    Argumentos: extras = lista de {contexto: DataFrame indexado por id} (OOF anidada).
    Devuelve: función(contexto) -> (x_ent, y_ent, x_eval).
    """
    def preparar(contexto):
        x_ent = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_ent.parquet")
        y_ent = x_ent.pop("pobre").to_numpy()
        x_eval = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_eval.parquet")
        for variables in extras:
            extra = variables[contexto].astype(np.float32)
            x_ent, x_eval = x_ent.join(extra), x_eval.join(extra)
            assert x_ent[extra.columns].notna().all().all() and x_eval[extra.columns].notna().all().all(), \
                f"{contexto}: variables extra con NA"
        return x_ent, y_ent, x_eval

    return preparar


def evaluar_lightgbm_a(nombre, particion, verdad, extras=(), semillas=(SEMILLA,), con_test=False):
    """LightGBM n6 sobre la base de A más variables extra, reanudable por contexto."""
    def evaluar():
        resultado = evaluar_modelo(nombre, crear_preparador_a(extras), crear_ajuste_lightgbm(PARAMETROS_N6, semillas),
                                   particion, con_test=con_test, reanudable=True)
        borrar_parciales(nombre)
        return resultado
    return evaluar_o_cargar(nombre, evaluar, verdad)


def referencia_1s(particion, verdad):
    """Referencia de A con 1 semilla (misma matriz y parámetros que la base, sin bagging)."""
    return evaluar_lightgbm_a(REFERENCIA_1S, particion, verdad)


def mejora_sobre(resultado, base):
    """(delta CV, delta holdout, pasa) de un candidato frente a una base."""
    delta_cv = resultado["f1_cv"] - base["f1_cv"]
    delta_holdout = resultado["f1_holdout"] - base["f1_holdout"]
    return delta_cv, delta_holdout, bool(delta_cv >= MEJORA_MINIMA and delta_holdout >= MEJORA_MINIMA)


def fila_tabla(resultado, verdad, descripcion, base=None, oof_lightgbm=None):
    """Fila de la tabla de la etapa: métricas, banda de Lp, tiempo, deltas y
    correlación OOF con el LightGBM de la base."""
    fila = {"candidato": resultado["nombre"], "descripcion": descripcion, "umbral": resultado["umbral"],
            "f1_cv": resultado["f1_cv"], "f1_cv_sd": resultado.get("f1_cv_sd", np.nan),
            "f1_holdout": resultado["f1_holdout"], "precision_holdout": resultado["precision_holdout"],
            "recall_holdout": resultado["recall_holdout"],
            **f1_banda(verdad, resultado["prediccion_holdout"], resultado["umbral"]),
            "tiempo_min": resultado.get("tiempo_min", np.nan)}
    if base is not None:
        fila["base"] = base["nombre"]
        fila["delta_cv"], fila["delta_holdout"], fila["pasa_regla"] = mejora_sobre(resultado, base)
    if oof_lightgbm is not None:
        correlacion = correlaciones_con(oof_lightgbm, {resultado["nombre"]: resultado["oof"]["probabilidad"]}).iloc[0]
        fila["pearson_oof_con_lightgbm"] = correlacion["pearson_con_lightgbm"]
        fila["spearman_oof_con_lightgbm"] = correlacion["spearman_con_lightgbm"]
    return fila


def guardar_filas(nombre_tabla, filas):
    """Agrega o reemplaza filas (por candidato) en una tabla de la etapa."""
    ruta = RUTA_TABLAS / f"{nombre_tabla}.csv"
    nuevas = pd.DataFrame(filas)
    if ruta.exists():
        previas = pd.read_csv(ruta)
        nuevas = pd.concat([previas[~previas["candidato"].isin(nuevas["candidato"])], nuevas], ignore_index=True)
    nuevas.to_csv(ruta, index=False)
    return nuevas


def contextos_sin_test():
    return {k: v for k, v in definir_contextos().items() if k != "test"}


def auxiliar_anidado(nombre, ajustar_predecir, grupos, con_test):
    """Variables de un auxiliar por contexto con OOF anidada y caché reanudable
    en etapa12/; sin el contexto test hasta que un candidato pase la regla (los
    ajustes compartidos se reutilizan al agregarlo).

    Devuelve: {contexto: DataFrame} ya verificado con verificar_sin_fuga().
    """
    contextos = definir_contextos() if con_test else contextos_sin_test()
    etiqueta = "completo" if con_test else "sin_test"
    ruta = RUTA_ETAPA12 / f"anidado_{nombre}_{etiqueta}.parquet"
    ruta_registro = RUTA_ETAPA12 / f"registro_{nombre}_{etiqueta}.parquet"
    if ruta.exists() and ruta_registro.exists():
        tabla, registro = pd.read_parquet(ruta), pd.read_parquet(ruta_registro)
    else:
        variables, registro = predecir_anidado(ajustar_predecir, grupos, nombre, contextos,
                                               ruta_parcial=RUTA_ETAPA12 / "parciales" / nombre)
        tabla = pd.concat([v.assign(contexto=c) for c, v in variables.items()]).rename_axis("id").reset_index()
        guardar_parquet_atomico(registro, ruta_registro)
        guardar_parquet_atomico(tabla, ruta)
    verificar_sin_fuga(registro, grupos, contextos)
    return {c: g.drop(columns="contexto").set_index("id") for c, g in tabla.groupby("contexto")}


def resultado_desde_probabilidad(nombre, variables, columna, verdad, grupos, invertir=False):
    """Candidato "directo" a partir de una variable de un auxiliar anidado: la OOF
    son las filas de evaluación de cv_1..cv_5 y el holdout las del contexto holdout.

    Argumentos: nombre = candidato; variables = {contexto: DataFrame}; columna =
        la que hace de P(pobre); verdad = pobre por id; grupos = asignar_grupos();
        invertir = True si la variable crece con el ingreso (se usa su negativo).
    Devuelve: resultado como el de evaluar_modelo(), ya guardado.
    """
    signo = -1.0 if invertir else 1.0
    partes = [pd.DataFrame({"probabilidad": signo * variables[f"cv_{k}"].loc[grupos.index[grupos == k], columna],
                            "fold": k}) for k in range(1, 6)]
    oof = pd.concat(partes)
    oof["verdad"] = verdad.loc[oof.index].astype(int)
    holdout = signo * variables["holdout"].loc[grupos.index[grupos == 6], columna]
    resultado = {"nombre": nombre, "oof": oof, "prediccion_holdout": holdout.rename("probabilidad")}
    if "test" in variables:
        resultado["prediccion_test"] = signo * variables["test"].loc[grupos.index[grupos == 0], columna].rename("probabilidad")
    from funciones.modelado import guardar_predicciones
    guardar_predicciones(resultado)
    return resultado_desde_predicciones(nombre, verdad)


# ---- preparar: base casi cruda y referencia -----------------------------------

COLUMNAS_HOGAR_CRUDAS = ["clase", "dominio", "departamento", "cuartos", "dormitorios", "tenencia", "n_personas",
                         "personas_ug", "linea_pobreza", "arriendo", "P5100", "Li", "Fex_c", "Fex_dpto"]


def construir_base_cruda(personas, hogar):
    """Base "casi cruda" para la corrida A de AutoGluon: las variables de hogar
    tal como vienen (renombradas por limpiar_hogares(), con el arriendo unificado)
    y, de cada variable original de personas, el valor del jefe y del cónyuge y
    la media y el máximo del resto y de todos, más los conteos por rol. Sin
    variables construidas, sin auxiliares, con NA tal cual.

    Argumentos: personas = cargar_personas(); hogar = cargar_hogares().
    Devuelve: DataFrame indexado por id (train + test).
    """
    originales = [c for c in personas.columns if re.match(PATRON_ORIGINALES, c)]
    # P6050 (parentesco) cumple el patrón: define el rol y además se agrega como las demás
    p = personas[["id"] + originales].copy()
    parentesco = p["P6050"].to_numpy()
    rol = pd.Series(np.select([parentesco == 1, parentesco == 2], ["jefe", "conyuge"], "resto"), index=p.index)
    partes = [hogar[COLUMNAS_HOGAR_CRUDAS]]
    for nombre_rol in ("jefe", "conyuge"):
        valores = p[rol == nombre_rol].drop_duplicates("id").set_index("id")[originales]
        partes.append(valores.add_prefix(f"{nombre_rol}_"))
    for nombre_rol, filas in (("resto", p[rol == "resto"]), ("todos", p)):
        agregado = filas.groupby("id")[originales].agg(["mean", "max"])
        agregado.columns = [f"{nombre_rol}_{v}_{'media' if e == 'mean' else 'max'}" for v, e in agregado.columns]
        partes.append(agregado)
    conteos = pd.crosstab(p["id"], rol).add_prefix("n_")
    partes.append(conteos)
    base = pd.concat(partes, axis=1).reindex(hogar.index)
    base[[c for c in base.columns if c.startswith("n_")]] = base[[c for c in base.columns if c.startswith("n_")]].fillna(0)
    return base


def preparar(entorno, reloj):
    RUTA_ETAPA12.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_ETAPA12 / "base_cruda.parquet"
    if not ruta.exists():
        base = construir_base_cruda(cargar_personas(), entorno["hogar"])
        guardar_parquet_atomico(base, ruta)
        registrar_log("etapa12", f"base cruda: {base.shape[0]} hogares x {base.shape[1]} columnas ({reloj.transcurrido():.1f} min)")
        del base
        gc.collect()
    referencia = referencia_1s(entorno["particion"], entorno["verdad"])
    registrar_log("etapa12", f"referencia {REFERENCIA_1S}: F1 CV {referencia['f1_cv']:.4f} | holdout "
                             f"{referencia['f1_holdout']:.4f} ({reloj.transcurrido():.1f} min)")
    return referencia


# ---- distribucional: cuantiles de log(Ingpcug / Lp) ---------------------------

ALFAS = tuple(round(a, 2) for a in np.arange(0.05, 0.951, 0.05))
PARAMETROS_CUANTIL = {"objective": "quantile", "learning_rate": 0.05, "num_iterations": 300, "num_leaves": 31,
                      "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                      "verbose": -1}
VARIABLES_CUANTILES = ["p_bajo_linea_q", "mediana_q", "ancho_intercuantil_q"]


def resumir_cuantiles(cuantiles):
    """P(Ingpcug < Lp) implícita, mediana y ancho intercuantil desde los 19 cuantiles
    de log(Ingpcug / Lp) de cada hogar.

    Los cuantiles se ordenan por fila (los ajustes separados pueden cruzarse); la
    probabilidad es la función de distribución en 0 interpolada linealmente entre
    alfas (0 por debajo del cuantil 0,05 y 1 por encima del 0,95).
    Argumentos: cuantiles = matriz hogares x alfas (en el orden de ALFAS).
    Devuelve: matriz hogares x 3 con las VARIABLES_CUANTILES.
    """
    q = np.sort(np.asarray(cuantiles, dtype=float), axis=1)
    alfas = np.asarray(ALFAS)
    p_bajo = np.array([np.interp(0.0, fila, alfas, left=0.0, right=1.0) for fila in q])
    mediana = q[:, int(np.argmin(np.abs(alfas - 0.5)))]
    ancho = q[:, int(np.argmin(np.abs(alfas - 0.75)))] - q[:, int(np.argmin(np.abs(alfas - 0.25)))]
    return np.column_stack([p_bajo, mediana, ancho])


def crear_auxiliar_cuantiles_19(matriz, objetivo, parametros=PARAMETROS_CUANTIL):
    """Función ajustar_predecir() para predecir_anidado(): un LightGBM por cuantil
    de log(Ingpcug / Lp) y las variables de resumir_cuantiles(); guarda además los
    19 cuantiles crudos (cuantil_05_q ... cuantil_95_q) para la calibración.

    Argumentos: matriz = predictores de hogar en dummies (armar_matriz()); objetivo
        = cargar_ingreso_hogar() (solo train); parametros = LightGBM.
    """
    import lightgbm as lgb
    from configuracion import N_HILOS
    from funciones.datos import imputar_mediana

    def ajustar_predecir(ids_entrenamiento, ids_prediccion):
        inicio = time.time()
        x_ent, x_pred = imputar_mediana(matriz.loc[ids_entrenamiento], matriz.loc[ids_prediccion])
        datos = lgb.Dataset(x_ent, label=objetivo.loc[ids_entrenamiento].to_numpy(), free_raw_data=False)
        cuantiles = np.column_stack([
            lgb.train({**parametros, "alpha": alfa, "num_threads": N_HILOS, "seed": SEMILLA}, datos).predict(x_pred)
            for alfa in ALFAS])
        salida = pd.DataFrame(resumir_cuantiles(cuantiles), index=pd.Index(ids_prediccion, name="id"),
                              columns=VARIABLES_CUANTILES)
        crudos = pd.DataFrame(np.sort(cuantiles, axis=1), index=salida.index,
                              columns=[f"cuantil_{int(round(a * 100)):02d}_q" for a in ALFAS])
        registrar_log("etapa12_distribucional", f"cuantiles n={len(x_ent)} ({time.time() - inicio:.0f} s)")
        return salida.join(crudos)

    return ajustar_predecir


def calibracion_cuantiles(variables, grupos, objetivo):
    """Cobertura empírica de cada cuantil en la OOF del 80% (filas de evaluación de
    cv_1..cv_5): fracción de hogares con log(Ingpcug / Lp) por debajo del cuantil."""
    oof = pd.concat([variables[f"cv_{k}"].loc[grupos.index[grupos == k]] for k in range(1, 6)])
    y = objetivo.reindex(oof.index).to_numpy()
    filas = [{"alfa": a, "cobertura_oof": float((y <= oof[f"cuantil_{int(round(a * 100)):02d}_q"].to_numpy()).mean())}
             for a in ALFAS]
    filas.append({"alfa": "p_bajo_linea_q vs pobre", "cobertura_oof": float(np.corrcoef(
        oof["p_bajo_linea_q"], (y < 0).astype(float))[0, 1])})
    return pd.DataFrame(filas)


def matriz_a4(hogar):
    """Los 280 predictores de hogar de a4 (presentes en test), en dummies."""
    from funciones.bases import armar_matriz
    from funciones.variables import construir_grupos
    return armar_matriz({"bloques": construir_grupos(cargar_personas(), hogar)})


def distribucional(entorno, reloj):
    from funciones.datos import cargar_ingreso_hogar
    particion, verdad, grupos, etapa = entorno["particion"], entorno["verdad"], entorno["grupos"], entorno["etapa"]
    referencia = referencia_1s(particion, verdad)
    base = resultado_desde_predicciones(BASE, verdad)
    objetivo = cargar_ingreso_hogar()
    matriz = matriz_a4(entorno["hogar"])
    registrar_log(etapa, f"matriz a4 {matriz.shape} y objetivo listos ({reloj.transcurrido():.1f} min)")

    crear = lambda: crear_auxiliar_cuantiles_19(matriz, objetivo)  # noqa: E731
    aux = auxiliar_anidado("cuantiles_19", crear(), grupos, con_test=False)
    registrar_log(etapa, f"auxiliar de cuantiles sin test listo ({reloj.transcurrido():.1f} min)")
    calibracion_cuantiles(aux, grupos, objetivo).to_csv(RUTA_TABLAS / "etapa12_cuantiles_calibracion.csv", index=False)
    resumen = {k: v[VARIABLES_CUANTILES] for k, v in aux.items()}

    filas = []
    directo = resultado_desde_probabilidad("n12_cuantiles_directo", aux, "p_bajo_linea_q", verdad, grupos)
    filas.append(fila_tabla(directo, verdad, "P(Ingpcug < Lp) implícita de los 19 cuantiles, umbral OOF", base,
                            base["oof"]["probabilidad"]))
    registrar_log(etapa, f"directo: F1 CV {directo['f1_cv']:.4f} | holdout {directo['f1_holdout']:.4f}")

    candidato = evaluar_lightgbm_a("n12_a_cuantiles_1s", particion, verdad, extras=[resumen])
    filas.append(fila_tabla(candidato, verdad, "A + P(<Lp), mediana y ancho intercuantil (1 semilla)", referencia,
                            base["oof"]["probabilidad"]))
    delta_cv, delta_holdout, pasa = mejora_sobre(candidato, referencia)
    registrar_log(etapa, f"A + cuantiles 1 semilla: CV {delta_cv:+.4f}, holdout {delta_holdout:+.4f} -> "
                         f"{'pasa' if pasa else 'no pasa'} a bagging")
    minutos_bag5 = candidato.get("tiempo_min", 5) * len(SEMILLAS_BAGGING) * 7 / 6 + 12
    if pasa and reloj.restante() > minutos_bag5:
        aux = auxiliar_anidado("cuantiles_19", crear(), grupos, con_test=True)
        resumen = {k: v[VARIABLES_CUANTILES] for k, v in aux.items()}
        bag5 = evaluar_lightgbm_a("n12_a_cuantiles_bag5", particion, verdad, extras=[resumen],
                                  semillas=SEMILLAS_BAGGING, con_test=True)
        filas.append(fila_tabla(bag5, verdad, "A + cuantiles (bagging 5 semillas)", base, base["oof"]["probabilidad"]))
    elif pasa:
        registrar_log(etapa, f"no caben los ~{minutos_bag5:.0f} min del bagging (quedan {reloj.restante():.0f})")
    guardar_filas("etapa12_candidatos", filas)
    print(pd.DataFrame(filas).round(4).to_string(index=False))
    return filas


# ---- knn: tasa de pobreza de los vecinos, fuera de fold ------------------------

K_VECINOS = (10, 50, 200)
N_VARIABLES_KNN = 60
FOLDS_INTERNOS_KNN = 5


def variables_top_gain(n=N_VARIABLES_KNN):
    """Las n variables de más gain del LightGBM de A (tabla de la etapa 7)."""
    importancia = pd.read_csv(RUTA_TABLAS / "etapa7_a_importancia.csv")
    return importancia.sort_values("gain", ascending=False)["variable"].head(n).tolist()


def tasas_vecinos(x_referencia, y_referencia, x_consulta):
    """Tasa de pobreza entre los k vecinos más cercanos (euclídea) de cada fila de
    consulta dentro de la referencia, para cada k de K_VECINOS.

    Argumentos: x_referencia, x_consulta = matrices ya estandarizadas (la consulta
        nunca está en la referencia); y_referencia = 0/1.
    Devuelve: matriz consultas x len(K_VECINOS).
    """
    from sklearn.neighbors import NearestNeighbors
    vecinos = NearestNeighbors(n_neighbors=max(K_VECINOS), algorithm="brute", n_jobs=-1).fit(x_referencia)
    indices = vecinos.kneighbors(x_consulta, return_distance=False)
    etiquetas = np.asarray(y_referencia)[indices]
    return np.column_stack([etiquetas[:, :k].mean(axis=1) for k in K_VECINOS])


def variables_knn_contexto(contexto, columnas):
    """Variables KNN de un contexto: las filas de entrenamiento reciben la tasa de
    sus vecinos en los otros FOLDS_INTERNOS_KNN - 1 folds internos (nunca su
    propia etiqueta ni las de su fold interno) y las de evaluación la tasa
    calculada con todo el entrenamiento del contexto. Estandarización con las
    medias y desviaciones del entrenamiento.
    """
    from sklearn.model_selection import StratifiedKFold
    x_ent = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_ent.parquet", columns=columnas + ["pobre"])
    y_ent = x_ent.pop("pobre").to_numpy().astype(int)
    x_eval = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_eval.parquet", columns=columnas)
    media, desvio = x_ent.mean(), x_ent.std().replace(0, 1)
    z_ent = ((x_ent - media) / desvio).to_numpy(np.float32)
    z_eval = ((x_eval - media) / desvio).to_numpy(np.float32)
    nombres = [f"tasa_pobre_knn_{k}" for k in K_VECINOS]
    salida_ent = np.full((len(z_ent), len(K_VECINOS)), np.nan)
    for ajuste, consulta in StratifiedKFold(FOLDS_INTERNOS_KNN, shuffle=True, random_state=SEMILLA).split(z_ent, y_ent):
        salida_ent[consulta] = tasas_vecinos(z_ent[ajuste], y_ent[ajuste], z_ent[consulta])
    salida_eval = tasas_vecinos(z_ent, y_ent, z_eval)
    tabla = pd.concat([pd.DataFrame(salida_ent, index=x_ent.index, columns=nombres),
                       pd.DataFrame(salida_eval, index=x_eval.index, columns=nombres)])
    assert tabla.notna().all().all()
    return tabla


def variables_knn(contextos, etapa, reloj):
    """{contexto: DataFrame} de las variables KNN, con caché por contexto."""
    columnas = variables_top_gain()
    carpeta = RUTA_ETAPA12 / "knn"
    carpeta.mkdir(parents=True, exist_ok=True)
    salida = {}
    for contexto in contextos:
        ruta = carpeta / f"{contexto}.parquet"
        if not ruta.exists():
            inicio = time.time()
            guardar_parquet_atomico(variables_knn_contexto(contexto, columnas), ruta)
            registrar_log(etapa, f"knn {contexto} ({time.time() - inicio:.0f} s; quedan {reloj.restante():.0f} min)")
        salida[contexto] = pd.read_parquet(ruta)
    return salida


def knn(entorno, reloj):
    particion, verdad, etapa = entorno["particion"], entorno["verdad"], entorno["etapa"]
    referencia = referencia_1s(particion, verdad)
    base = resultado_desde_predicciones(BASE, verdad)
    extras = variables_knn(list(contextos_sin_test()), etapa, reloj)
    candidato = evaluar_lightgbm_a("n12_a_knn_1s", particion, verdad, extras=[extras])
    filas = [fila_tabla(candidato, verdad, f"A + tasa de pobreza de k = {K_VECINOS} vecinos en las "
                        f"{N_VARIABLES_KNN} variables de más gain (1 semilla)", referencia, base["oof"]["probabilidad"])]
    delta_cv, delta_holdout, pasa = mejora_sobre(candidato, referencia)
    registrar_log(etapa, f"A + knn 1 semilla: CV {delta_cv:+.4f}, holdout {delta_holdout:+.4f} -> "
                         f"{'pasa' if pasa else 'no pasa'} a bagging")
    minutos_bag5 = candidato.get("tiempo_min", 5) * len(SEMILLAS_BAGGING) * 7 / 6 + 5
    if pasa and reloj.restante() > minutos_bag5:
        extras = variables_knn(list(definir_contextos()), etapa, reloj)
        bag5 = evaluar_lightgbm_a("n12_a_knn_bag5", particion, verdad, extras=[extras], semillas=SEMILLAS_BAGGING,
                                  con_test=True)
        filas.append(fila_tabla(bag5, verdad, "A + knn (bagging 5 semillas)", base, base["oof"]["probabilidad"]))
    elif pasa:
        registrar_log(etapa, f"no caben los ~{minutos_bag5:.0f} min del bagging (quedan {reloj.restante():.0f})")
    guardar_filas("etapa12_candidatos", filas)
    print(pd.DataFrame(filas).round(4).to_string(index=False))
    return filas


# ---- gpboost: efecto aleatorio por dominio -------------------------------------

# Misma estructura de árbol que n6 con menos rondas y más learning rate (el
# producto rondas x tasa se conserva): la aproximación de Laplace por ronda hace
# a GPBoost varias veces más lento que LightGBM
# GPBoost no admite bagging de filas (bagging_freq debe ser 0): se conserva el
# muestreo de columnas por nodo de n6 y el resto de la estructura del árbol.
PARAMETROS_GPBOOST = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "max_depth": 12,
                      "min_data_in_leaf": 171, "feature_fraction_bynode": PARAMETROS_N6["feature_fraction_bynode"],
                      "bagging_freq": 0, "verbose": 0}
RONDAS_GPBOOST = 700


def crear_ajuste_gpboost(dominio, semilla=SEMILLA):
    """Función ajustar() para evaluar_modelo(): GPBoost con intercepto aleatorio por
    dominio (verosimilitud bernoulli-logit). Los dominios del test existen en train.

    Argumentos: dominio = Serie id -> dominio; semilla = semilla del motor.
    Devuelve: función(x_ent, y_ent, x_eval) -> (probabilidad, importancia por gain).
    """
    import gpboost as gpb
    from configuracion import N_HILOS

    def ajustar(x_ent, y_ent, x_eval):
        modelo_grupo = gpb.GPModel(group_data=dominio.reindex(x_ent.index).astype(str).to_numpy(),
                                   likelihood="bernoulli_logit")
        datos = gpb.Dataset(x_ent, label=y_ent)
        booster = gpb.train({**PARAMETROS_GPBOOST, "num_threads": N_HILOS, "seed": semilla}, datos,
                            gp_model=modelo_grupo, num_boost_round=RONDAS_GPBOOST)
        prediccion = booster.predict(data=x_eval, group_data_pred=dominio.reindex(x_eval.index).astype(str).to_numpy(),
                                     pred_latent=False)
        return np.asarray(prediccion["response_mean"]), pd.Series(booster.feature_importance("gain"), index=x_ent.columns)

    return ajustar


def gpboost(entorno, reloj):
    particion, verdad, etapa = entorno["particion"], entorno["verdad"], entorno["etapa"]
    referencia = referencia_1s(particion, verdad)
    base = resultado_desde_predicciones(BASE, verdad)
    dominio = entorno["hogar"]["dominio"]
    nombre = "n12_gpboost_dominio_1s"

    def evaluar():
        resultado = evaluar_modelo(nombre, crear_preparador_a(), crear_ajuste_gpboost(dominio), particion,
                                   reanudable=True)
        borrar_parciales(nombre)
        return resultado
    candidato = evaluar_o_cargar(nombre, evaluar, verdad)
    filas = [fila_tabla(candidato, verdad, f"GPBoost sobre A con efecto aleatorio por dominio ({RONDAS_GPBOOST} rondas, "
                        f"lr {PARAMETROS_GPBOOST['learning_rate']}, 1 semilla)", referencia, base["oof"]["probabilidad"])]
    delta_cv, delta_holdout, pasa = mejora_sobre(candidato, referencia)
    registrar_log(etapa, f"gpboost 1 semilla: CV {delta_cv:+.4f}, holdout {delta_holdout:+.4f} -> "
                         f"{'pasa' if pasa else 'no pasa'} a bagging")
    minutos_bag5 = candidato.get("tiempo_min", 10) * len(SEMILLAS_BAGGING) * 7 / 6 + 3
    if pasa and reloj.restante() > minutos_bag5:
        def evaluar_bag5():
            ajustes = [crear_ajuste_gpboost(dominio, s) for s in SEMILLAS_BAGGING]

            def ajustar(x_ent, y_ent, x_eval):
                salidas = [a(x_ent, y_ent, x_eval) for a in ajustes]
                return np.mean([p for p, _ in salidas], axis=0), sum(g for _, g in salidas) / len(salidas)
            resultado = evaluar_modelo("n12_gpboost_dominio_bag5", crear_preparador_a(), ajustar, particion,
                                       con_test=True, reanudable=True)
            borrar_parciales("n12_gpboost_dominio_bag5")
            return resultado
        bag5 = evaluar_o_cargar("n12_gpboost_dominio_bag5", evaluar_bag5, verdad)
        filas.append(fila_tabla(bag5, verdad, "GPBoost por dominio (bagging 5 semillas)", base, base["oof"]["probabilidad"]))
    elif pasa:
        registrar_log(etapa, f"no caben los ~{minutos_bag5:.0f} min del bagging (quedan {reloj.restante():.0f})")
    guardar_filas("etapa12_candidatos", filas)
    print(pd.DataFrame(filas).round(4).to_string(index=False))
    return filas


# ---- deepsets: cada hogar como conjunto de personas ----------------------------

SEMILLAS_CONJUNTOS = 3
PESOS_MEZCLA = [(1 - w, w) for w in np.round(np.arange(0.0, 0.501, 0.05), 2)]


def construir_conjuntos(hogar):
    """Estructura de conjuntos: variables de persona (presentes en test) y de hogar
    (nivel 3 en dummies, NA numéricos a la mediana de train)."""
    from funciones.bases import armar_matriz
    from funciones.conjuntos import Conjuntos, preparar_personas_conjuntos
    from funciones.variables import construir_grupos
    personas = cargar_personas()
    x_personas, ids = preparar_personas_conjuntos(personas)
    x_hogar = armar_matriz({"bloques": construir_grupos(personas, hogar)}, bloques=["nivel_3"])
    ids_train = cargar_particion().query("base == 'train'")["id"]
    x_hogar = x_hogar.fillna(x_hogar.loc[ids_train].median(numeric_only=True)).fillna(0)
    return Conjuntos(x_personas, ids, x_hogar.astype(np.float32))


def crear_preparador_ids(grupos):
    """preparar() que solo entrega los ids de cada contexto (la red busca los datos
    en la estructura de conjuntos)."""
    contextos = definir_contextos()
    verdad = cargar_particion().set_index("id")["pobre"]

    def preparar(contexto):
        entrenamiento, evaluacion = contextos[contexto]
        ids_ent = grupos.index[grupos.isin(entrenamiento)]
        ids_eval = grupos.index[grupos.isin(evaluacion)]
        return (pd.DataFrame(index=ids_ent), verdad.loc[ids_ent].astype(int).to_numpy(), pd.DataFrame(index=ids_eval))

    return preparar


def deepsets(entorno, reloj):
    from funciones.conjuntos import PARAMETROS_CONJUNTOS, crear_ajuste_conjuntos
    from funciones.mezclas import promediar_semillas
    particion, verdad, grupos, etapa = entorno["particion"], entorno["verdad"], entorno["grupos"], entorno["etapa"]
    base = resultado_desde_predicciones(BASE, verdad)
    conjuntos = construir_conjuntos(entorno["hogar"])
    registrar_log(etapa, f"conjuntos: {conjuntos.x_personas.shape[1]} variables por persona, "
                         f"{conjuntos.x_hogar.shape[1]} de hogar ({reloj.transcurrido():.1f} min)")
    preparar = crear_preparador_ids(grupos)
    historias, filas, terminadas = [], [], []
    for k in range(SEMILLAS_CONJUNTOS):
        nombre = f"n12_conjuntos_s{k}"
        ya = (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists()
        estimado = (leer_tiempos().get(terminadas[-1], 15) if terminadas else 15) + 2
        if not ya and k > 0 and reloj.restante() < estimado + 8:
            registrar_log(etapa, f"semilla {k}: no cabe (~{estimado:.0f} min, quedan {reloj.restante():.0f})")
            break

        def evaluar(nombre=nombre, k=k):
            resultado = evaluar_modelo(nombre, preparar, crear_ajuste_conjuntos(conjuntos, PARAMETROS_CONJUNTOS,
                                                                                 SEMILLA + k, historias),
                                       particion, con_test=True, reanudable=True)
            borrar_parciales(nombre)
            return resultado
        resultado = evaluar_o_cargar(nombre, evaluar, verdad)
        terminadas.append(nombre)
        registrar_log(etapa, f"{nombre}: F1 CV {resultado['f1_cv']:.4f} | holdout {resultado['f1_holdout']:.4f} "
                             f"({reloj.transcurrido():.1f} min)")
        if k == 0:
            filas.append(fila_tabla(resultado, verdad, "DeepSets con atención (1 semilla)", base, base["oof"]["probabilidad"]))
    if historias:
        pd.concat([pd.DataFrame(h).assign(ajuste=i) for i, h in enumerate(historias)]).to_csv(
            RUTA_TABLAS / "etapa12_conjuntos_historia.csv", index=False)
    nombre_final = terminadas[0]
    if len(terminadas) > 1:
        nombre_final = f"n12_conjuntos_bag{len(terminadas)}"
        resultado = promediar_semillas(nombre_final, terminadas, verdad)
        resultado["tiempo_min"] = sum(leer_tiempos().get(n, np.nan) for n in terminadas)
        filas.append(fila_tabla(resultado, verdad, f"DeepSets con atención (bagging {len(terminadas)} semillas)", base,
                                base["oof"]["probabilidad"]))
    mezcla, pesos, tabla_pesos = mezclar_por_oof("n12_mezcla_lgb_conjuntos", [BASE, nombre_final], PESOS_MEZCLA, verdad)
    tabla_pesos.to_csv(RUTA_TABLAS / "etapa12_pesos_lgb_conjuntos.csv", index=False)
    filas.append(fila_tabla(mezcla, verdad, f"mezcla {BASE} {pesos[0]:.2f} + {nombre_final} {pesos[1]:.2f} (pesos por OOF)",
                            base, base["oof"]["probabilidad"]))
    registrar_log(etapa, f"mezcla con pesos {pesos}: F1 CV {mezcla['f1_cv']:.4f} | holdout {mezcla['f1_holdout']:.4f}")
    guardar_filas("etapa12_candidatos", filas)
    print(pd.DataFrame(filas).round(4).to_string(index=False))
    return filas


# ---- cierre: tabla final, CSV y bitácora ----------------------------------------

DESCRIPCIONES_AUTOGLUON = {
    "n12_autogluon_a": ("AutoGluon best_quality sobre la base casi cruda (hogar + agregados simples de personas), "
                        "bagging por los 5 folds de R", "AutoML (AutoGluon)"),
    "n12_autogluon_b": ("AutoGluon best_quality sobre la base de A (auxiliares de la OOF anidada), bagging por los "
                        "5 folds de R", "AutoML (AutoGluon)"),
}
ALGORITMO_CANDIDATO = {"n12_conjuntos": "Red neuronal", "n12_mezcla": "Stacking", "n12_gpboost": "Boosting",
                       "n12_cuantiles_directo": "Regresión por cuantiles"}


def algoritmo_de(nombre):
    for prefijo, algoritmo in ALGORITMO_CANDIDATO.items():
        if nombre.startswith(prefijo):
            return algoritmo
    return DESCRIPCIONES_AUTOGLUON.get(nombre, ("", "Boosting"))[1]


def cierre(entorno, reloj):
    particion, verdad, etapa = entorno["particion"], entorno["verdad"], entorno["etapa"]
    base = resultado_desde_predicciones(BASE, verdad)
    oof_lightgbm = base["oof"]["probabilidad"]
    ruta_candidatos = RUTA_TABLAS / "etapa12_candidatos.csv"
    candidatos = pd.read_csv(ruta_candidatos) if ruta_candidatos.exists() else pd.DataFrame(columns=["candidato"])
    descripciones = dict(zip(candidatos.get("candidato", []), candidatos.get("descripcion", [])))
    tiempos_candidatos = dict(zip(candidatos.get("candidato", []), candidatos.get("tiempo_min", [])))
    for nombre, (descripcion, _) in DESCRIPCIONES_AUTOGLUON.items():
        if (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
            descripciones[nombre] = descripcion
    referencia = referencia_1s(particion, verdad)
    filas = [fila_tabla(base, verdad, "base: A (LightGBM n6, bagging 5 semillas)", base, oof_lightgbm),
             fila_tabla(referencia, verdad, "referencia: A con 1 semilla", base, oof_lightgbm)]
    for nombre, descripcion in descripciones.items():
        if nombre in (BASE, REFERENCIA_1S) or not (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
            continue
        resultado = resultado_desde_predicciones(nombre, verdad)
        tiempo = RUTA_ETAPA12 / f"tiempo_{nombre}.json"
        resultado["tiempo_min"] = (json.loads(tiempo.read_text(encoding="utf-8"))["minutos"] if tiempo.exists()
                                   else leer_tiempos().get(nombre, tiempos_candidatos.get(nombre, np.nan)))
        contra = referencia if nombre.endswith("_1s") else base
        fila = fila_tabla(resultado, verdad, descripcion, contra, oof_lightgbm)
        fila["delta_cv_vs_bag5"] = resultado["f1_cv"] - base["f1_cv"]
        fila["delta_holdout_vs_bag5"] = resultado["f1_holdout"] - base["f1_holdout"]
        fila["cumple_regla_csv"] = bool(fila["delta_cv_vs_bag5"] >= MEJORA_MINIMA
                                        and fila["delta_holdout_vs_bag5"] >= MEJORA_MINIMA
                                        and "prediccion_test" in resultado)
        fila["tiene_test"] = "prediccion_test" in resultado
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / "etapa12_resumen.csv", index=False)

    ids_test = particion.loc[particion["base"] == "test", "id"]
    generados = []
    for fila in filas:
        if not fila.get("cumple_regla_csv", False):
            continue
        nombre = fila["candidato"]
        resultado = resultado_desde_predicciones(nombre, verdad)
        ruta, tasa = guardar_submission(resultado, f"{nombre}.csv", ids_test,
                                        f"{fila['descripcion']} | umbral {resultado['umbral']:.4f} (OOF)",
                                        algoritmo=algoritmo_de(nombre))
        generados.append(ruta.name)
        registrar_log(etapa, f"CSV {ruta.name} (tasa de pobres {tasa:.3f})")
    if not generados:
        registrar_log(etapa, "ningún candidato cumple la regla (+0,002 en CV y holdout sobre la base): sin CSV")
    columnas = ["candidato", "f1_cv", "f1_cv_sd", "f1_holdout", "precision_holdout", "recall_holdout", "f1_banda_08_12",
                "tiempo_min", "delta_cv_vs_bag5", "delta_holdout_vs_bag5", "pearson_oof_con_lightgbm"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(tabla[[c for c in columnas if c in tabla.columns]].round(4).to_string(index=False))
    return tabla


# ---- Orquestación -------------------------------------------------------------

EXPERIMENTOS = {"preparar": preparar, "distribucional": distribucional, "deepsets": deepsets,
                "gpboost": gpboost, "knn": knn, "cierre": cierre}


def main(experimento, minutos):
    RUTA_ETAPA12.mkdir(parents=True, exist_ok=True)
    etapa = f"etapa12_{experimento}"
    reloj = Presupuesto(minutos)
    registrar_log(etapa, f"inicio ({minutos} min)")
    iniciar_vigilante_memoria(etapa)
    desactivar = threading.Event()
    iniciar_corte_duro(etapa, reloj, desactivar)
    particion = cargar_particion()
    entorno = {"particion": particion, "verdad": particion.set_index("id")["pobre"], "hogar": cargar_hogares(),
               "grupos": asignar_grupos(particion), "reloj": reloj, "etapa": etapa}
    salida = correr_seguro(etapa, experimento, lambda: EXPERIMENTOS[experimento](entorno, reloj))
    desactivar.set()
    registrar_log(etapa, f"fin ({reloj.transcurrido():.1f} min){'' if salida is not None else ' con falla'}")


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("experimento", choices=list(MINUTOS))
    analizador.add_argument("--minutos", type=float, default=None, help="tope de minutos (por defecto el de la etapa)")
    args = analizador.parse_args()
    main(args.experimento, args.minutos or MINUTOS[args.experimento])
