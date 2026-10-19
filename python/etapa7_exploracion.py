"""Etapa 7: ¿hay señal no explotada que acerque el F1 a 0,85?

Base de comparación: n7_v3_fuentes_bag5 (F1 CV 0,7738, holdout 0,7792). Mismos
folds, holdout, umbral OOF y OOF anidada de los auxiliares.

  A. Agregación a lo bruto (máximo 60 min): agregados por rol (jefe, cónyuge,
     resto, todos) de las 56 variables originales de personas del test, sobre la
     base de n7_v3, con LightGBM (hiperparámetros de n6, bagging de 5 semillas).
     Si un ajuste con todas las columnas no cabe en el tiempo, dentro de cada
     contexto un LightGBM rápido ajustado solo con el entrenamiento del contexto
     se queda con los K_FILTRO agregados de más gain (sin fuga).
  B. Comparación de algoritmos sobre la misma base (máximo 90 min), o sobre la
     de A si A sube en CV y en holdout:
       B1 ExtraTrees (bagging de árboles) con valores por defecto razonables;
       B2 el mismo ExtraTrees con variables e hiperparámetros elegidos por una
          colonia artificial de abejas (aptitud: F1 OOF con 3 folds sobre el 80%);
       B3 CatBoost con categóricas nativas;
       B4 promedio LightGBM + CatBoost con peso elegido en OOF, si CatBoost queda
          a menos de 0,01 de LightGBM en F1 CV.
     B2 corre al final y usa el tiempo que dejen los demás.

Reglas: una mejora cuenta solo si sube en CV Y en holdout frente a n7_v3; solo
esos candidatos generan CSV. Si un candidato falla, se anota en el log y se
sigue con el siguiente. Reanudable: auxiliares, agregados, filtros, cada fold de
cada candidato y cada evaluación de la colonia quedan en disco.

Memoria: un hilo vigilante sale limpio (código 3) si el proceso pasa de 6 GB o
el sistema queda con menos de 2 GB libres; CatBoost corre con hilos y RAM
acotados y la colonia evalúa sobre una submuestra estratificada del 80%.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa7_exploracion.py
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from configuracion import N_HILOS, PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa2_optuna import SEMILLAS_BAGGING
from etapa4_leaderboard import (RUTA_LEADERBOARD_EXTRA, anexar_filas_extra, armar_fila, formatear_como_r,
                                texto_hiperparametros)
from etapa6_auxiliar_v3 import f1_banda
from funciones.abejas import ColoniaAbejas
from funciones.agregados import PREFIJO, construir_agregados
from funciones.anidado import guardar_parquet_atomico
from funciones.bases import BLOQUES_A4, armar_matriz, cargar_insumos
from funciones.bitacora import Presupuesto, iniciar_vigilante_memoria, registrar_log
from funciones.datos import cargar_personas
from funciones.metricas import buscar_umbral_f1
from funciones.modelado import (ajustar_lightgbm, borrar_parciales, crear_ajuste_lightgbm, crear_preparador,
                                evaluar_modelo, guardar_predicciones, resultado_desde_predicciones)
from funciones.submission import guardar_submission

BASE = "n7_v3_fuentes_bag5"
AUXILIARES_BASE = ["laboral", "total", "fuentes"]
MINUTOS_A = 60
MINUTOS_B = 90
RUTA_ETAPA7 = RUTA_NIVEL7 / "etapa7"

# A: filtro dentro del contexto si el ajuste completo no cabe en el tiempo
K_FILTRO = 150
PARAMETROS_FILTRO = {"objective": "binary", "learning_rate": 0.1, "num_iterations": 300, "num_leaves": 31,
                     "min_data_in_leaf": 100, "feature_fraction": 0.5, "bagging_fraction": 0.8,
                     "bagging_freq": 1, "verbose": -1}

# B: ExtraTrees con bootstrap (bagging) y CatBoost
PARAMETROS_ET_DEFECTO = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 10,
                         "bootstrap": True, "max_samples": None}
# max_ctr_complexity = 1: sin combinaciones de categóricas (las combinaciones son lo
# que más memoria pide); hilos y RAM acotados para no ahogar el equipo
PARAMETROS_CATBOOST = {"iterations": 1500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3,
                       "max_ctr_complexity": 1, "thread_count": min(N_HILOS, 10), "used_ram_limit": "3gb"}
DISTANCIA_MEZCLA = 0.01
PESOS_MEZCLA = np.round(np.arange(0, 1.0001, 0.05), 2)

# B2: colonia de abejas. Rangos de los hiperparámetros (min_samples_leaf en log)
RANGOS_ABC = {"n_estimators": (100, 300), "max_features": (0.05, 0.6), "max_samples": (0.3, 1.0),
              "min_samples_leaf": (5, 60)}
ABC_FUENTES = 6
ABC_LIMITE = 4
ABC_PLIEGUES = 3
ABC_MAX_CICLOS = 8
# La aptitud se mide sobre esta fracción del 80% (estratificada por pobre)
ABC_SUBMUESTRA = 0.5

ALGORITMO_CURSO = {"lightgbm": "Boosting", "extratrees": "Random Forest", "catboost": "Boosting",
                   "mezcla": "Boosting"}


# ---- Utilidades ---------------------------------------------------------------

def correr_seguro(etapa, nombre, funcion):
    """Ejecuta funcion(); si falla, anota el error en el log y devuelve None
    (para seguir con el siguiente candidato en vez de detener la noche)."""
    try:
        return funcion()
    except Exception as error:  # noqa: BLE001 (cualquier falla se registra y se sigue)
        registrar_log(etapa, f"FALLA {nombre}: {type(error).__name__}: {error}"[:300])
        traceback.print_exc()
        return None


def leer_tiempos():
    """Minutos de cada candidato de la etapa (para reportar al retomar)."""
    ruta = RUTA_ETAPA7 / "tiempos.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def anotar_tiempo(nombre, minutos):
    """Guarda los minutos de un candidato en tiempos.json."""
    tiempos = leer_tiempos()
    tiempos[nombre] = minutos
    (RUTA_ETAPA7 / "tiempos.json").write_text(json.dumps(tiempos, indent=1), encoding="utf-8")


def evaluar_o_cargar(nombre, evaluar, verdad):
    """Resultado de un candidato: desde sus predicciones si ya terminó, si no evaluar().

    Argumentos: nombre = candidato; evaluar = función sin argumentos que corre
        evaluar_modelo(); verdad = pobre por id.
    Devuelve: diccionario de resultado (con tiempo_min).
    """
    if (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
        registrar_log(nombre, "desde predicciones guardadas")
        resultado = resultado_desde_predicciones(nombre, verdad)
        resultado["tiempo_min"] = leer_tiempos().get(nombre, np.nan)
        return resultado
    resultado = evaluar()
    anotar_tiempo(nombre, resultado["tiempo_min"])
    return resultado


def fila_tabla(resultado, verdad, descripcion, base=None):
    """Fila de las tablas de A y B: métricas, banda de Lp, tiempo y si supera a la base."""
    fila = {"candidato": resultado["nombre"], "descripcion": descripcion, "umbral": resultado["umbral"],
            "f1_cv": resultado["f1_cv"], "f1_cv_sd": resultado["f1_cv_sd"],
            "f1_holdout": resultado["f1_holdout"], "precision_holdout": resultado["precision_holdout"],
            "recall_holdout": resultado["recall_holdout"],
            **f1_banda(verdad, resultado["prediccion_holdout"], resultado["umbral"]),
            "tiempo_min": resultado.get("tiempo_min", np.nan)}
    if base is not None:
        fila["delta_cv"] = fila["f1_cv"] - base["f1_cv"]
        fila["delta_holdout"] = fila["f1_holdout"] - base["f1_holdout"]
        fila["supera_base"] = bool(fila["delta_cv"] > 0 and fila["delta_holdout"] > 0)
    return fila


def con_columnas(preparar, columnas):
    """Envuelve preparar() para quedarse con un subconjunto fijo de columnas."""
    def preparar_subconjunto(contexto):
        x_ent, y_ent, x_eval = preparar(contexto)
        return x_ent[columnas], y_ent, x_eval[columnas]
    return preparar_subconjunto


# ---- Experimento A ------------------------------------------------------------

def cargar_o_construir_agregados(ids_hogares):
    """Agregados por rol de las variables originales de personas (caché en parquet)."""
    ruta = RUTA_ETAPA7 / "agregados_personas.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    agregados = construir_agregados(cargar_personas(), ids_hogares)
    guardar_parquet_atomico(agregados, ruta)
    return agregados


def agregados_filtrados(contexto, x_ent, y_ent, k):
    """Los k agregados de más gain en un LightGBM rápido ajustado solo con el
    entrenamiento del contexto (caché por contexto en etapa7/filtro/).

    Argumentos: contexto = nombre; x_ent, y_ent = entrenamiento del contexto; k = cuántos.
    Devuelve: lista de columnas ag_* elegidas.
    """
    ruta = RUTA_ETAPA7 / "filtro" / f"{contexto}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    modelo = lgb.train({**PARAMETROS_FILTRO, "num_threads": N_HILOS, "seed": SEMILLA},
                       lgb.Dataset(x_ent, label=y_ent))
    ganancia = pd.Series(modelo.feature_importance("gain"), index=x_ent.columns)
    ganancia = ganancia[ganancia.index.str.startswith(PREFIJO) & (ganancia > 0)]
    elegidas = ganancia.sort_values(ascending=False).index[:k].tolist()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(elegidas), encoding="utf-8")
    temporal.replace(ruta)
    registrar_log("etapa7_a", f"filtro {contexto}: {len(elegidas)} agregados")
    return elegidas


def crear_filtro(preparar, k):
    """Envuelve preparar(): todas las columnas de la base más los k agregados que
    elige agregados_filtrados() en ese contexto. El filtro nunca ve las filas de
    evaluación del contexto, así que no hay fuga.
    """
    def preparar_filtrado(contexto):
        x_ent, y_ent, x_eval = preparar(contexto)
        elegidas = agregados_filtrados(contexto, x_ent, y_ent, k)
        columnas = [c for c in x_ent.columns if not c.startswith(PREFIJO)] + elegidas
        return x_ent[columnas], y_ent, x_eval[columnas]
    return preparar_filtrado


def decidir_variante(preparar_completo, reloj):
    """Mide un ajuste de LightGBM (n6, una semilla) con todas las columnas en el
    contexto holdout y decide si los 35 ajustes (7 contextos x 5 semillas) caben
    en el presupuesto de A. La decisión se guarda para que un reinicio no cambie.

    Devuelve: diccionario con variante ('completa' o 'filtro') y la medición.
    """
    ruta = RUTA_ETAPA7 / "decision_a.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    x_ent, y_ent, _ = preparar_completo("holdout")
    inicio = time.time()
    ajustar_lightgbm(x_ent, y_ent, PARAMETROS_N6, SEMILLA)
    segundos = time.time() - inicio
    estimado = segundos * len(SEMILLAS_BAGGING) * 7 / 60 * 1.1
    variante = "completa" if reloj.transcurrido() + estimado <= MINUTOS_A else "filtro"
    decision = {"variante": variante, "columnas": x_ent.shape[1], "segundos_por_ajuste": segundos,
                "minutos_estimados_completa": estimado}
    ruta.write_text(json.dumps(decision, indent=1), encoding="utf-8")
    registrar_log("etapa7_a", f"{x_ent.shape[1]} columnas, {segundos:.0f} s por ajuste, "
                              f"completa ~{estimado:.0f} min -> {variante}")
    return decision


def ganancia_por_variable(importancia):
    """Gain de los agregados sumado por variable original y rol (qué explica la ganancia)."""
    nuevas = importancia[importancia.index.str.startswith(PREFIJO)]
    partes = nuevas.index.str.removeprefix(PREFIJO).str.split("_", n=2)
    tabla = pd.DataFrame({"rol": [p[0] for p in partes], "variable": [p[1] for p in partes],
                          "gain": nuevas.to_numpy()})
    total = importancia.sum()
    resumen = tabla.groupby(["variable", "rol"], as_index=False)["gain"].sum()
    resumen["share_gain_total"] = resumen["gain"] / total
    return resumen.sort_values("gain", ascending=False)


def experimento_a(entorno):
    """Corre A y devuelve lo que B necesita para usar su base si A mejora."""
    reloj = Presupuesto(MINUTOS_A)
    registrar_log("etapa7_a", "inicio")
    agregados = cargar_o_construir_agregados(entorno["matriz"].index)
    registrar_log("etapa7_a", f"agregados listos: {agregados.shape[1]} columnas")
    matriz_a = entorno["matriz"].join(agregados)
    preparar_completo = crear_preparador(matriz_a, entorno["particion"], entorno["dinamicas"])
    decision = decidir_variante(preparar_completo, reloj)
    if decision["variante"] == "completa":
        nombre, preparar = "n7_a_agregados_bag5", preparar_completo
        descripcion = f"n7_v3 + {agregados.shape[1]} agregados por rol"
    else:
        nombre, preparar = "n7_a_agregados_filtro_bag5", crear_filtro(preparar_completo, K_FILTRO)
        descripcion = (f"n7_v3 + {K_FILTRO} de {agregados.shape[1]} agregados por rol "
                       "(filtro por gain dentro de cada contexto)")

    resultado = evaluar_o_cargar(nombre, lambda: evaluar_modelo(
        nombre, preparar, crear_ajuste_lightgbm(PARAMETROS_N6, SEMILLAS_BAGGING), entorno["particion"],
        con_test=True, reanudable=True), entorno["verdad"])

    ruta_importancia = RUTA_TABLAS / "etapa7_a_importancia.csv"
    if len(resultado.get("importancia", [])):
        importancia = resultado["importancia"].sort_values(ascending=False)
        pd.DataFrame({"variable": importancia.index, "gain": importancia.to_numpy(),
                      "share_gain": importancia.to_numpy() / importancia.sum(),
                      "nueva": importancia.index.str.startswith(PREFIJO)}).to_csv(ruta_importancia, index=False)
        ganancia_por_variable(importancia).to_csv(RUTA_TABLAS / "etapa7_a_ganancia_por_variable.csv", index=False)

    filas = [fila_tabla(entorno["base"], entorno["verdad"], "base: n7_v3 (a4 + auxiliares v2 y v3)"),
             fila_tabla(resultado, entorno["verdad"], descripcion, entorno["base"])]
    pd.DataFrame(filas).to_csv(RUTA_TABLAS / "etapa7_a.csv", index=False)
    mejora = filas[1]["supera_base"]
    registrar_log("etapa7_a", f"{nombre}: CV {filas[1]['delta_cv']:+.4f} holdout {filas[1]['delta_holdout']:+.4f}"
                              f" -> {'mejora' if mejora else 'no mejora'} ({reloj.transcurrido():.1f} min)")

    if decision["variante"] == "completa":
        variables_a = list(agregados.columns)
    else:
        variables_a = json.loads((RUTA_ETAPA7 / "filtro" / "holdout.json").read_text(encoding="utf-8"))
    entorno["descripciones"][nombre] = {
        "algoritmo": "lightgbm", "especificacion": f"LightGBM (n6, bagging 5 semillas) | {descripcion}",
        "variables": entorno["variables_base"] + variables_a,
        "busqueda": "hiperparámetros de n6_lightgbm fijos", "n_configuraciones": 1,
        "hiperparametros": texto_hiperparametros(PARAMETROS_N6)}
    return {"nombre": nombre, "mejora": mejora, "variante": decision["variante"],
            "preparar": preparar, "variables": entorno["variables_base"] + variables_a}


# ---- Experimento B ------------------------------------------------------------

def crear_ajuste_extratrees(parametros):
    """Función ajustar() de ExtraTrees para evaluar_modelo()."""
    def ajustar(x_ent, y_ent, x_eval):
        modelo = ExtraTreesClassifier(**parametros, n_jobs=N_HILOS, random_state=SEMILLA)
        modelo.fit(x_ent.to_numpy(np.float32), y_ent)
        probabilidad = modelo.predict_proba(x_eval.to_numpy(np.float32))[:, 1]
        return probabilidad, pd.Series(modelo.feature_importances_, index=x_ent.columns)
    return ajustar


def crear_ajuste_catboost(parametros, categoricas):
    """Función ajustar() de CatBoost con las categóricas nativas (NA como categoría)."""
    def a_texto(x, columnas):
        # Copia superficial: solo las columnas categóricas se escriben de nuevo
        x = x.copy(deep=False)
        for c in columnas:
            x[c] = x[c].astype("string").fillna("NA").astype(str)
        return x

    def ajustar(x_ent, y_ent, x_eval):
        columnas = [c for c in categoricas if c in x_ent.columns]
        modelo = CatBoostClassifier(**parametros, random_seed=SEMILLA, verbose=0, allow_writing_files=False)
        modelo.fit(a_texto(x_ent, columnas), y_ent, cat_features=columnas)
        probabilidad = modelo.predict_proba(a_texto(x_eval, columnas))[:, 1]
        importancia = pd.Series(modelo.get_feature_importance(), index=x_ent.columns)
        del modelo
        gc.collect()
        return probabilidad, importancia
    return ajustar


def mezclar(nombre, nombre_lgb, nombre_cat, verdad):
    """Promedio ponderado de dos candidatos con el peso que maximiza el F1 OOF.

    El peso y el umbral se eligen solo con OOF; holdout y test reciben la misma
    mezcla. Escribe las predicciones como un candidato más.
    Devuelve: (resultado, peso de CatBoost).
    """
    a = resultado_desde_predicciones(nombre_lgb, verdad)
    b = resultado_desde_predicciones(nombre_cat, verdad)
    oof_b = b["oof"]["probabilidad"].reindex(a["oof"].index)
    f1_por_peso = {w: buscar_umbral_f1(a["oof"]["verdad"], (1 - w) * a["oof"]["probabilidad"] + w * oof_b)[1]
                   for w in PESOS_MEZCLA}
    peso = max(f1_por_peso, key=f1_por_peso.get)

    def combinar(serie_a, serie_b):
        return ((1 - peso) * serie_a + peso * serie_b.reindex(serie_a.index)).rename("probabilidad")

    resultado = {"nombre": nombre,
                 "oof": pd.DataFrame({"probabilidad": combinar(a["oof"]["probabilidad"], b["oof"]["probabilidad"]),
                                      "fold": a["oof"]["fold"]}),
                 "prediccion_holdout": combinar(a["prediccion_holdout"], b["prediccion_holdout"])}
    if "prediccion_test" in a and "prediccion_test" in b:
        resultado["prediccion_test"] = combinar(a["prediccion_test"], b["prediccion_test"])
    guardar_predicciones(resultado)
    pd.DataFrame({"peso_catboost": list(f1_por_peso), "f1_oof": list(f1_por_peso.values())}) \
        .to_csv(RUTA_TABLAS / "etapa7_b_mezcla_pesos.csv", index=False)
    registrar_log(nombre, f"peso CatBoost {peso:.2f} (F1 OOF {f1_por_peso[peso]:.4f})")
    return resultado_desde_predicciones(nombre, verdad), float(peso)


def decodificar_abc(z, columnas):
    """Vector de la colonia -> (columnas elegidas, hiperparámetros de ExtraTrees)."""
    p = len(columnas)
    mascara = np.asarray(z[:p]) > 0.5
    h = np.asarray(z[p:])
    (n0, n1), (f0, f1), (s0, s1), (m0, m1) = RANGOS_ABC.values()
    parametros = {
        "n_estimators": int(round(n0 + h[0] * (n1 - n0))),
        "max_features": float(f0 + h[1] * (f1 - f0)),
        "max_samples": float(s0 + h[2] * (s1 - s0)),
        "min_samples_leaf": int(round(np.exp(np.log(m0) + h[3] * (np.log(m1) - np.log(m0))))),
        "bootstrap": True,
    }
    return [c for c, m in zip(columnas, mascara) if m], parametros


def codificar_defecto(columnas):
    """Solución inicial: todas las columnas con los hiperparámetros de B1."""
    p = len(columnas)
    (n0, n1), (f0, f1), (s0, s1), (m0, m1) = RANGOS_ABC.values()
    fraccion_sqrt = np.sqrt(p) / p
    h = [(300 - n0) / (n1 - n0), (fraccion_sqrt - f0) / (f1 - f0), (1.0 - s0) / (s1 - s0),
         (np.log(10) - np.log(m0)) / (np.log(m1) - np.log(m0))]
    return np.concatenate([np.ones(p), np.clip(h, 0, 1)])


def buscar_con_abejas(preparar, reloj, reserva_min, buscar=True):
    """Colonia de abejas sobre el 80% (matriz de entrenamiento del contexto
    holdout, con los auxiliares en su OOF interna): aptitud = F1 OOF con 3 folds
    sobre una submuestra estratificada (ABC_SUBMUESTRA) del 80%, para acotar
    memoria y tiempo. A lo sumo ABC_MAX_CICLOS ciclos.

    Corre ciclos mientras quepan antes de la reserva para la evaluación final;
    con buscar=False (la evaluación final ya existe) solo lee la mejor solución.
    Devuelve: (columnas, parámetros, aptitud, evaluaciones).
    """
    x80, y80, _ = preparar("holdout")
    columnas = list(x80.columns)
    filas, _ = train_test_split(np.arange(len(y80)), train_size=ABC_SUBMUESTRA, stratify=y80,
                                random_state=SEMILLA)
    x80, y80 = x80.to_numpy(np.float32)[filas], y80[filas]
    gc.collect()
    pliegues = list(StratifiedKFold(ABC_PLIEGUES, shuffle=True, random_state=SEMILLA).split(x80, y80))

    def aptitud(z):
        _, parametros = decodificar_abc(z, columnas)
        indices = np.flatnonzero(np.asarray(z[:len(columnas)]) > 0.5)
        probabilidad = np.zeros(len(y80))
        for entrenamiento, validacion in pliegues:
            modelo = ExtraTreesClassifier(**parametros, n_jobs=N_HILOS, random_state=SEMILLA)
            modelo.fit(x80[np.ix_(entrenamiento, indices)], y80[entrenamiento])
            probabilidad[validacion] = modelo.predict_proba(x80[np.ix_(validacion, indices)])[:, 1]
            del modelo
        umbral, f1 = buscar_umbral_f1(y80, probabilidad)
        gc.collect()
        return f1, {"umbral": umbral, **parametros}

    colonia = ColoniaAbejas(aptitud, len(columnas), len(RANGOS_ABC), ABC_FUENTES, ABC_LIMITE,
                            RUTA_ETAPA7 / "abc", SEMILLA, iniciales=[codificar_defecto(columnas)],
                            bits_por_movimiento=max(3, len(columnas) // 100),
                            registrar=lambda texto: registrar_log("abc", texto))
    inicio_ciclos, ciclos = time.time(), 0
    while buscar:
        duracion = (time.time() - inicio_ciclos) / 60 / ciclos if ciclos else 0
        if colonia.ciclo >= ABC_MAX_CICLOS or reloj.transcurrido() + max(duracion, 1) + reserva_min > MINUTOS_B:
            break
        colonia.correr_ciclo()
        ciclos += 1
    registrar_log("abc", f"fin de la búsqueda: ciclo {colonia.ciclo}, {len(colonia.cache)} evaluaciones")
    historia = pd.DataFrame([{k: v for k, v in f.items() if k not in ("z", "huella")}
                             for f in colonia.cache.values()])
    historia.to_csv(RUTA_TABLAS / "etapa7_abc_historia.csv", index=False)
    z, mejor_aptitud = colonia.mejor()
    elegidas, parametros = decodificar_abc(z, columnas)
    return elegidas, parametros, mejor_aptitud, len(colonia.cache)


def experimento_b(entorno, a):
    """Corre B1, B3, B4 y B2 (en ese orden: la colonia usa el tiempo que sobre)."""
    reloj = Presupuesto(MINUTOS_B)
    registrar_log("etapa7_b", "inicio")
    particion, verdad, base = entorno["particion"], entorno["verdad"], entorno["base"]
    usar_a = bool(a and a["mejora"])
    sufijo = "_agr" if usar_a else ""
    if usar_a:
        preparar_b, nombre_lgb, variables_b = a["preparar"], a["nombre"], a["variables"]
    else:
        preparar_b, nombre_lgb = crear_preparador(entorno["matriz"], particion, entorno["dinamicas"]), BASE
        variables_b = entorno["variables_base"]
    registrar_log("etapa7_b", f"base de B: {'n7_v3 + agregados de A' if usar_a else 'n7_v3'}")
    lgb_resultado = resultado_desde_predicciones(nombre_lgb, verdad)
    lgb_resultado["tiempo_min"] = leer_tiempos().get(nombre_lgb, entorno["base"].get("tiempo_min"))
    filas = [fila_tabla(lgb_resultado, verdad, f"LightGBM de referencia ({nombre_lgb})", base)]
    resultados = {}

    def anotar(resultado, descripcion, especificacion):
        resultados[resultado["nombre"]] = resultado
        filas.append(fila_tabla(resultado, verdad, descripcion, base))
        entorno["descripciones"][resultado["nombre"]] = especificacion
        pd.DataFrame(filas).to_csv(RUTA_TABLAS / "etapa7_b.csv", index=False)

    # B1: ExtraTrees por defecto
    nombre = f"n7_b1_extratrees{sufijo}"
    r = correr_seguro("etapa7_b", nombre, lambda: evaluar_o_cargar(nombre, lambda: evaluar_modelo(
        nombre, preparar_b, crear_ajuste_extratrees(PARAMETROS_ET_DEFECTO), particion, con_test=True,
        reanudable=True), verdad))
    if r:
        anotar(r, "ExtraTrees por defecto (300 árboles, sqrt, hoja 10, bootstrap)", {
            "algoritmo": "extratrees", "variables": variables_b, "busqueda": "valores por defecto razonables",
            "n_configuraciones": 1, "hiperparametros": texto_hiperparametros(PARAMETROS_ET_DEFECTO),
            "especificacion": "ExtraTrees (bagging de árboles extremadamente aleatorizados, bootstrap) | "
                              "base n7_v3" + (" + agregados de A" if usar_a else "")})

    # B3: CatBoost con categóricas nativas
    nombre_cat = f"n7_b3_catboost{sufijo}"

    def correr_catboost():
        categoricas = [c for b in BLOQUES_A4 for c in entorno["bloques"][b].columns
                       if isinstance(entorno["bloques"][b][c].dtype, pd.CategoricalDtype)]
        matriz_nativa = armar_matriz(entorno["insumos"], BLOQUES_A4, nativas=categoricas)
        preparar_cat = crear_preparador(matriz_nativa, particion, entorno["dinamicas"])
        if usar_a:
            matriz_nativa = matriz_nativa.join(cargar_o_construir_agregados(matriz_nativa.index))
            preparar_cat = crear_preparador(matriz_nativa, particion, entorno["dinamicas"])
            if a["variante"] == "filtro":
                preparar_cat = crear_filtro(preparar_cat, K_FILTRO)
        return evaluar_o_cargar(nombre_cat, lambda: evaluar_modelo(
            nombre_cat, preparar_cat, crear_ajuste_catboost(PARAMETROS_CATBOOST, categoricas), particion,
            con_test=True, reanudable=True), verdad)

    r = correr_seguro("etapa7_b", nombre_cat, correr_catboost)
    if r:
        anotar(r, "CatBoost, categóricas nativas (1500 iter., lr 0,05, prof. 6)", {
            "algoritmo": "catboost", "variables": variables_b, "busqueda": "valores por defecto razonables",
            "n_configuraciones": 1, "hiperparametros": texto_hiperparametros(PARAMETROS_CATBOOST),
            "especificacion": "CatBoost con categóricas nativas | base n7_v3" + (" + agregados de A" if usar_a else "")})

    # B4: mezcla LightGBM + CatBoost si CatBoost queda cerca en CV
    if nombre_cat in resultados:
        distancia = lgb_resultado["f1_cv"] - resultados[nombre_cat]["f1_cv"]
        if distancia < DISTANCIA_MEZCLA:
            nombre = f"n7_b4_mezcla_lgb_cat{sufijo}"
            salida = correr_seguro("etapa7_b", nombre, lambda: mezclar(nombre, nombre_lgb, nombre_cat, verdad))
            if salida:
                r, peso = salida
                anotar(r, f"Promedio LightGBM + CatBoost (peso CatBoost {peso:.2f}, elegido en OOF)", {
                    "algoritmo": "mezcla", "variables": variables_b,
                    "busqueda": f"peso de CatBoost en grilla 0-1 paso 0,05 por F1 OOF; elegido {peso:.2f}",
                    "n_configuraciones": len(PESOS_MEZCLA), "hiperparametros": f"peso_catboost = {peso:.2f}",
                    "especificacion": f"Promedio de {nombre_lgb} y {nombre_cat}"})
        else:
            registrar_log("etapa7_b", f"CatBoost a {distancia:.4f} de LightGBM en CV: sin mezcla")

    # B2: ExtraTrees con colonia de abejas, con el tiempo que quede
    nombre = f"n7_b2_extratrees_abc{sufijo}"

    def correr_abc():
        tiempo_b1 = leer_tiempos().get(f"n7_b1_extratrees{sufijo}", 10)
        reserva = 1.5 * tiempo_b1 + 3
        terminado = (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists()
        elegidas, parametros, aptitud, evaluaciones = buscar_con_abejas(preparar_b, reloj, reserva,
                                                                         buscar=not terminado)
        registrar_log("abc", f"mejor: aptitud {aptitud:.4f}, {len(elegidas)} columnas, {parametros}")
        entorno["descripciones"][nombre] = {
            "algoritmo": "extratrees", "variables": elegidas,
            "busqueda": (f"colonia artificial de abejas ({ABC_FUENTES} fuentes, límite {ABC_LIMITE}): máscara de "
                         f"variables + n_estimators {RANGOS_ABC['n_estimators']}, max_features "
                         f"{RANGOS_ABC['max_features']}, max_samples {RANGOS_ABC['max_samples']}, "
                         f"min_samples_leaf {RANGOS_ABC['min_samples_leaf']} log; aptitud F1 OOF 3 folds "
                         f"sobre una submuestra estratificada del {ABC_SUBMUESTRA:.0%} del 80%"),
            "n_configuraciones": evaluaciones, "hiperparametros": texto_hiperparametros(parametros),
            "especificacion": f"ExtraTrees con variables e hiperparámetros por colonia de abejas "
                              f"({len(elegidas)} columnas)"}
        resultado = evaluar_o_cargar(nombre, lambda: evaluar_modelo(
            nombre, con_columnas(preparar_b, elegidas), crear_ajuste_extratrees(parametros), particion,
            con_test=True, reanudable=True), verdad)
        return resultado, len(elegidas), parametros

    salida = correr_seguro("etapa7_b", nombre, correr_abc)
    if salida:
        r, n_columnas, parametros = salida
        resultados[nombre] = r
        filas.append(fila_tabla(r, verdad, f"ExtraTrees + colonia de abejas ({n_columnas} columnas, "
                                           f"{parametros['n_estimators']} árboles, hoja "
                                           f"{parametros['min_samples_leaf']})", base))
        pd.DataFrame(filas).to_csv(RUTA_TABLAS / "etapa7_b.csv", index=False)
    registrar_log("etapa7_b", f"fin ({reloj.transcurrido():.1f} min)")
    return resultados


# ---- Cierre: CSV, leaderboard y bitácora --------------------------------------

def cerrar(entorno, candidatos):
    """CSV (solo los que superan a n7_v3 en CV y holdout), leaderboard y bitácora."""
    particion, verdad, base = entorno["particion"], entorno["verdad"], entorno["base"]
    ids_test = particion.loc[particion["base"] == "test", "id"]
    tiempos = leer_tiempos()
    filas = []
    for nombre in candidatos:
        descripcion = entorno["descripciones"].get(nombre)
        if descripcion is None or not (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
            continue
        resultado = resultado_desde_predicciones(nombre, verdad)
        supera = resultado["f1_cv"] > base["f1_cv"] and resultado["f1_holdout"] > base["f1_holdout"]
        archivo = None
        if supera and "prediccion_test" in resultado:
            ruta, tasa = guardar_submission(
                resultado, f"{nombre}.csv", ids_test, descripcion["especificacion"] + " | umbral OOF",
                algoritmo=ALGORITMO_CURSO[descripcion["algoritmo"]])
            archivo = f"nivel7/{ruta.name}"
            registrar_log("etapa7", f"submission {ruta.name} tasa {tasa:.4f}")
        filas.append(armar_fila(nombre, ALGORITMO_CURSO[descripcion["algoritmo"]], descripcion["especificacion"],
                                descripcion["variables"], descripcion["busqueda"], descripcion["n_configuraciones"],
                                descripcion["hiperparametros"], tiempos.get(nombre, pd.NA), archivo, verdad))
    if filas:
        formatear_como_r(pd.DataFrame(filas)).to_csv(RUTA_LEADERBOARD_EXTRA, index=False, na_rep="NA")
        anexar_filas_extra()
        registrar_log("etapa7", f"{len(filas)} filas en el leaderboard")


def main():
    RUTA_ETAPA7.mkdir(parents=True, exist_ok=True)
    registrar_log("etapa7", "inicio")
    iniciar_vigilante_memoria("etapa7")
    insumos = cargar_insumos(auxiliar_total=True, auxiliar_fuentes=True)
    particion = insumos["particion"]
    verdad = particion.set_index("id")["pobre"]
    bloques = insumos["bloques"]
    base = resultado_desde_predicciones(BASE, verdad)
    base["tiempo_min"] = pd.read_csv(RUTA_TABLAS / "etapa6_ablacion.csv").set_index("candidato") \
        .loc[BASE, "tiempo_min"]
    entorno = {
        "insumos": insumos, "particion": particion, "verdad": verdad, "bloques": bloques, "base": base,
        "matriz": armar_matriz(insumos),
        "dinamicas": [insumos["auxiliares"][a] for a in AUXILIARES_BASE],
        "variables_base": [c for b in BLOQUES_A4 for c in bloques[b].columns]
                          + [f"aux_ingreso_{a}" for a in AUXILIARES_BASE],
        "descripciones": {},
    }

    a = correr_seguro("etapa7_a", "experimento A", lambda: experimento_a(entorno))
    resultados_b = correr_seguro("etapa7_b", "experimento B", lambda: experimento_b(entorno, a)) or {}
    candidatos = ([a["nombre"]] if a else []) + list(resultados_b)
    correr_seguro("etapa7", "cierre", lambda: cerrar(entorno, candidatos))
    for nombre in candidatos:
        borrar_parciales(nombre)
    registrar_log("etapa7", "fin")


if __name__ == "__main__":
    main()
