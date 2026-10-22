"""Etapa 10: redes neuronales tabulares (RealMLP y TabM) en ensamble con LightGBM.

Base de comparación: n7_a_agregados_bag5 (F1 CV 0,7748, holdout 0,7806). Mismos
folds, holdout, umbral OOF, OOF anidada de los auxiliares de ingreso y
evaluar_modelo() de siempre.

  1. Redes sobre la base completa de A (1.324 columnas: dummies de las
     categóricas, numéricas y auxiliares; se quitan solo las constantes del
     entrenamiento de cada contexto), con las implementaciones oficiales:
       - RealMLP (pytabkit): su preprocesamiento de robust scaling + smooth clip,
         sin embeddings numéricos PBLD (medido en cv_1: 25 s por época con ellos
         frente a 3,5 s sin ellos; no caben en 3 horas de CPU).
       - TabM (tabm): cuantiles a normal en las continuas; k = 8, d_block = 256,
         2 bloques (la configuración del paper, k = 32 y d_block = 512, tarda
         170 s por época en cv_1).
     Early stopping con un 10% interno estratificado del entrenamiento de cada
     contexto (nunca el fold de evaluación). Tuning breve por grilla con una
     validación interna del 25% del entrenamiento del holdout (folds 1-5);
     holdout y test no se tocan. Bagging de 3 a 5 semillas: cada semilla es un
     candidato completo (n10_<red>_s<k>) y el bag promedia las que terminaron.
  2. Métrica: F1 de la clase pobre con el umbral OOF de buscar_umbral_f1().
  3. Ensambles con pesos elegidos solo en OOF: LightGBM (A) + red y, aparte,
     LightGBM + red + CatBoost B3 (n7_b3_catboost_agr) en un simplex de paso 0,05.
  4. Tabla etapa10_candidatos.csv: F1 CV (sd), holdout, precisión, recall,
     banda [0,8; 1,2] de Lp, tiempo y correlación OOF con LightGBM.
  5. CSV de Kaggle solo si un candidato sube al menos 0,002 en CV y en holdout
     sobre la base (máximo uno).

Presupuesto: 180 minutos contados desde el primer arranque (el reloj se guarda
en disco y un relanzamiento sigue contando). Antes de cada tuning o semilla se
estima su costo y solo se corre si cabe dejando la reserva del cierre; un hilo
de corte duro termina el proceso si el reloj se agota igual. Reanudable (cada
configuración del tuning y cada fold de cada semilla quedan en disco) y con el
vigilante de memoria. Un solo proceso pesado.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa10_redes.py
"""

import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import funciones.bitacora as bitacora
from configuracion import RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa4_leaderboard import agregar_filas_extra, armar_fila, texto_hiperparametros
from etapa7_exploracion import AUXILIARES_BASE, PESOS_MEZCLA, RUTA_ETAPA7, correr_seguro, fila_tabla
from etapa8_cierre import grilla_simplex
from etapa9_evolutivo import escribir_json, leer_json
from funciones.anidado import definir_contextos, guardar_parquet_atomico
from funciones.bases import armar_matriz, cargar_insumos
from funciones.datos import cargar_particion
from funciones.bitacora import Presupuesto, iniciar_corte_duro, iniciar_vigilante_memoria, registrar_log
from funciones.mezclas import mezclar_por_oof, promediar_semillas
from funciones.metricas import buscar_umbral_f1
from funciones.modelado import (borrar_parciales, crear_preparador, en_grupos, evaluar_modelo,
                                resultado_desde_predicciones)
from funciones.redes import (N_NUCLEOS_FISICOS, columnas_variables, correlaciones_con, crear_ajuste_red,
                             describir_columnas, particion_parada)
from funciones.submission import guardar_submission

BASE = "n7_a_agregados_bag5"
CATBOOST = "n7_b3_catboost_agr"
MINUTOS_TOTALES = 180
RESERVA_CIERRE_MIN = 8
RUTA_ETAPA10 = RUTA_NIVEL7 / "etapa10"
MEJORA_MINIMA = 0.002
SEMILLAS_MINIMO, SEMILLAS_MAXIMO = 3, 5
FRACCION_TUNEO = 0.25
PASO_SIMPLEX = 0.05
# Una red solo empieza su tuning si quedan al menos estos minutos (más la reserva)
MINUTOS_MINIMOS_RED = 30
# Las matrices del contexto test (165 mil hogares x 1.324 columnas) más las copias
# internas de las redes pasan de 6 GB; el piso de 2 GB libres del sistema sigue igual
LIMITE_PROCESO_GB = 7.5

REDES = {
    "realmlp": {
        "etiqueta": "RealMLP",
        "fijos": {"n_epochs": 32, "batch_size": 1024, "hidden_sizes": [256, 256, 256], "num_emb_type": "none",
                  "wd": 1e-2},
        "grilla": [{"lr": lr, "p_drop": p} for lr in (0.02, 0.04) for p in (0.15, 0.30)],
        "rango": "grilla lr {0.02, 0.04} x p_drop {0.15, 0.30}",
        "especificacion": ("RealMLP-TD (pytabkit) 3 x 256, SELU paramétrica, robust scaling + smooth clip, "
                           "sin embeddings PBLD, 32 épocas con selección del mejor epoch y early stopping por "
                           "log loss en un 10% interno"),
    },
    "tabm": {
        "etiqueta": "TabM",
        "fijos": {"k": 8, "d_block": 256, "n_blocks": 2, "arch_type": "tabm", "dropout": 0.1,
                  "weight_decay": 3e-4, "batch_size": 1024, "max_epocas": 10, "paciencia": 3},
        "grilla": [{"lr": 1e-3}, {"lr": 2e-3}],
        "rango": "grilla lr {0.001, 0.002}",
        "especificacion": ("TabM (tabm, BatchEnsemble k = 8) 2 x 256, ReLU, cuantiles a normal en las continuas, "
                           "AdamW, hasta 10 épocas con early stopping por log loss en un 10% interno"),
    },
}


# ---- Utilidades ---------------------------------------------------------------

def nombre_semilla(red, k):
    return f"n10_{red}_s{k}"


def anotar_tiempo(nombre, minutos):
    """Minutos de un candidato en etapa10/tiempos.json (para reportar al retomar)."""
    ruta = RUTA_ETAPA10 / "tiempos.json"
    tiempos = leer_json(ruta) or {}
    tiempos[nombre] = minutos
    escribir_json(tiempos, ruta)


def leer_tiempos():
    return leer_json(RUTA_ETAPA10 / "tiempos.json") or {}


def evaluar_o_cargar(nombre, evaluar, verdad):
    """Resultado desde las predicciones si ya terminó; si no, evaluar() y anota el tiempo."""
    if (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
        resultado = resultado_desde_predicciones(nombre, verdad)
        resultado["tiempo_min"] = leer_tiempos().get(nombre, np.nan)
        return resultado
    resultado = evaluar()
    anotar_tiempo(nombre, resultado["tiempo_min"])
    return resultado


def crear_preparador_disco(preparar_completo):
    """preparar() que lee las matrices de cada contexto desde etapa10/matrices/.

    La primera vez arma el contexto con preparar_completo() y lo guarda; así el
    proceso no carga la matriz completa de A (con los auxiliares) mientras las
    redes entrenan, que es lo que faltaba de memoria en el contexto test.
    Argumentos: preparar_completo = crear_preparador() sobre la base de A, o
        None si todas las matrices ya están en disco.
    Devuelve: función(contexto) -> (x_ent, y_ent, x_eval).
    """
    def preparar(contexto):
        carpeta = RUTA_ETAPA10 / "matrices"
        rutas = {p: carpeta / f"{contexto}_{p}.parquet" for p in ("ent", "eval")}
        if all(r.exists() for r in rutas.values()):
            x_ent = pd.read_parquet(rutas["ent"])
            y_ent = x_ent.pop("pobre").to_numpy()
            return x_ent, y_ent, pd.read_parquet(rutas["eval"])
        x_ent, y_ent, x_eval = preparar_completo(contexto)
        carpeta.mkdir(parents=True, exist_ok=True)
        guardar_parquet_atomico(x_ent.assign(pobre=y_ent), rutas["ent"])
        guardar_parquet_atomico(x_eval, rutas["eval"])
        return x_ent, y_ent, x_eval

    return preparar


def filas_por_contexto(particion):
    """Hogares de entrenamiento de cada contexto (el costo de una red escala con ellos)."""
    tabla = particion.set_index("id")
    return {c: int(en_grupos(tabla, ent).sum()) for c, (ent, _) in definir_contextos().items()}


# ---- 1. Tuning ----------------------------------------------------------------

def particion_tuneo(y_ent):
    """Ajuste y validación del tuning dentro del entrenamiento del holdout (folds 1-5).

    Devuelve: (posiciones de ajuste, posiciones de validación), disjuntas.
    """
    return particion_parada(y_ent, SEMILLA + 10, FRACCION_TUNEO)


def tunear(red, preparar, reloj, costo_semilla):
    """Grilla de la red evaluada por F1 (umbral óptimo) en la validación del tuning.

    Cada configuración se guarda al terminar en etapa10/tuneo/<red>.json. Una
    configuración solo se corre si, después de ella, todavía caben las semillas
    mínimas y la reserva del cierre.
    Argumentos: red = clave de REDES; preparar = función de contextos;
        reloj = Presupuesto; costo_semilla = función(segundos de un ajuste de
        tuning) -> minutos estimados de una semilla completa.
    Devuelve: (parámetros elegidos o None si no se evaluó nada, tabla del tuning).
    """
    especificacion = REDES[red]
    ruta = RUTA_ETAPA10 / "tuneo" / f"{red}.json"
    filas = leer_json(ruta) or []
    hechas = [f["parametros"] for f in filas]
    pendientes = [p for p in especificacion["grilla"] if p not in hechas]
    if pendientes:
        if not filas and reloj.restante() - RESERVA_CIERRE_MIN < MINUTOS_MINIMOS_RED:
            registrar_log(f"tuneo_{red}", f"quedan {reloj.restante():.0f} min: la red no empieza")
            return None, pd.DataFrame()
        x_ent, y_ent, _ = preparar("holdout")
        escribir_json(columnas_variables(x_ent), RUTA_ETAPA10 / "columnas_holdout.json")
        escribir_json(describir_columnas(x_ent), RUTA_ETAPA10 / "columnas_tipos.json")
        ajuste, validacion = particion_tuneo(y_ent)
        x_ajuste, x_validacion = x_ent.iloc[ajuste], x_ent.iloc[validacion]
        del x_ent
        gc.collect()
        for parametros in pendientes:
            previo = np.mean([f["segundos"] for f in filas]) if filas else None
            if previo is not None:
                necesario = previo / 60 + SEMILLAS_MINIMO * costo_semilla(previo)
                if reloj.restante() - RESERVA_CIERRE_MIN < necesario:
                    registrar_log(f"tuneo_{red}", f"sin tiempo para {parametros}: se queda con lo evaluado")
                    break
            inicio = time.time()
            ajustar = crear_ajuste_red(red, {**especificacion["fijos"], **parametros}, SEMILLA)
            probabilidad, _ = ajustar(x_ajuste, y_ent[ajuste], x_validacion)
            umbral, f1 = buscar_umbral_f1(y_ent[validacion], probabilidad)
            filas.append({"parametros": parametros, "f1_validacion": f1, "umbral": umbral,
                          "segundos": time.time() - inicio, "filas_ajuste": len(ajuste)})
            escribir_json(filas, ruta)
            registrar_log(f"tuneo_{red}", f"{parametros}: F1 {f1:.4f} ({filas[-1]['segundos']:.0f} s)")
            gc.collect()
    tabla = pd.DataFrame([{"red": red, **f["parametros"], **{k: v for k, v in f.items() if k != "parametros"}}
                          for f in filas])
    if not filas:
        return None, tabla
    mejor = max(filas, key=lambda f: f["f1_validacion"])
    return {**especificacion["fijos"], **mejor["parametros"]}, tabla


# ---- 1. Semillas y bagging ----------------------------------------------------

def semillas_terminadas(red):
    """Semillas de la red con predicciones completas (OOF, holdout y test)."""
    return [k for k in range(SEMILLAS_MAXIMO)
            if (RUTA_NIVEL7 / f"predicciones_{nombre_semilla(red, k)}.parquet").exists()]


def correr_semillas(red, objetivo, parametros, entorno, estimar):
    """Corre semillas de la red hasta llegar a `objetivo` terminadas, mientras quepan.

    Argumentos: red = clave de REDES; objetivo = número de semillas a tener;
        parametros = hiperparámetros elegidos; entorno = preparar, particion,
        verdad y reloj; estimar = función(red) -> minutos de una semilla.
    """
    reloj = entorno["reloj"]
    for k in range(SEMILLAS_MAXIMO):
        if len(semillas_terminadas(red)) >= objetivo:
            return
        if k in semillas_terminadas(red):
            continue
        estimado = estimar(red)
        if reloj.restante() - RESERVA_CIERRE_MIN < estimado:
            registrar_log(f"etapa10_{red}", f"semilla {k}: no cabe (~{estimado:.0f} min, quedan "
                                            f"{reloj.restante():.0f})")
            return
        nombre = nombre_semilla(red, k)
        registrar_log(nombre, f"inicio (~{estimado:.0f} min estimados)")
        correr_seguro("etapa10", nombre, lambda: evaluar_o_cargar(nombre, lambda: evaluar_modelo(
            nombre, entorno["preparar"], crear_ajuste_red(red, parametros, SEMILLA + k), entorno["particion"],
            con_test=True, reanudable=True), entorno["verdad"]))
        gc.collect()


def crear_estimador(tablas_tuneo, filas_contexto):
    """Función red -> minutos estimados de una semilla completa (7 contextos).

    Escala el tiempo medio de un ajuste del tuning por las filas de entrenamiento
    de los 7 contextos; con semillas terminadas toma el mayor entre eso y su
    tiempo medio (una semilla retomada tras un relanzamiento anota menos tiempo
    del que costó, así que su tiempo solo no basta).
    """
    total_filas = sum(filas_contexto.values())

    def desde_tuneo(segundos, filas_ajuste):
        return segundos * total_filas / filas_ajuste / 60

    def estimar(red):
        tiempos = leer_tiempos()
        reales = [tiempos[nombre_semilla(red, k)] for k in semillas_terminadas(red)
                  if np.isfinite(tiempos.get(nombre_semilla(red, k), np.nan))]
        tabla = tablas_tuneo.get(red)
        if tabla is None or tabla.empty:
            return float(np.mean(reales)) if reales else np.inf
        estimado = desde_tuneo(tabla["segundos"].mean(), tabla["filas_ajuste"].iloc[0])
        return max(estimado, float(np.mean(reales))) if reales else estimado

    return estimar, desde_tuneo


# ---- 3. Ensambles y correlación -----------------------------------------------

def ensamblar(bags, verdad):
    """LightGBM + cada red (peso por F1 OOF) y LightGBM + mejor red + CatBoost B3.

    Devuelve: diccionario {nombre: (resultado, pesos, componentes)}.
    """
    salida = {}
    grilla_dos = [np.array([1 - w, w]) for w in PESOS_MEZCLA]
    for red, nombre_bag in bags.items():
        nombre = f"n10_mezcla_lgb_{red}"
        resultado, pesos, tabla = mezclar_por_oof(nombre, [BASE, nombre_bag], grilla_dos, verdad)
        tabla.to_csv(RUTA_TABLAS / f"etapa10_pesos_lgb_{red}.csv", index=False)
        salida[nombre] = (resultado, pesos, [BASE, nombre_bag])
        registrar_log(nombre, f"pesos {np.round(pesos, 2)} (F1 OOF {tabla['f1_oof'].max():.4f})")
    if bags and (RUTA_NIVEL7 / f"predicciones_{CATBOOST}.parquet").exists():
        mejor_red = max(bags, key=lambda r: resultado_desde_predicciones(bags[r], verdad)["f1_cv"])
        nombre = f"n10_mezcla_lgb_{mejor_red}_cat"
        componentes = [BASE, bags[mejor_red], CATBOOST]
        resultado, pesos, tabla = mezclar_por_oof(nombre, componentes, grilla_simplex(3, PASO_SIMPLEX), verdad)
        tabla.to_csv(RUTA_TABLAS / f"etapa10_pesos_lgb_{mejor_red}_cat.csv", index=False)
        salida[nombre] = (resultado, pesos, componentes)
        registrar_log(nombre, f"pesos {np.round(pesos, 2)} (F1 OOF {tabla['f1_oof'].max():.4f})")
    return salida


# ---- 4-5. Tabla, CSV y leaderboard --------------------------------------------

def cerrar(entorno, resultados, descripciones, correlacion):
    """Tabla de candidatos, CSV del mejor que cumple la regla y leaderboard."""
    verdad, base = entorno["verdad"], resultados[BASE]
    filas = []
    for nombre, resultado in resultados.items():
        fila = fila_tabla(resultado, verdad, descripciones[nombre]["descripcion"], base)
        fila["cumple_regla_csv"] = bool(nombre.startswith("n10_") and fila["delta_cv"] >= MEJORA_MINIMA
                                        and fila["delta_holdout"] >= MEJORA_MINIMA)
        filas.append(fila)
    tabla = pd.DataFrame(filas).drop(columns=["supera_base"]).merge(correlacion, on="candidato", how="left")
    tabla.to_csv(RUTA_TABLAS / "etapa10_candidatos.csv", index=False)

    cumplen = tabla[tabla["cumple_regla_csv"]].sort_values("f1_holdout", ascending=False)
    elegido = cumplen["candidato"].iloc[0] if len(cumplen) else None
    registrar_log("etapa10", f"candidatos que cumplen la regla: {list(cumplen['candidato'])} -> CSV: {elegido}")
    ids_test = entorno["particion"].loc[entorno["particion"]["base"] == "test", "id"]

    filas_lb = []
    for nombre, descripcion in descripciones.items():
        if not nombre.startswith("n10_"):
            continue
        archivo = None
        if nombre == elegido:
            ruta, tasa = guardar_submission(resultados[nombre], f"{nombre}.csv", ids_test,
                                            descripcion["especificacion"] + " | umbral OOF",
                                            algoritmo=descripcion["algoritmo"])
            archivo = f"nivel7/{ruta.name}"
            registrar_log("etapa10", f"submission {ruta.name} tasa {tasa:.4f}")
        filas_lb.append(armar_fila(nombre, descripcion["algoritmo"], descripcion["especificacion"],
                                   descripcion["variables"], descripcion["busqueda"],
                                   descripcion["n_configuraciones"], descripcion["hiperparametros"],
                                   resultados[nombre].get("tiempo_min", pd.NA), archivo, verdad))
    # Se agregan a las filas de las etapas anteriores (no las reemplazan)
    agregar_filas_extra(filas_lb)
    return tabla, elegido


def main():
    RUTA_ETAPA10.mkdir(parents=True, exist_ok=True)
    reloj = Presupuesto(MINUTOS_TOTALES, RUTA_ETAPA10 / "inicio.txt")
    registrar_log("etapa10", f"inicio ({reloj.transcurrido():.1f} min ya usados, {N_NUCLEOS_FISICOS} núcleos)")
    bitacora.LIMITE_PROCESO_GB = LIMITE_PROCESO_GB
    iniciar_vigilante_memoria("etapa10")
    en_cierre = threading.Event()
    iniciar_corte_duro("etapa10", reloj, en_cierre)

    # Matrices de los 7 contextos a disco y fuera de memoria la base completa
    particion = cargar_particion()
    verdad = particion.set_index("id")["pobre"]
    if not all((RUTA_ETAPA10 / "matrices" / f"{c}_eval.parquet").exists() for c in definir_contextos()):
        insumos = cargar_insumos(auxiliar_total=True, auxiliar_fuentes=True)
        matriz_a = armar_matriz(insumos).join(pd.read_parquet(RUTA_ETAPA7 / "agregados_personas.parquet"))
        preparar_completo = crear_preparador(matriz_a, particion,
                                             [insumos["auxiliares"][a] for a in AUXILIARES_BASE])
        armar = crear_preparador_disco(preparar_completo)
        for contexto in definir_contextos():
            armar(contexto)
            gc.collect()
        del insumos, matriz_a, preparar_completo, armar
        gc.collect()
        registrar_log("etapa10", f"matrices en disco ({reloj.transcurrido():.1f} min)")
    preparar = crear_preparador_disco(None)
    entorno = {"particion": particion, "verdad": verdad, "preparar": preparar, "reloj": reloj}
    filas_contexto = filas_por_contexto(particion)
    tablas_tuneo, parametros = {}, {}
    estimar, desde_tuneo = crear_estimador(tablas_tuneo, filas_contexto)
    filas_ajuste_tuneo = int(round((1 - FRACCION_TUNEO) * filas_contexto["holdout"]))

    def costo_semilla(segundos):
        return desde_tuneo(segundos, filas_ajuste_tuneo)

    # 1. Primero RealMLP completo (tuning + semillas mínimas), luego TabM y al final
    #    semillas extra de cada red mientras quepan
    for red in REDES:
        salida = correr_seguro("etapa10", f"tuneo_{red}", lambda: tunear(red, preparar, reloj, costo_semilla))
        if salida is None:
            continue
        parametros[red], tablas_tuneo[red] = salida
        if parametros[red] is None:
            continue
        estimado = estimar(red)
        faltan = SEMILLAS_MINIMO - len(semillas_terminadas(red))
        if faltan > 0 and reloj.restante() - RESERVA_CIERRE_MIN < faltan * estimado:
            registrar_log(f"etapa10_{red}", f"no caben {faltan} semillas (~{estimado:.0f} min c/u, quedan "
                                            f"{reloj.restante():.0f} min): se omite")
            continue
        correr_semillas(red, SEMILLAS_MINIMO, parametros[red], entorno, estimar)
    for red in parametros:
        if parametros[red] is not None and len(semillas_terminadas(red)) >= SEMILLAS_MINIMO:
            correr_semillas(red, SEMILLAS_MAXIMO, parametros[red], entorno, estimar)

    # Desde aquí no se entrena nada: el corte duro ya no aplica
    en_cierre.set()
    registrar_log("etapa10", f"cierre ({reloj.transcurrido():.1f} min)")
    if tablas_tuneo:
        pd.concat(tablas_tuneo.values(), ignore_index=True).to_csv(RUTA_TABLAS / "etapa10_tuneo.csv", index=False)
    columnas = leer_json(RUTA_ETAPA10 / "columnas_holdout.json") or []
    resultados = {BASE: resultado_desde_predicciones(BASE, verdad),
                  CATBOOST: resultado_desde_predicciones(CATBOOST, verdad)}
    resultados[BASE]["tiempo_min"] = resultados[CATBOOST]["tiempo_min"] = np.nan
    descripciones = {BASE: {"descripcion": "base: A (LightGBM n6, bagging 5 semillas, todas las variables)"},
                     CATBOOST: {"descripcion": "referencia: CatBoost B3 sobre A (etapa 7)"}}

    tiempos, filas_semillas, bags = leer_tiempos(), [], {}
    for red, especificacion in REDES.items():
        terminadas = semillas_terminadas(red)
        for k in terminadas:
            r = resultado_desde_predicciones(nombre_semilla(red, k), verdad)
            filas_semillas.append({"red": red, "semilla": SEMILLA + k, "f1_cv": r["f1_cv"],
                                   "f1_holdout": r["f1_holdout"], "tiempo_min": tiempos.get(nombre_semilla(red, k))})
        if len(terminadas) < SEMILLAS_MINIMO:
            registrar_log("etapa10", f"{red}: {len(terminadas)} semillas terminadas, sin bag")
            continue
        nombre = f"n10_{red}_bag{len(terminadas)}"
        resultados[nombre] = promediar_semillas(nombre, [nombre_semilla(red, k) for k in terminadas], verdad)
        resultados[nombre]["tiempo_min"] = sum(tiempos.get(nombre_semilla(red, k), np.nan) for k in terminadas)
        bags[red] = nombre
        elegidos = {k: v for k, v in parametros[red].items() if k not in especificacion["fijos"]}
        descripciones[nombre] = {
            "descripcion": f"{especificacion['etiqueta']} ({texto_hiperparametros(elegidos)}), "
                           f"bagging {len(terminadas)} semillas",
            "algoritmo": "Red neuronal", "variables": columnas,
            "especificacion": f"{especificacion['especificacion']} | base A completa | bagging "
                              f"{len(terminadas)} semillas",
            "busqueda": f"{especificacion['rango']}; F1 con umbral óptimo en una validación interna del "
                        f"{FRACCION_TUNEO:.0%} del entrenamiento del holdout (folds 1-5)",
            "n_configuraciones": len(tablas_tuneo[red]),
            "hiperparametros": texto_hiperparametros(parametros[red])}
    pd.DataFrame(filas_semillas).to_csv(RUTA_TABLAS / "etapa10_semillas.csv", index=False)

    # 3. Ensambles con pesos elegidos solo en OOF
    for nombre, (resultado, pesos, componentes) in (correr_seguro("etapa10", "ensambles",
                                                                  lambda: ensamblar(bags, verdad)) or {}).items():
        resultados[nombre] = resultado
        texto_pesos = " + ".join(f"{c} {w:.2f}" for c, w in zip(componentes, pesos))
        descripciones[nombre] = {
            "descripcion": f"mezcla {texto_pesos}", "algoritmo": "Stacking", "variables": columnas,
            "especificacion": f"Promedio ponderado de probabilidades: {texto_pesos}",
            "busqueda": ("pesos en {0, 0.05, ..., 1} por F1 OOF" if len(componentes) == 2 else
                         f"pesos en el simplex de paso {PASO_SIMPLEX} por F1 OOF") + "; umbral OOF",
            "n_configuraciones": len(PESOS_MEZCLA) if len(componentes) == 2 else
            len(grilla_simplex(3, PASO_SIMPLEX)),
            "hiperparametros": "; ".join(f"peso_{c} = {w:.2f}" for c, w in zip(componentes, pesos))}

    base_oof = resultados[BASE]["oof"]["probabilidad"]
    correlacion = correlaciones_con(base_oof, {n: r["oof"]["probabilidad"] for n, r in resultados.items()
                                               if n != BASE})
    semillas_oof = {nombre_semilla(red, k): resultado_desde_predicciones(nombre_semilla(red, k), verdad)["oof"]
                    ["probabilidad"] for red in REDES for k in semillas_terminadas(red)}
    pd.concat([correlacion, correlaciones_con(base_oof, semillas_oof)]) \
        .to_csv(RUTA_TABLAS / "etapa10_correlacion_oof.csv", index=False)

    tabla, elegido = cerrar(entorno, resultados, descripciones, correlacion)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(tabla[["candidato", "f1_cv", "f1_cv_sd", "f1_holdout", "precision_holdout", "recall_holdout",
                     "f1_banda_08_12", "tiempo_min", "pearson_con_lightgbm", "delta_cv", "delta_holdout",
                     "cumple_regla_csv"]].round(4))
    # Solo las semillas terminadas: los folds de una semilla a medias sirven para retomarla
    for red in REDES:
        for k in semillas_terminadas(red):
            borrar_parciales(nombre_semilla(red, k))
    registrar_log("etapa10", f"fin: CSV {elegido} ({reloj.transcurrido():.1f} min)")


if __name__ == "__main__":
    main()
