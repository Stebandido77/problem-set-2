"""Etapa 14: dos búsquedas sobre la base de A, desacopladas y con corte duro.

Bases: n13_mezcla_final (CV 0,7753 / holdout 0,7820; regla del CSV) y
n7_a_agregados_bag5 (correlación OOF y mezclas). Mismos folds y holdout de R,
umbral elegido en OOF con buscar_umbral_f1(), variables construidas con la
etiqueta ajustadas solo con el entrenamiento de cada contexto (como la OOF
anidada de los auxiliares) y el F1 de la clase pobre con umbral OOF como aptitud
en las dos búsquedas.

Experimentos (cada uno con su tope de minutos y corte duro):
  genetico  programación genética de variables (gplearn SymbolicTransformer)
            sobre las 60 variables de más gain: fórmulas con +, -, x, /, log, máx
            y mín evolucionadas con cruce y mutación; aptitud = F1 con umbral
            óptimo en una submuestra del entrenamiento del contexto. Las 15
            mejores (menos correlacionadas entre sí) se agregan a A y se ajustan
            LightGBM (n6) y XGBoost (n6 traducido). Primero 1 semilla de cada uno
            contra la referencia de 1 semilla (n12_a_1s); el que mejora +0,002 en
            CV y holdout sigue con más semillas (hasta 5) mientras quepan.
  abejas    colonia artificial de abejas sobre la arquitectura e hiperparámetros
            de RealMLP (capas, ancho, dropout, lr, weight decay, embeddings
            numéricos PBLD sí/no); aptitud = F1 OOF (umbral óptimo) en los folds
            1-3 del 80 %, cada uno ajustado con una submuestra estratificada del
            25 % de su entrenamiento. La mejor configuración se evalúa con los
            5 folds, el holdout y el test (y más semillas si caben).
  cierre    tabla final, mezclas con n7_a_agregados_bag5 (pesos solo en OOF),
            correlación OOF con LightGBM; CSV (máximo 1) solo si supera a
            n13_mezcla_final en CV y en holdout; bitácora y leaderboard.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa14_busquedas.py <experimento> [--minutos N] [--prueba] [--sin-test]
--prueba corre todo en miniatura (candidatos n14p_*, carpeta etapa14_prueba,
tablas etapa14p_*) para probar el flujo sin tocar la bitácora ni el leaderboard.
--sin-test cierra con OOF y holdout los candidatos cuyos 5 folds y holdout ya
están en parciales (un proceso que murió antes del contexto test) y no entrena
ninguno; esos candidatos no pueden dar CSV y se completan con el test al volver
a correr sin la opción.
"""

import argparse
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from configuracion import PARAMETROS_N6, PARAMETROS_N6_XGBOOST, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa4_leaderboard import agregar_filas_extra, armar_fila, texto_hiperparametros
from etapa12_enfoques import (MEJORA_MINIMA, RUTA_MATRICES_A, correr_seguro, crear_preparador_a, fila_tabla,
                              guardar_filas, mejora_sobre, variables_top_gain)
from funciones import bitacora
from funciones.abejas import ColoniaAbejas
from funciones.anidado import definir_contextos
from funciones.bitacora import Presupuesto, iniciar_corte_duro, iniciar_vigilante_memoria, registrar_log
from funciones.datos import cargar_particion
from funciones.evolutivo import muestra_estratificada
from funciones.metricas import buscar_umbral_f1
from funciones.mezclas import mezclar_por_oof, promediar_semillas
from funciones.modelado import (borrar_parciales, crear_ajuste_lightgbm, crear_ajuste_xgboost, evaluar_modelo,
                                resultado_desde_predicciones)
from funciones.redes import crear_ajuste_realmlp
from funciones.submission import guardar_submission

BASE_CSV = "n13_mezcla_final"       # regla del CSV: superar en CV y en holdout
BASE_LGB = "n7_a_agregados_bag5"    # correlación OOF y mezclas
REFERENCIA_1S = "n12_a_1s"          # gate de los candidatos de 1 semilla (LightGBM, XGBoost)
REFERENCIA_REALMLP = "n10_realmlp_s0"  # referencia de RealMLP con 1 semilla (etapa 10)
MINUTOS = {"genetico": 120, "abejas": 90, "cierre": 15}
RESERVA_MIN = 3
PESOS_MEZCLA = [np.array([1 - w, w]) for w in np.round(np.arange(0.0, 0.501, 0.05), 2)]
LIMITE_PROCESO_GB = 7.5  # matrices del contexto test (165 mil x 1.324) más las copias de XGBoost y las redes

# Programación genética (en un subproceso: ver funciones.formulas.main())
N_VARIABLES_GP = 60
N_FORMULAS = 15
N_JOBS_GP = 8  # workers de gplearn (~150 MB cada uno); el sistema anda con poca memoria libre
MINUTOS_XGB_ESTIMADOS = 40  # ~4,7 min por ajuste x 7 contextos, medido en cv_1

# Colonia de abejas sobre RealMLP: codificación en [0, 1] de cada posición
RANGOS_ABC = {"capas": (2, 4), "ancho": (128, 512), "p_drop": (0.0, 0.4), "lr": (0.005, 0.08),
              "wd": (1e-4, 5e-2), "embeddings": ("none", "pbld")}
CONFIGURACION_REALMLP_ETAPA10 = {"hidden_sizes": [256, 256, 256], "p_drop": 0.15, "lr": 0.02, "wd": 1e-2,
                                 "num_emb_type": "none"}
FOLDS_APTITUD_ABC = ("cv_1", "cv_2", "cv_3")
ABC_FUENTES, ABC_LIMITE = 3, 2

# Modo normal; configurar_modo(True) lo cambia por la miniatura de --prueba
MODO = {
    "prueba": False, "prefijo": "n14_", "carpeta": RUTA_NIVEL7 / "etapa14", "tablas": "etapa14",
    "gp": {"poblacion": 1000, "generaciones": 25, "fraccion_filas": 0.25},
    "lgb": PARAMETROS_N6, "xgb": PARAMETROS_N6_XGBOOST, "minutos_xgb": MINUTOS_XGB_ESTIMADOS,
    "fijos_realmlp": {"n_epochs": 32, "batch_size": 1024}, "fraccion_abc": 0.25, "abc_max_ciclos": 50,
    "reserva_final_abc": 40, "semillas_maximo": 5,
    "con_test": True,  # --sin-test: cierra con OOF y holdout lo que ya tiene parciales, sin entrenar nada
}


def configurar_modo(prueba):
    """Miniatura para probar el flujo: candidatos n14p_*, carpeta y tablas aparte."""
    if prueba:
        MODO.update({
            "prueba": True, "prefijo": "n14p_", "carpeta": RUTA_NIVEL7 / "etapa14_prueba", "tablas": "etapa14p",
            "gp": {"poblacion": 60, "generaciones": 2, "fraccion_filas": 0.1},
            "lgb": {**PARAMETROS_N6, "num_iterations": 20}, "xgb": {**PARAMETROS_N6_XGBOOST, "num_boost_round": 20},
            "minutos_xgb": 1, "fijos_realmlp": {"n_epochs": 1, "batch_size": 1024}, "fraccion_abc": 0.02,
            "abc_max_ciclos": 1, "reserva_final_abc": 0, "semillas_maximo": 2,
        })


# ---- Utilidades comunes -------------------------------------------------------

def ruta_predicciones(nombre):
    return RUTA_NIVEL7 / f"predicciones_{nombre}.parquet"


def leer_json(nombre):
    ruta = MODO["carpeta"] / f"{nombre}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def anotar_json(nombre, clave, valor):
    datos = leer_json(nombre)
    datos[clave] = valor
    MODO["carpeta"].mkdir(parents=True, exist_ok=True)
    (MODO["carpeta"] / f"{nombre}.json").write_text(json.dumps(datos, indent=1), encoding="utf-8")


def resultado_guardado(nombre, verdad):
    """Resultado desde las predicciones guardadas, con su tiempo anotado."""
    resultado = resultado_desde_predicciones(nombre, verdad)
    resultado["tiempo_min"] = leer_json("tiempos").get(nombre, np.nan)
    return resultado


CONTEXTOS_SIN_TEST = ("cv_1", "cv_2", "cv_3", "cv_4", "cv_5", "holdout")


def parciales_completos(nombre):
    """True si los 5 folds y el holdout del candidato ya están en parciales/<nombre>/."""
    carpeta = RUTA_NIVEL7 / "parciales" / nombre
    return all((carpeta / f"{contexto}.parquet").exists() for contexto in CONTEXTOS_SIN_TEST)


def tiene_test(nombre):
    """True si las predicciones guardadas del candidato incluyen el test."""
    return bool((pd.read_parquet(ruta_predicciones(nombre), columns=["tipo"])["tipo"] == "test").any())


def minutos_segun_parciales(nombre):
    """Minutos de entrenamiento estimados desde las marcas de tiempo de los
    parciales (los escribió un proceso que murió antes del test): al primer
    contexto, que no tiene marca de inicio, se le asigna el promedio de los demás."""
    marcas = sorted(p.stat().st_mtime for p in (RUTA_NIVEL7 / "parciales" / nombre).glob("*.parquet")
                    if not p.stem.endswith("_importancia"))
    return (marcas[-1] - marcas[0]) / 60 * len(marcas) / (len(marcas) - 1) if len(marcas) > 1 else np.nan


def evaluar_o_cargar(nombre, evaluar, verdad):
    """Resultado desde las predicciones si el candidato ya terminó; si no, lo
    evalúa (reanudable por contexto), anota el tiempo y borra los parciales.

    Con --sin-test (MODO con_test False) los parciales se conservan y el tiempo
    se estima desde sus marcas; un candidato cerrado así (predicciones sin test)
    se completa solo con el test cuando se vuelve a correr sin la opción, y su
    tiempo suma el ya anotado."""
    previo = 0.0
    if ruta_predicciones(nombre).exists():
        resultado = resultado_guardado(nombre, verdad)
        if "prediccion_test" in resultado or not MODO["con_test"]:
            return resultado
        previo = float(np.nan_to_num(resultado["tiempo_min"]))
    resultado = evaluar()
    if MODO["con_test"]:
        resultado["tiempo_min"] += previo
        borrar_parciales(nombre)
    else:
        resultado["tiempo_min"] = minutos_segun_parciales(nombre)
    anotar_json("tiempos", nombre, resultado["tiempo_min"])
    return resultado


def correr_semillas(prefijo, crear_ajuste, preparar, entorno, desde=0, objetivo=None, estimado=None):
    """Semillas s<desde>..s<objetivo-1> de un candidato mientras quepan en el reloj.

    Argumentos: prefijo = nombre del candidato sin la semilla; crear_ajuste =
        función(semilla) -> ajustar() de evaluar_modelo(); preparar = función de
        contextos; entorno = particion, verdad, reloj y etapa; desde = primera
        semilla a correr; objetivo = semillas a tener (MODO por defecto);
        estimado = minutos de una semilla cuando aún no hay ninguna medida
        (None: la primera corre sin chequeo de tiempo).
    Devuelve: nombres de las semillas terminadas (incluidas las previas).
    """
    reloj, etapa = entorno["reloj"], entorno["etapa"]
    objetivo = objetivo or MODO["semillas_maximo"]
    terminadas = [f"{prefijo}_s{k}" for k in range(desde) if ruta_predicciones(f"{prefijo}_s{k}").exists()]
    for k in range(desde, objetivo):
        nombre = f"{prefijo}_s{k}"
        if ruta_predicciones(nombre).exists() and (tiene_test(nombre) or not MODO["con_test"]):
            terminadas.append(nombre)
            continue
        if not MODO["con_test"] and not parciales_completos(nombre):
            registrar_log(etapa, f"{nombre}: sin los parciales de los 5 folds y el holdout; con --sin-test no se entrena")
            break
        tiempos = [leer_json("tiempos")[n] for n in terminadas if n in leer_json("tiempos")]
        necesario = float(np.mean(tiempos)) if tiempos else estimado
        if necesario is not None and reloj.restante() - RESERVA_MIN < necesario:
            registrar_log(etapa, f"{nombre}: no cabe (~{necesario:.0f} min, quedan {reloj.restante():.0f})")
            break
        resultado = correr_seguro(etapa, nombre, lambda: evaluar_o_cargar(nombre, lambda: evaluar_modelo(
            nombre, preparar, crear_ajuste(SEMILLA + k), entorno["particion"], con_test=MODO["con_test"],
            reanudable=True),
            entorno["verdad"]))
        gc.collect()
        if resultado is None:
            break
        terminadas.append(nombre)
    return terminadas


def promediar_si_caben(prefijo, terminadas, verdad):
    """Bagging de las semillas terminadas (si hay al menos dos): (nombre, resultado) o None."""
    if len(terminadas) < 2:
        return None
    nombre = f"{prefijo}_bag{len(terminadas)}"
    resultado = promediar_semillas(nombre, terminadas, verdad)
    resultado["tiempo_min"] = float(sum(leer_json("tiempos").get(n, np.nan) for n in terminadas))
    anotar_json("tiempos", nombre, resultado["tiempo_min"])
    return nombre, resultado


def supera_en_ambos(resultado, base):
    """Regla del CSV de la etapa: F1 CV y F1 holdout estrictamente mayores que los de la base."""
    return bool(resultado["f1_cv"] > base["f1_cv"] and resultado["f1_holdout"] > base["f1_holdout"])


def anotar_metadatos(nombre, algoritmo, especificacion, busqueda, n_configuraciones, hiperparametros,
                     variables_extra=()):
    """Guarda lo que el leaderboard necesita de un candidato (lo lee el cierre)."""
    anotar_json("metadatos", nombre, {
        "algoritmo": algoritmo, "especificacion": especificacion, "busqueda": busqueda,
        "n_configuraciones": int(n_configuraciones), "hiperparametros": hiperparametros,
        "variables_extra": list(variables_extra)})


# ---- genetico: programación genética de variables + LightGBM y XGBoost ---------

def esperar_subproceso_previo(ruta_pid):
    """Si un relanzamiento (por memoria) dejó vivo el subproceso de gplearn de este
    contexto, espera a que termine en vez de lanzar otro."""
    if not ruta_pid.exists():
        return
    try:
        pid = int(ruta_pid.read_text(encoding="utf-8"))
        proceso = psutil.Process(pid)
        if "funciones.formulas" in " ".join(proceso.cmdline()):
            proceso.wait()
    except (ValueError, psutil.Error):
        pass
    ruta_pid.unlink(missing_ok=True)


def ajustar_formulas_contexto(contexto, columnas, ruta, ruta_formulas):
    """Lanza `python -m funciones.formulas` para un contexto (proceso liviano, sin
    torch, para que loky pueda paralelizar) y deja la tabla y las fórmulas en disco."""
    ruta_pid = ruta.with_suffix(".pid")
    esperar_subproceso_previo(ruta_pid)
    if ruta.exists():
        return
    gp = MODO["gp"]
    comando = [sys.executable, "-m", "funciones.formulas",
               "--entrenamiento", str(RUTA_MATRICES_A / f"{contexto}_ent.parquet"),
               "--evaluacion", str(RUTA_MATRICES_A / f"{contexto}_eval.parquet"),
               "--salida", str(ruta), "--formulas", str(ruta_formulas), "--n-formulas", str(N_FORMULAS),
               "--poblacion", str(gp["poblacion"]), "--generaciones", str(gp["generaciones"]),
               "--fraccion-filas", str(gp["fraccion_filas"]), "--semilla", str(SEMILLA), "--n-jobs", str(N_JOBS_GP),
               "--columnas", *columnas]
    with open(ruta.with_suffix(".log"), "w", encoding="utf-8") as registro:
        proceso = subprocess.Popen(comando, cwd=Path(__file__).resolve().parent, stdout=registro, stderr=registro)
        ruta_pid.write_text(str(proceso.pid), encoding="utf-8")
        codigo = proceso.wait()
    ruta_pid.unlink(missing_ok=True)
    if codigo != 0 or not ruta.exists():
        raise RuntimeError(f"gplearn falló en {contexto} (código {codigo}); ver {ruta.with_suffix('.log')}")


def variables_formulas(contextos, entorno):
    """{contexto: DataFrame} de las fórmulas de cada contexto, ajustadas solo con
    su entrenamiento y aplicadas a entrenamiento y evaluación; caché por contexto
    y tabla con las fórmulas ganadoras."""
    etapa, reloj = entorno["etapa"], entorno["reloj"]
    columnas = variables_top_gain(N_VARIABLES_GP)
    carpeta = MODO["carpeta"] / "gp"
    carpeta.mkdir(parents=True, exist_ok=True)
    salida, filas = {}, []
    for contexto in contextos:
        ruta, ruta_formulas = carpeta / f"{contexto}.parquet", carpeta / f"{contexto}_formulas.json"
        if not ruta.exists():
            inicio = time.time()
            ajustar_formulas_contexto(contexto, columnas, ruta, ruta_formulas)
            formulas = json.loads(ruta_formulas.read_text(encoding="utf-8"))
            registrar_log(etapa, f"fórmulas {contexto}: mejor F1 {max(f['aptitud'] for f in formulas):.4f} "
                                 f"({(time.time() - inicio) / 60:.1f} min; quedan {reloj.restante():.0f})")
        salida[contexto] = pd.read_parquet(ruta)
        formulas = json.loads(ruta_formulas.read_text(encoding="utf-8"))
        filas += [{"contexto": contexto, "orden": i + 1, **f} for i, f in enumerate(formulas)]
    pd.DataFrame(filas).to_csv(RUTA_TABLAS / f"{MODO['tablas']}_gp_formulas.csv", index=False)
    return salida


# Clave (también la de sus parámetros en MODO) -> (etiqueta, crear_ajuste(parametros, semillas))
MODELOS_GP = {"lgb": ("LightGBM n6", crear_ajuste_lightgbm), "xgb": ("XGBoost (n6 traducido)", crear_ajuste_xgboost)}


def busqueda_gp():
    gp = MODO["gp"]
    return (f"programación genética (gplearn SymbolicTransformer): población {gp['poblacion']}, "
            f"{gp['generaciones']} generaciones, torneo 20, cruce 0,7, mutación de subárbol 0,1, de elevación 0,05 "
            f"y de punto 0,1, parsimonia 0,0005; aptitud F1 con umbral óptimo en el {gp['fraccion_filas']:.0%} del "
            f"entrenamiento del contexto; {N_FORMULAS} fórmulas menos correlacionadas entre las 50 mejores; "
            f"hiperparámetros del modelo fijos (n6)")


def genetico(entorno, reloj):
    etapa, particion, verdad = entorno["etapa"], entorno["particion"], entorno["verdad"]
    referencia = resultado_desde_predicciones(REFERENCIA_1S, verdad)
    base = resultado_desde_predicciones(BASE_LGB, verdad)
    oof_lgb = base["oof"]["probabilidad"]
    extras = variables_formulas(list(definir_contextos()), entorno)
    nombres_formulas = list(extras["holdout"].columns)
    preparar = crear_preparador_a([extras])
    filas, pasan, prefijos, crear_ajustes = [], {}, {}, {}
    for clave, (_, crear) in MODELOS_GP.items():
        prefijos[clave] = f"{MODO['prefijo']}a_gp_{clave}"
        crear_ajustes[clave] = lambda semilla, clave=clave, crear=crear: crear(MODO[clave], (semilla,))

    # 1. Una semilla de cada modelo contra la referencia de 1 semilla
    for clave, (etiqueta, _) in MODELOS_GP.items():
        estimado = MODO["minutos_xgb"] if clave == "xgb" else None
        terminadas = correr_semillas(prefijos[clave], crear_ajustes[clave], preparar, entorno, objetivo=1,
                                     estimado=estimado)
        if not terminadas:
            pasan[clave] = False
            continue
        s0 = resultado_guardado(terminadas[0], verdad)
        descripcion = f"A + {N_FORMULAS} fórmulas GP, {etiqueta}, 1 semilla"
        filas.append(fila_tabla(s0, verdad, descripcion, referencia, oof_lgb))
        anotar_metadatos(terminadas[0], "Boosting", f"{etiqueta} sobre A + {N_FORMULAS} fórmulas de programación "
                         f"genética ({N_VARIABLES_GP} variables de más gain; +, -, x, /, log, máx, mín) | 1 semilla",
                         busqueda_gp(), MODO["gp"]["poblacion"] * MODO["gp"]["generaciones"],
                         texto_hiperparametros(MODO[clave]), nombres_formulas)
        delta_cv, delta_holdout, pasan[clave] = mejora_sobre(s0, referencia)
        registrar_log(etapa, f"{prefijos[clave]} 1 semilla: CV {delta_cv:+.4f}, holdout {delta_holdout:+.4f} frente "
                             f"a {REFERENCIA_1S} -> {'pasa' if pasan[clave] else 'no pasa'} a más semillas")
        guardar_filas(f"{MODO['tablas']}_candidatos", filas)

    # 2. Más semillas (hasta 5) para los que pasan, LightGBM primero porque es más barato
    for clave, (etiqueta, _) in MODELOS_GP.items():
        if not pasan.get(clave):
            continue
        terminadas = correr_semillas(prefijos[clave], crear_ajustes[clave], preparar, entorno, desde=1)
        bag = promediar_si_caben(prefijos[clave], terminadas, verdad)
        if bag is None:
            continue
        nombre, resultado = bag
        filas.append(fila_tabla(resultado, verdad, f"A + {N_FORMULAS} fórmulas GP, {etiqueta}, bagging "
                                f"{len(terminadas)} semillas", base, oof_lgb))
        anotar_metadatos(nombre, "Boosting", f"{etiqueta} sobre A + {N_FORMULAS} fórmulas de programación genética "
                         f"({N_VARIABLES_GP} variables de más gain; +, -, x, /, log, máx, mín) | bagging "
                         f"{len(terminadas)} semillas", busqueda_gp(),
                         MODO["gp"]["poblacion"] * MODO["gp"]["generaciones"], texto_hiperparametros(MODO[clave]),
                         nombres_formulas)
        guardar_filas(f"{MODO['tablas']}_candidatos", filas)
    print(pd.DataFrame(filas).round(4).to_string(index=False) if filas else "sin candidatos")
    return filas


# ---- abejas: colonia artificial sobre RealMLP ----------------------------------

def decodificar_abc(z):
    """Vector en [0, 1]^6 -> hiperparámetros de RealMLP (capas, ancho, dropout, lr y
    weight decay en escala log, embeddings numéricos PBLD si la última posición >= 0,5)."""
    z = np.clip(np.asarray(z, dtype=float), 0, 1)
    capas = RANGOS_ABC["capas"][0] + min(int(z[0] * 3), 2)
    ancho = 128 * (1 + min(int(z[1] * 4), 3))
    log_lr = np.log(RANGOS_ABC["lr"][0]) + z[3] * (np.log(RANGOS_ABC["lr"][1]) - np.log(RANGOS_ABC["lr"][0]))
    log_wd = np.log(RANGOS_ABC["wd"][0]) + z[4] * (np.log(RANGOS_ABC["wd"][1]) - np.log(RANGOS_ABC["wd"][0]))
    return {"hidden_sizes": [ancho] * capas, "p_drop": float(RANGOS_ABC["p_drop"][1] * z[2]),
            "lr": float(np.exp(log_lr)), "wd": float(np.exp(log_wd)),
            "num_emb_type": RANGOS_ABC["embeddings"][1] if z[5] >= 0.5 else RANGOS_ABC["embeddings"][0]}


def codificar_abc(parametros):
    """Hiperparámetros de RealMLP -> vector en [0, 1]^6 (centro de la celda en las discretas)."""
    capas, ancho = len(parametros["hidden_sizes"]), parametros["hidden_sizes"][0]
    return np.array([
        (capas - RANGOS_ABC["capas"][0] + 0.5) / 3, (ancho // 128 - 1 + 0.5) / 4,
        parametros["p_drop"] / RANGOS_ABC["p_drop"][1],
        (np.log(parametros["lr"]) - np.log(RANGOS_ABC["lr"][0])) / (np.log(RANGOS_ABC["lr"][1]) - np.log(RANGOS_ABC["lr"][0])),
        (np.log(parametros["wd"]) - np.log(RANGOS_ABC["wd"][0])) / (np.log(RANGOS_ABC["wd"][1]) - np.log(RANGOS_ABC["wd"][0])),
        0.75 if parametros["num_emb_type"] == RANGOS_ABC["embeddings"][1] else 0.25,
    ])


def describir_realmlp(parametros):
    """Texto corto de una configuración de RealMLP."""
    return (f"{len(parametros['hidden_sizes'])} x {parametros['hidden_sizes'][0]}, dropout {parametros['p_drop']:.2f}, "
            f"lr {parametros['lr']:.4f}, wd {parametros['wd']:.4f}, embeddings {parametros['num_emb_type']}")


def cargar_contextos_aptitud(etapa):
    """Folds de la aptitud: submuestra estratificada del entrenamiento (misma para
    todas las configuraciones) y la evaluación completa de cada contexto."""
    rng = np.random.default_rng(SEMILLA)
    contextos = {}
    for contexto in FOLDS_APTITUD_ABC:
        x_ent = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_ent.parquet")
        y_ent = x_ent.pop("pobre").to_numpy().astype(int)
        filas = muestra_estratificada(y_ent, int(round(MODO["fraccion_abc"] * len(y_ent))), rng)
        x_eval = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_eval.parquet")
        contextos[contexto] = (x_ent.iloc[filas].copy(), y_ent[filas], x_eval)
        del x_ent
        gc.collect()
    registrar_log(etapa, f"aptitud sobre {', '.join(FOLDS_APTITUD_ABC)}: "
                         f"{len(contextos['cv_1'][1])} filas de ajuste por fold")
    return contextos


def crear_aptitud_abc(cargar_contextos, verdad):
    """Aptitud de la colonia: F1 OOF (umbral óptimo) de los folds de FOLDS_APTITUD_ABC.

    Argumentos: cargar_contextos = función() -> contextos de cargar_contextos_aptitud(),
        llamada solo al evaluar (una colonia retomada sin ciclos pendientes no los
        carga); verdad = pobre por id.
    """
    def aptitud(z):
        parametros = {**decodificar_abc(z), **MODO["fijos_realmlp"]}
        inicio = time.time()
        verdades, probabilidades, detalle = [], [], {}
        for contexto, (x_ent, y_ent, x_eval) in cargar_contextos().items():
            probabilidad, _ = crear_ajuste_realmlp(parametros, SEMILLA)(x_ent, y_ent, x_eval)
            y_eval = verdad.loc[x_eval.index].astype(int).to_numpy()
            verdades.append(y_eval)
            probabilidades.append(probabilidad)
            detalle[f"f1_{contexto}"] = buscar_umbral_f1(y_eval, probabilidad)[1]
            gc.collect()
        umbral, f1 = buscar_umbral_f1(np.concatenate(verdades), np.concatenate(probabilidades))
        detalle.update({"capas": len(parametros["hidden_sizes"]), "ancho": parametros["hidden_sizes"][0],
                        "p_drop": parametros["p_drop"], "lr": parametros["lr"], "wd": parametros["wd"],
                        "embeddings": parametros["num_emb_type"], "umbral": umbral,
                        "segundos": time.time() - inicio})
        return float(f1), detalle

    return aptitud


def abejas(entorno, reloj):
    etapa, particion, verdad = entorno["etapa"], entorno["particion"], entorno["verdad"]
    cargados = {}

    def contextos():
        if not cargados:
            cargados.update(cargar_contextos_aptitud(etapa))
        return cargados

    colonia = ColoniaAbejas(crear_aptitud_abc(contextos, verdad), n_binarias=0, n_continuas=6,
                            n_fuentes=ABC_FUENTES, limite=ABC_LIMITE, ruta=MODO["carpeta"] / "abc", semilla=SEMILLA,
                            iniciales=[codificar_abc(CONFIGURACION_REALMLP_ETAPA10)],
                            registrar=lambda texto: registrar_log(f"{etapa}_abc", texto))
    while colonia.ciclo < MODO["abc_max_ciclos"]:
        ciclo_min = (2 * ABC_FUENTES + 1) * max(f["segundos"] for f in colonia.cache.values()) / 60
        if reloj.restante() - MODO["reserva_final_abc"] < ciclo_min:
            registrar_log(etapa, f"no cabe otro ciclo (~{ciclo_min:.0f} min; quedan {reloj.restante():.0f} y la "
                                 f"evaluación completa reserva {MODO['reserva_final_abc']})")
            break
        colonia.correr_ciclo()
    historia = pd.DataFrame(colonia.cache.values()).drop(columns=["z"])
    historia.to_csv(RUTA_TABLAS / f"{MODO['tablas']}_abc_historia.csv", index=False)
    z, mejor_aptitud = colonia.mejor()
    parametros = {**decodificar_abc(z), **MODO["fijos_realmlp"]}
    registrar_log(etapa, f"fin de la búsqueda: ciclo {colonia.ciclo}, {len(colonia.cache)} evaluaciones, mejor F1 "
                         f"{mejor_aptitud:.4f} con {describir_realmlp(parametros)}")
    cargados.clear()
    gc.collect()

    # Evaluación completa de la mejor configuración (5 folds, holdout y test), y más semillas si caben
    prefijo = f"{MODO['prefijo']}realmlp_abc"
    terminadas = correr_semillas(prefijo, lambda semilla: crear_ajuste_realmlp(parametros, semilla),
                                 crear_preparador_a(), entorno)
    if not terminadas:
        return []
    referencia = resultado_desde_predicciones(REFERENCIA_REALMLP, verdad)
    base = resultado_desde_predicciones(BASE_LGB, verdad)
    busqueda = (f"colonia artificial de abejas ({ABC_FUENTES} fuentes, límite {ABC_LIMITE}): capas {{2, 3, 4}}, ancho "
                f"{{128, 256, 384, 512}}, dropout [0, 0,4], lr [0,005, 0,08] log, weight decay [1e-4, 5e-2] log, "
                f"embeddings PBLD sí/no; aptitud F1 OOF (umbral óptimo) en los folds 1-3 con el "
                f"{MODO['fraccion_abc']:.0%} del entrenamiento de cada uno; {len(colonia.cache)} evaluaciones")
    especificacion = (f"RealMLP-TD (pytabkit) {describir_realmlp(parametros)}, {MODO['fijos_realmlp']['n_epochs']} "
                      f"épocas con early stopping por log loss en un 10% interno | base A completa")
    filas = []
    for nombre in terminadas:
        resultado = resultado_guardado(nombre, verdad)
        filas.append(fila_tabla(resultado, verdad, f"RealMLP por colonia de abejas ({describir_realmlp(parametros)}), "
                                f"1 semilla", referencia, base["oof"]["probabilidad"]))
        anotar_metadatos(nombre, "Red neuronal", f"{especificacion} | 1 semilla", busqueda, len(colonia.cache),
                         texto_hiperparametros({**parametros, "hidden_sizes": str(parametros["hidden_sizes"])}))
    bag = promediar_si_caben(prefijo, terminadas, verdad)
    if bag is not None:
        nombre, resultado = bag
        filas.append(fila_tabla(resultado, verdad, f"RealMLP por colonia de abejas, bagging {len(terminadas)} "
                                f"semillas", base, base["oof"]["probabilidad"]))
        anotar_metadatos(nombre, "Red neuronal", f"{especificacion} | bagging {len(terminadas)} semillas", busqueda,
                         len(colonia.cache), texto_hiperparametros({**parametros, "hidden_sizes": str(parametros["hidden_sizes"])}))
    guardar_filas(f"{MODO['tablas']}_candidatos", filas)
    print(pd.DataFrame(filas).round(4).to_string(index=False))
    return filas


# ---- cierre: tabla, mezclas, CSV (máximo 1), bitácora y leaderboard ------------

def candidatos_guardados():
    """Candidatos de la etapa con predicciones guardadas, sin las mezclas del
    cierre y sin las semillas sueltas de un candidato que ya tiene bagging."""
    prefijo = MODO["prefijo"]
    nombres = sorted(p.stem[len("predicciones_"):] for p in RUTA_NIVEL7.glob(f"predicciones_{prefijo}*.parquet")
                     if not p.stem.startswith(f"predicciones_{prefijo}mezcla_"))
    con_bag = {n.rsplit("_bag", 1)[0] for n in nombres if "_bag" in n}
    return [n for n in nombres if not ("_s" in n and n.rsplit("_s", 1)[0] in con_bag and n.rsplit("_s", 1)[1].isdigit())]


def algoritmo_de(nombre):
    if "_mezcla_" in nombre:
        return "Stacking"
    return "Red neuronal" if "realmlp" in nombre else "Boosting"


def cierre(entorno, reloj):
    etapa, particion, verdad = entorno["etapa"], entorno["particion"], entorno["verdad"]
    prefijo, tablas = MODO["prefijo"], MODO["tablas"]
    base_csv = resultado_desde_predicciones(BASE_CSV, verdad)
    base_lgb = resultado_desde_predicciones(BASE_LGB, verdad)
    oof_lgb = base_lgb["oof"]["probabilidad"]
    ruta_candidatos = RUTA_TABLAS / f"{tablas}_candidatos.csv"
    previas = pd.read_csv(ruta_candidatos) if ruta_candidatos.exists() else pd.DataFrame(columns=["candidato"])
    descripciones = dict(zip(previas.get("candidato", []), previas.get("descripcion", [])))
    metadatos = leer_json("metadatos")

    def fila_con_regla(resultado, descripcion):
        fila = fila_tabla(resultado, verdad, descripcion, base_csv, oof_lgb)
        fila["delta_cv_vs_bag5"] = resultado["f1_cv"] - base_lgb["f1_cv"]
        fila["delta_holdout_vs_bag5"] = resultado["f1_holdout"] - base_lgb["f1_holdout"]
        fila["tiene_test"] = "prediccion_test" in resultado
        fila["supera_n13"] = supera_en_ambos(resultado, base_csv) and fila["tiene_test"]
        return fila

    filas = [fila_con_regla(base_csv, "base: n13_mezcla_final (LightGBM 0,80 + RealMLP 0,15 + DeepSets 0,05)"),
             fila_con_regla(base_lgb, "base: n7_a_agregados_bag5 (LightGBM n6 sobre A, bagging 5 semillas)")]
    for nombre in candidatos_guardados():
        resultado = resultado_guardado(nombre, verdad)
        filas.append(fila_con_regla(resultado, descripciones.get(nombre, nombre)))
        # La mezcla se calcula aunque el candidato no tenga test (sin test no hay CSV, pero sí CV y holdout)
        resto = nombre[len(prefijo):]
        nombre_mezcla = f"{prefijo}mezcla_lgb_{resto}"
        mezcla, pesos, tabla_pesos = mezclar_por_oof(nombre_mezcla, [BASE_LGB, nombre], PESOS_MEZCLA, verdad)
        tabla_pesos.to_csv(RUTA_TABLAS / f"{tablas}_pesos_{resto}.csv", index=False)
        texto_pesos = f"{BASE_LGB} {pesos[0]:.2f} + {nombre} {pesos[1]:.2f}"
        filas.append(fila_con_regla(mezcla, f"mezcla {texto_pesos} (pesos por OOF)"))
        metadatos[nombre_mezcla] = {
            "algoritmo": "Stacking", "especificacion": f"Promedio ponderado de probabilidades: {texto_pesos}",
            "busqueda": "pesos en {0, 0.05, ..., 0.5} por F1 OOF; umbral OOF", "n_configuraciones": len(PESOS_MEZCLA),
            "hiperparametros": f"peso_{BASE_LGB} = {pesos[0]:.2f}; peso_{nombre} = {pesos[1]:.2f}",
            "variables_extra": metadatos.get(nombre, {}).get("variables_extra", [])}
        registrar_log(etapa, f"{nombre_mezcla}: pesos {np.round(pesos, 2)} | CV {mezcla['f1_cv']:.4f} | holdout "
                             f"{mezcla['f1_holdout']:.4f}")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / f"{tablas}_resumen.csv", index=False)

    # CSV: a lo sumo uno, el de mayor holdout entre los que superan a n13 en CV y en holdout
    cumplen = tabla[tabla["supera_n13"] & tabla["candidato"].str.startswith(prefijo)]
    elegido = cumplen.sort_values("f1_holdout", ascending=False)["candidato"].iloc[0] if len(cumplen) else None
    registrar_log(etapa, f"superan a {BASE_CSV} en CV y holdout: {list(cumplen['candidato'])} -> CSV: {elegido}")
    ids_test = particion.loc[particion["base"] == "test", "id"]
    columnas = json.loads((RUTA_NIVEL7 / "etapa10" / "columnas_holdout.json").read_text(encoding="utf-8"))
    filas_lb = []
    for nombre, datos in metadatos.items():
        if not ruta_predicciones(nombre).exists():
            continue
        archivo = None
        if nombre == elegido and not MODO["prueba"]:
            resultado = resultado_desde_predicciones(nombre, verdad)
            ruta, tasa = guardar_submission(resultado, f"{nombre}.csv", ids_test,
                                            f"{datos['especificacion']} | umbral {resultado['umbral']:.4f} (OOF)",
                                            algoritmo=datos["algoritmo"])
            archivo = f"nivel7/{ruta.name}"
            registrar_log(etapa, f"CSV {ruta.name} (tasa de pobres {tasa:.3f})")
        filas_lb.append(armar_fila(nombre, datos["algoritmo"], datos["especificacion"],
                                   datos.get("variables", columnas) + datos.get("variables_extra", []), datos["busqueda"],
                                   datos["n_configuraciones"], datos["hiperparametros"],
                                   leer_json("tiempos").get(nombre, pd.NA), archivo, verdad))
    if filas_lb and not MODO["prueba"]:
        agregar_filas_extra(filas_lb)
    elif MODO["prueba"]:
        registrar_log(etapa, f"prueba: {len(filas_lb)} filas de leaderboard y el CSV no se escriben")
    columnas_tabla = ["candidato", "f1_cv", "f1_cv_sd", "f1_holdout", "precision_holdout", "recall_holdout",
                      "f1_banda_08_12", "pearson_oof_con_lightgbm", "delta_cv", "delta_holdout", "supera_n13",
                      "tiempo_min"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(tabla[[c for c in columnas_tabla if c in tabla.columns]].round(4).to_string(index=False))
    return tabla


# ---- Orquestación -------------------------------------------------------------

EXPERIMENTOS = {"genetico": genetico, "abejas": abejas, "cierre": cierre}


def main(experimento, minutos, prueba=False, sin_test=False):
    configurar_modo(prueba)
    MODO["con_test"] = not sin_test
    MODO["carpeta"].mkdir(parents=True, exist_ok=True)
    etapa = f"etapa14_{experimento}" + ("_prueba" if prueba else "")
    # El reloj se guarda en disco: un relanzamiento (salida por memoria) sigue contando
    reloj = Presupuesto(minutos, MODO["carpeta"] / f"inicio_{experimento}.txt")
    registrar_log(etapa, f"inicio ({minutos} min{', sin test' if sin_test else ''})")
    bitacora.LIMITE_PROCESO_GB = LIMITE_PROCESO_GB
    iniciar_vigilante_memoria(etapa)
    desactivar = threading.Event()
    iniciar_corte_duro(etapa, reloj, desactivar)
    particion = cargar_particion()
    entorno = {"particion": particion, "verdad": particion.set_index("id")["pobre"], "reloj": reloj, "etapa": etapa}
    salida = correr_seguro(etapa, experimento, lambda: EXPERIMENTOS[experimento](entorno, reloj))
    desactivar.set()
    registrar_log(etapa, f"fin ({reloj.transcurrido():.1f} min){'' if salida is not None else ' con falla'}")


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("experimento", choices=list(MINUTOS))
    analizador.add_argument("--minutos", type=float, default=None, help="tope de minutos (por defecto el de la etapa)")
    analizador.add_argument("--prueba", action="store_true", help="miniatura para probar el flujo")
    analizador.add_argument("--sin-test", action="store_true",
                            help="cierra con OOF y holdout los candidatos que ya tienen parciales de los 5 folds y "
                                 "el holdout (sin reajuste sobre el 100 %% ni CSV) y no entrena ninguno")
    args = analizador.parse_args()
    main(args.experimento, args.minutos or MINUTOS[args.experimento], args.prueba, args.sin_test)
