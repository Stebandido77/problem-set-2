"""Etapa 15: experimentos de modelo y análisis de política, desacoplados, con
corte duro, vigilante de memoria y guardado reanudable.

Bases: n13_mezcla_final (CV 0,7753 / holdout 0,7820; regla del CSV) y
n7_a_agregados_bag5 (correlación OOF y mezclas). Mismos folds y holdout de R,
umbral elegido en OOF con buscar_umbral_f1(), OOF anidada en lo que se construye
con la etiqueta y el F1 de la clase pobre con umbral OOF como métrica.

Parte A, experimentos de modelo (cada uno con su tope de minutos):
  fold      ensamble por fold: en vez del reajuste sobre el 100 %, el promedio en
            test de los 5 modelos de la CV de la base, si sus predicciones de
            test por fold están guardadas; si no, se informa y se sigue.
  pseudo    pseudoetiquetado: los hogares del test de Kaggle con P(pobre) >= 0,95
            o <= 0,05 según el LightGBM n6 ajustado solo con el entrenamiento del
            contexto entran a ese entrenamiento con su etiqueta predicha y peso
            0,5; se reentrena LightGBM n6 (1 semilla contra n12_a_1s y bagging de
            hasta 5 si mejora +0,002 en CV y holdout).
  fex       factores de expansión como pesos: LightGBM n6 sobre A con
            sample_weight = Fex_c / media(Fex_c) del entrenamiento del contexto;
            F1 sin ponderar (el de Kaggle) y ponderado por Fex_c en holdout.
  ebm       Explainable Boosting Machine (interpret) sobre las 60 variables de
            más gain de A con interacciones automáticas; curvas de las 10
            variables más importantes en views/figuras/nivel7/ebm_*.png.
Parte B, análisis de política con n13_mezcla_final en holdout (sin reentrenar):
  politica  errores por subgrupo (zona, tamaño, jefatura, adultos mayores sin
            pensión, dominio); subcobertura y filtración ponderadas por la brecha
            de pobreza y la distancia a la línea para n13, nE_logit, n3_logit y la
            encuesta corta de 8 variables; curva de cobertura.
  cierre    tabla final, mezclas con n7_a_agregados_bag5 (pesos solo en OOF), CSV
            (máximo 1) solo si supera a n13_mezcla_final en CV y en holdout,
            bitácora y leaderboard (el cierre de la etapa 14 con prefijo n15_).

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa15_modelos_politica.py <experimento> [--minutos N] [--sin-test]
    python/.venv/Scripts/python.exe python/etapa15_modelos_politica.py todo [--minutos 180]
`todo` corre los seis en orden como subprocesos, relanza el que sale por memoria
y escribe stores/procesados/nivel7/etapa15/fin.txt al terminar. Desacoplado de la
terminal (PowerShell):
    Start-Process -FilePath python\\.venv\\Scripts\\python.exe -ArgumentList python\\etapa15_modelos_politica.py,todo
      -WorkingDirectory . -WindowStyle Hidden -RedirectStandardOutput stores\\logs\\etapa15_todo_salida.txt
      -RedirectStandardError stores\\logs\\etapa15_todo_error.txt
"""

import argparse
import gc
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

from configuracion import N_HILOS, PARAMETROS_N6, RAIZ, RUTA_FIGURAS, RUTA_LOGS, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa4_leaderboard import texto_hiperparametros
from funciones import bitacora
from funciones.bitacora import (CODIGO_SALIDA_MEMORIA, Presupuesto, iniciar_corte_duro, iniciar_vigilante_memoria,
                                registrar_log)
from funciones.datos import cargar_hogares, cargar_ingreso_hogar, cargar_particion, cargar_personas, preparar_matriz
from funciones.metricas import calcular_metricas
from funciones.modelado import (ajustar_lightgbm, crear_ajuste_lightgbm, guardar_parquet_atomico,
                                resultado_desde_predicciones)
from funciones.politica import (curva_cobertura, errores_por_subgrupo, f1_ponderado, grupos_de_hogares,
                                pesos_bienestar, punto_operacion)

BASE_CSV = "n13_mezcla_final"       # regla del CSV: superar en CV y en holdout
BASE_LGB = "n7_a_agregados_bag5"    # correlación OOF, mezclas y ensamble por fold
REFERENCIA_1S = "n12_a_1s"          # gate de los candidatos de 1 semilla
PREFIJO, TABLAS = "n15_", "etapa15"
CARPETA = RUTA_NIVEL7 / "etapa15"
CONTEXTOS = ("cv_1", "cv_2", "cv_3", "cv_4", "cv_5", "holdout", "test")
MINUTOS = {"fold": 20, "pseudo": 75, "fex": 30, "ebm": 40, "politica": 15, "cierre": 15}
MINUTOS_TOTAL = 180
RELANZAMIENTOS = 8        # intentos de un experimento que sale por memoria
LIMITE_PROCESO_GB = 7.5   # matrices del contexto test (165 mil x 1.324) más las pseudoetiquetas
CON_TEST = True           # --sin-test lo apaga (ver etapa14_busquedas)
PRUEBA = False            # --prueba: miniatura (ver configurar_prueba())
PARAMETROS_LGB = PARAMETROS_N6

# Pseudoetiquetado
UMBRAL_PSEUDO_BAJO, UMBRAL_PSEUDO_ALTO, PESO_PSEUDO = 0.05, 0.95, 0.5
MINUTOS_SEMILLA_PSEUDO, MINUTOS_SEMILLA_FEX = 12, 10  # estimados para la primera semilla

# EBM: 60 variables de más gain, interacciones elegidas por el propio EBM (FAST)
N_VARIABLES_EBM, N_CURVAS_EBM = 60, 10
PARAMETROS_EBM = {"outer_bags": 8, "interactions": 20, "max_rounds": 8000}
MINUTOS_EBM_ESTIMADOS = 15

# Política
MODELOS_R = {"n3_logit": "Logit nivel 3 (48 variables, R)", "nE_logit": "Logit nivel E (10 variables, R)"}
VARIABLES_ENCUESTA_CORTA = ["prop_ocupados", "recibe_arriendos", "log_arriendo_relativo", "n_cotizan",
                            "n_quiere_mas_horas", "horas_por_persona_ug", "arriendo_por_persona_lp",
                            "dormitorios_por_persona"]  # punto de 8 variables de la frontera (etapa 3b)
ENCUESTA_CORTA = "encuesta_corta_8"
FACTOR_MARCA = 1.5
DIMENSIONES_BARRAS = ["zona", "tamano", "jefatura", "mayor_sin_pension"]
ETIQUETAS_DIMENSION = {"zona": "Zona", "tamano": "Tamaño del hogar", "jefatura": "Jefatura",
                       "mayor_sin_pension": "Adulto mayor sin pensión", "dominio": "Dominio"}
AZUL, NARANJA, GRIS, TINTA, TINTA_SUAVE = "#2a78d6", "#e8710a", "#9aa0a6", "#202124", "#5f6368"


# ---- Infraestructura compartida con la etapa 14 ----------------------------------

def harness():
    """etapa14_busquedas configurada para esta etapa (prefijo n15_, carpeta y tablas
    propias): correr_semillas(), promediar_si_caben(), metadatos, tiempos y cierre()."""
    import etapa14_busquedas as etapa14  # noqa: PLC0415 (carga torch; solo cuando se entrena o se cierra)
    etapa14.MODO.update({"prueba": PRUEBA, "prefijo": PREFIJO, "carpeta": CARPETA, "tablas": TABLAS,
                         "semillas_maximo": 5, "con_test": CON_TEST})
    return etapa14


def configurar_prueba():
    """Miniatura para probar el flujo: candidatos n15p_*, carpeta etapa15_prueba,
    tablas etapa15p_*, LightGBM de 20 iteraciones, EBM mínimo, figuras del EBM en
    views/figuras/nivel7/prueba/; el cierre no escribe CSV ni leaderboard."""
    global PRUEBA, PREFIJO, TABLAS, CARPETA, PARAMETROS_LGB, PARAMETROS_EBM
    PRUEBA, PREFIJO, TABLAS, CARPETA = True, "n15p_", "etapa15p", RUTA_NIVEL7 / "etapa15_prueba"
    PARAMETROS_LGB = {**PARAMETROS_N6, "num_iterations": 20}
    PARAMETROS_EBM = {"outer_bags": 2, "interactions": 2, "max_rounds": 50}


def preparador_a():
    from etapa12_enfoques import crear_preparador_a  # noqa: PLC0415
    return crear_preparador_a()


def matriz_test_kaggle(columnas=None):
    """Base de A de los hogares del test de Kaggle (auxiliares del contexto test)."""
    from etapa12_enfoques import RUTA_MATRICES_A  # noqa: PLC0415
    return pd.read_parquet(RUTA_MATRICES_A / "test_eval.parquet", columns=columnas)


def evaluar_candidato_con_gate(prefijo, crear_ajuste, preparar, entorno, descripcion, metadatos, estimado):
    """Protocolo de la etapa: 1 semilla contra n12_a_1s; si mejora +0,002 en CV y
    holdout, más semillas (hasta 5) mientras quepan y bagging contra la base.

    Argumentos: prefijo = nombre sin la semilla; crear_ajuste = función(semilla)
        -> ajustar(); preparar = función de contextos; entorno = de main();
        descripcion = texto de la tabla; metadatos = (algoritmo, especificacion,
        busqueda, n_configuraciones, hiperparametros, variables_extra, variables);
        estimado = minutos de la primera semilla.
    Devuelve: lista de filas de la tabla de candidatos (vacía si nada terminó).
    """
    from etapa12_enfoques import fila_tabla, guardar_filas, mejora_sobre  # noqa: PLC0415
    h = harness()
    etapa, verdad = entorno["etapa"], entorno["verdad"]
    referencia = resultado_desde_predicciones(REFERENCIA_1S, verdad)
    base = resultado_desde_predicciones(BASE_LGB, verdad)
    oof_lgb = base["oof"]["probabilidad"]
    algoritmo, especificacion, busqueda, n_configuraciones, hiperparametros, variables_extra, variables = metadatos
    terminadas = h.correr_semillas(prefijo, crear_ajuste, preparar, entorno, objetivo=1, estimado=estimado)
    if not terminadas:
        return []
    s0 = h.resultado_guardado(terminadas[0], verdad)
    filas = [fila_tabla(s0, verdad, f"{descripcion}, 1 semilla", referencia, oof_lgb)]
    h.anotar_metadatos(terminadas[0], algoritmo, f"{especificacion} | 1 semilla", busqueda, n_configuraciones,
                       hiperparametros, variables_extra)
    if variables is not None:
        h.anotar_json("metadatos", terminadas[0], {**h.leer_json("metadatos")[terminadas[0]], "variables": variables})
    delta_cv, delta_holdout, pasa = mejora_sobre(s0, referencia)
    registrar_log(etapa, f"{prefijo} 1 semilla: CV {delta_cv:+.4f}, holdout {delta_holdout:+.4f} frente a "
                         f"{REFERENCIA_1S} -> {'pasa' if pasa else 'no pasa'} a más semillas")
    guardar_filas(f"{TABLAS}_candidatos", filas)
    if pasa:
        terminadas = h.correr_semillas(prefijo, crear_ajuste, preparar, entorno, desde=1)
        bag = h.promediar_si_caben(prefijo, terminadas, verdad)
        if bag is not None:
            nombre, resultado = bag
            filas.append(fila_tabla(resultado, verdad, f"{descripcion}, bagging {len(terminadas)} semillas", base,
                                    oof_lgb))
            h.anotar_metadatos(nombre, algoritmo, f"{especificacion} | bagging {len(terminadas)} semillas", busqueda,
                               n_configuraciones, hiperparametros, variables_extra)
            if variables is not None:
                h.anotar_json("metadatos", nombre, {**h.leer_json("metadatos")[nombre], "variables": variables})
            guardar_filas(f"{TABLAS}_candidatos", filas)
    print(pd.DataFrame(filas).round(4).to_string(index=False))
    return filas


# ---- fold: ensamble de los modelos de la CV en test ------------------------------

def predicciones_test_por_fold(nombre, raiz=RUTA_NIVEL7):
    """Predicciones de test de los 5 modelos de la CV de un candidato, si se
    guardaron en <raiz>/test_por_fold/<nombre>_cv_k.parquet (probabilidad por id).

    Devuelve: DataFrame indexado por id con columnas cv_1..cv_5, o None si falta alguna.
    """
    rutas = [raiz / "test_por_fold" / f"{nombre}_cv_{k}.parquet" for k in range(1, 6)]
    if not all(ruta.exists() for ruta in rutas):
        return None
    return pd.concat([pd.read_parquet(ruta)["probabilidad"].rename(f"cv_{k}") for k, ruta in enumerate(rutas, 1)],
                     axis=1)


def fold(entorno, reloj):
    etapa, verdad = entorno["etapa"], entorno["verdad"]
    por_fold = predicciones_test_por_fold(BASE_LGB)
    ruta_tabla = RUTA_TABLAS / f"{TABLAS}_ensamble_fold.csv"
    if por_fold is None:
        motivo = (f"no hay predicciones de test por fold guardadas de {BASE_LGB}: evaluar_modelo() predice en "
                  f"cv_k solo el fold de validación y los parciales se borran al terminar")
        registrar_log(etapa, f"{motivo}; se omite el ensamble por fold")
        fila = {"candidato": f"{PREFIJO}ensamble_fold", "disponible": False, "motivo": motivo}
        pd.DataFrame([fila]).to_csv(ruta_tabla, index=False)
        return fila
    base = resultado_desde_predicciones(BASE_LGB, verdad)
    umbral = base["umbral"]
    promedio = por_fold.mean(axis=1).reindex(base["prediccion_test"].index)
    clase_fold = (promedio >= umbral).astype(int)
    clase_base = (base["prediccion_test"] >= umbral).astype(int)
    csv_actual = pd.read_csv(RAIZ / "stores" / "submissions" / "nivel7" / f"{BASE_CSV}.csv").set_index("id")["pobre"]
    fila = {"candidato": f"{PREFIJO}ensamble_fold", "disponible": True, "umbral_oof": umbral,
            "tasa_pobres_ensamble_fold": clase_fold.mean(), "tasa_pobres_reajuste_100": clase_base.mean(),
            "tasa_pobres_csv_actual": csv_actual.reindex(clase_fold.index).mean(),
            "concordancia_con_reajuste_100": (clase_fold == clase_base).mean(),
            "concordancia_con_csv_actual": (clase_fold == csv_actual.reindex(clase_fold.index)).mean(),
            "correlacion_probabilidades": float(np.corrcoef(promedio, base["prediccion_test"])[0, 1])}
    pd.DataFrame([fila]).to_csv(ruta_tabla, index=False)
    registrar_log(etapa, f"ensamble por fold: tasa {fila['tasa_pobres_ensamble_fold']:.4f} | concordancia con el "
                         f"CSV {fila['concordancia_con_csv_actual']:.4f}")
    return fila


# ---- pseudo: pseudoetiquetado del test sin fuga ----------------------------------

def seleccionar_pseudo(probabilidad, bajo=UMBRAL_PSEUDO_BAJO, alto=UMBRAL_PSEUDO_ALTO):
    """Etiqueta predicha (0/1) de los hogares con P(pobre) >= alto o <= bajo."""
    confiables = probabilidad[(probabilidad >= alto) | (probabilidad <= bajo)]
    return (confiables >= alto).astype(int)


def agregar_pseudo(x_ent, y_ent, x_pseudo, etiquetas, peso=PESO_PSEUDO):
    """Entrenamiento más los hogares pseudoetiquetados, con la columna peso_fila
    (1 para las filas reales y `peso` para las pseudoetiquetadas).

    Argumentos: x_ent, y_ent = entrenamiento del contexto; x_pseudo = base de los
        hogares candidatos (indexada por id); etiquetas = seleccionar_pseudo().
    Devuelve: (x_nuevo, y_nuevo).
    """
    x_nuevo = pd.concat([x_ent.assign(peso_fila=1.0), x_pseudo.loc[etiquetas.index].assign(peso_fila=float(peso))])
    y_nuevo = np.concatenate([np.asarray(y_ent), etiquetas.to_numpy()])
    return x_nuevo, y_nuevo


def probabilidades_test_por_contexto(contexto, preparar, etapa):
    """P(pobre) de los hogares del test de Kaggle según el LightGBM n6 (1 semilla)
    ajustado solo con el entrenamiento del contexto (nunca con su evaluación);
    caché en etapa15/pseudo/<contexto>.parquet."""
    ruta = CARPETA / "pseudo" / f"{contexto}.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)["probabilidad"]
    inicio = time.time()
    x_ent, y_ent, _ = preparar(contexto)
    modelo = ajustar_lightgbm(x_ent, y_ent, PARAMETROS_LGB, SEMILLA)
    del x_ent
    gc.collect()
    x_test = matriz_test_kaggle()
    probabilidad = pd.Series(modelo.predict(x_test), index=x_test.index, name="probabilidad")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    guardar_parquet_atomico(probabilidad.to_frame(), ruta)
    registrar_log(etapa, f"modelo base de {contexto} sobre el test de Kaggle", inicio)
    return probabilidad


def pseudo(entorno, reloj):
    etapa = entorno["etapa"]
    preparar_a = preparador_a()
    etiquetas, filas = {}, []
    for contexto in CONTEXTOS:
        probabilidad = probabilidades_test_por_contexto(contexto, preparar_a, etapa)
        etiquetas[contexto] = seleccionar_pseudo(probabilidad)
        filas.append({"contexto": contexto, "n_test": len(probabilidad), "n_pseudo": len(etiquetas[contexto]),
                      "n_pseudo_pobres": int(etiquetas[contexto].sum()),
                      "n_pseudo_no_pobres": int((etiquetas[contexto] == 0).sum()),
                      "share_test": len(etiquetas[contexto]) / len(probabilidad), "peso": PESO_PSEUDO})
        registrar_log(etapa, f"pseudo {contexto}: {len(etiquetas[contexto])} hogares confiables "
                             f"({int(etiquetas[contexto].sum())} pobres) de {len(probabilidad)}; quedan "
                             f"{reloj.restante():.0f} min")
        gc.collect()
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / f"{TABLAS}_pseudo.csv", index=False)
    print(tabla.round(4).to_string(index=False))

    def preparar_pseudo(contexto):
        x_ent, y_ent, x_eval = preparar_a(contexto)
        x_pseudo = matriz_test_kaggle().loc[etiquetas[contexto].index]
        x_ent, y_ent = agregar_pseudo(x_ent, y_ent, x_pseudo, etiquetas[contexto])
        del x_pseudo
        gc.collect()
        return x_ent, y_ent, x_eval

    n_medio = int(round(tabla["n_pseudo"].mean()))
    metadatos = ("Boosting",
                 f"LightGBM n6 sobre A + pseudoetiquetas del test de Kaggle (P >= {UMBRAL_PSEUDO_ALTO} o <= "
                 f"{UMBRAL_PSEUDO_BAJO} según el modelo del contexto, ~{n_medio} hogares por contexto, peso "
                 f"{PESO_PSEUDO})",
                 "hiperparámetros fijos (n6) y umbrales de confianza fijos; pseudoetiquetas del modelo de cada "
                 "contexto (sin holdout ni fold de validación)", 1, texto_hiperparametros(PARAMETROS_LGB), (), None)
    return evaluar_candidato_con_gate(
        f"{PREFIJO}pseudo", lambda semilla: crear_ajuste_lightgbm(PARAMETROS_LGB, (semilla,), columna_peso="peso_fila"),
        preparar_pseudo, entorno, f"A + pseudoetiquetas del test (~{n_medio} hogares, peso {PESO_PSEUDO}), LightGBM n6",
        metadatos, MINUTOS_SEMILLA_PSEUDO)


# ---- fex: factores de expansión como pesos ---------------------------------------

def metricas_ponderadas(verdad, prediccion, pesos):
    """F1, precisión y recall con cada hogar contando su peso."""
    verdad, prediccion, pesos = (np.asarray(a, dtype=float) for a in (verdad, prediccion, pesos))
    vp = pesos[(verdad == 1) & (prediccion == 1)].sum()
    fp = pesos[(verdad == 0) & (prediccion == 1)].sum()
    fn = pesos[(verdad == 1) & (prediccion == 0)].sum()
    return {"f1_fex": f1_ponderado(verdad, prediccion, pesos),
            "precision_fex": vp / (vp + fp) if vp + fp else 0.0, "recall_fex": vp / (vp + fn) if vp + fn else 0.0}


def fex(entorno, reloj):
    etapa, verdad = entorno["etapa"], entorno["verdad"]
    preparar_a = preparador_a()
    fex_c = cargar_hogares()["Fex_c"].astype(float)

    def preparar_fex(contexto):
        x_ent, y_ent, x_eval = preparar_a(contexto)
        peso = fex_c.loc[x_ent.index]
        return x_ent.assign(peso_fila=(peso / peso.mean()).to_numpy()), y_ent, x_eval

    metadatos = ("Boosting", "LightGBM n6 sobre A con peso de fila = Fex_c / media(Fex_c) del entrenamiento del "
                 "contexto (el factor de expansión no entra como variable)",
                 "hiperparámetros fijos (n6)", 1, texto_hiperparametros(PARAMETROS_LGB), (), None)
    filas = evaluar_candidato_con_gate(
        f"{PREFIJO}fex", lambda semilla: crear_ajuste_lightgbm(PARAMETROS_LGB, (semilla,), columna_peso="peso_fila"),
        preparar_fex, entorno, "A con Fex_c como peso de fila, LightGBM n6", metadatos, MINUTOS_SEMILLA_FEX)

    # F1 sin ponderar (el de Kaggle) y ponderado por Fex_c en holdout, también para las bases
    candidatos = [f["candidato"] for f in filas] + [REFERENCIA_1S, BASE_LGB, BASE_CSV]
    comparacion = []
    for nombre in candidatos:
        resultado = resultado_desde_predicciones(nombre, verdad)
        prob = resultado["prediccion_holdout"]
        clase = (prob >= resultado["umbral"]).astype(int)
        y = verdad.loc[prob.index].astype(int)
        comparacion.append({"candidato": nombre, "umbral": resultado["umbral"], "f1_holdout": resultado["f1_holdout"],
                            "precision_holdout": resultado["precision_holdout"],
                            "recall_holdout": resultado["recall_holdout"],
                            **metricas_ponderadas(y, clase, fex_c.loc[prob.index])})
    tabla = pd.DataFrame(comparacion)
    tabla.to_csv(RUTA_TABLAS / f"{TABLAS}_fex.csv", index=False)
    print(tabla.round(4).to_string(index=False))
    return filas


# ---- ebm: Explainable Boosting Machine --------------------------------------------

def importancia_ebm(modelo):
    """Importancia de los términos de una variable (sin interacciones), por variable."""
    importancias = modelo.term_importances()
    filas = {modelo.term_names_[i]: float(importancias[i]) for i, terminos in enumerate(modelo.term_features_)
             if len(terminos) == 1}
    return pd.Series(filas, dtype=float)


def crear_ajuste_ebm(parametros, estado, carpeta):
    """ajustar() de un EBM para evaluar_modelo(); guarda el modelo de los contextos
    holdout y test en carpeta/ebm_<contexto>.joblib para dibujar sus curvas.

    Argumentos: parametros = del ExplainableBoostingClassifier; estado = dict con
        "contexto" (lo escribe el preparador); carpeta = destino de los modelos.
    Devuelve: función(x_ent, y_ent, x_eval) -> (probabilidad, importancia por variable).
    """
    def ajustar(x_ent, y_ent, x_eval):
        import joblib  # noqa: PLC0415
        from interpret.glassbox import ExplainableBoostingClassifier  # noqa: PLC0415
        modelo = ExplainableBoostingClassifier(**parametros, n_jobs=N_HILOS, random_state=SEMILLA)
        modelo.fit(x_ent, np.asarray(y_ent).astype(int))
        probabilidad = modelo.predict_proba(x_eval)[:, 1]
        if estado.get("contexto") in ("holdout", "test"):
            carpeta.mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo, carpeta / f"ebm_{estado['contexto']}.joblib")
        return probabilidad, importancia_ebm(modelo)

    return ajustar


def crear_preparador_columnas(columnas, estado):
    """preparar() de la base de A restringida a unas columnas (lee solo esas del
    parquet) que además anota el contexto en curso en `estado`."""
    from etapa12_enfoques import RUTA_MATRICES_A  # noqa: PLC0415

    def preparar(contexto):
        estado["contexto"] = contexto
        x_ent = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_ent.parquet", columns=list(columnas) + ["pobre"])
        y_ent = x_ent.pop("pobre").to_numpy()
        x_eval = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_eval.parquet", columns=list(columnas))
        return x_ent, y_ent, x_eval

    return preparar


def graficar_ebm(ruta_modelo, n_curvas=N_CURVAS_EBM, carpeta_figuras=RUTA_FIGURAS):
    """Curvas (aporte en log-odds por tramo) de las n variables más importantes del
    EBM y gráfico de importancia de términos; tabla etapa15_ebm_importancia.csv."""
    import joblib  # noqa: PLC0415
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    carpeta_figuras.mkdir(parents=True, exist_ok=True)
    modelo = joblib.load(ruta_modelo)
    explicacion = modelo.explain_global()
    importancias = modelo.term_importances()
    tabla = pd.DataFrame({"termino": modelo.term_names_, "importancia": importancias,
                          "tipo": ["variable" if len(t) == 1 else "interacción" for t in modelo.term_features_]})
    tabla = tabla.sort_values("importancia", ascending=False).reset_index(drop=True)
    tabla["orden"] = tabla.index + 1
    tabla.to_csv(RUTA_TABLAS / f"{TABLAS}_ebm_importancia.csv", index=False)
    principales = [i for i in np.argsort(-importancias) if len(modelo.term_features_[i]) == 1][:n_curvas]
    for orden, indice in enumerate(principales, 1):
        datos = explicacion.data(indice)
        nombres, puntajes = np.asarray(datos["names"]), np.asarray(datos["scores"], dtype=float)
        figura, eje = plt.subplots(figsize=(7, 4.2))
        if len(nombres) == len(puntajes) + 1:  # variable continua: tramos
            bordes = nombres.astype(float)
            eje.stairs(puntajes, bordes, color=AZUL, linewidth=2, baseline=None)
            if np.abs(bordes).max() > 1e5:  # pesos: en millones
                eje.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f} M"))
            eje.fill_between(bordes[:-1], np.asarray(datos["lower_bounds"], float), np.asarray(datos["upper_bounds"], float),
                             step="post", color=AZUL, alpha=0.15, linewidth=0)
            densidad = datos.get("density", {})
            if densidad:
                eje2 = eje.twinx()
                bordes_d = np.asarray(densidad["names"], float)
                conteos = np.asarray(densidad["scores"], float)
                eje2.bar(bordes_d[:-1], conteos, width=np.diff(bordes_d), align="edge", color=GRIS,
                         alpha=0.25, linewidth=0)
                eje2.set_yticks([])
                eje2.set_ylabel("hogares por tramo (gris)", color=TINTA_SUAVE, fontsize=8)
                # El eje x cubre el 99 % central de los hogares (las colas raras no se dibujan)
                acumulado = np.cumsum(conteos) / conteos.sum()
                eje.set_xlim(bordes_d[max(int(np.searchsorted(acumulado, 0.005)), 0)],
                             bordes_d[min(int(np.searchsorted(acumulado, 0.995)) + 1, len(bordes_d) - 1)])
        else:  # categórica o dummy: barras
            eje.bar([str(c) for c in nombres], puntajes, color=AZUL, width=0.6)
            eje.axhline(0, color=GRIS, linewidth=0.8)
        variable = modelo.term_names_[indice]
        eje.set_xlabel(variable, color=TINTA)
        eje.set_ylabel("Aporte al log-odds de ser pobre", color=TINTA)
        eje.axhline(0, color=GRIS, linewidth=0.8, linestyle="--")
        figura.text(0.01, 0.97, f"{orden}. {variable}", fontsize=12, fontweight="bold", color=TINTA, va="top")
        figura.text(0.01, 0.91, f"Importancia {importancias[indice]:.3f} (media de |aporte| en el entrenamiento). "
                    f"Sombra: intervalo entre bolsas", fontsize=8.5, color=TINTA_SUAVE, va="top")
        eje.grid(axis="y", color="#e8eaed", linewidth=0.8)
        eje.spines[["top", "right"]].set_visible(False)
        figura.tight_layout(rect=(0, 0, 1, 0.9))
        for extension in ("png", "pdf"):
            figura.savefig(carpeta_figuras / f"ebm_{orden:02d}_{variable}.{extension}", dpi=300, facecolor="white")
        plt.close(figura)
    # Importancia de los 15 términos principales (variables e interacciones)
    top = tabla.head(15).iloc[::-1]
    figura, eje = plt.subplots(figsize=(8, 5.5))
    eje.barh(top["termino"], top["importancia"], color=[AZUL if t == "variable" else NARANJA for t in top["tipo"]])
    eje.set_xlabel("Importancia (media de |aporte| al log-odds)", color=TINTA)
    figura.text(0.01, 0.97, "EBM: términos más importantes", fontsize=13, fontweight="bold", color=TINTA, va="top")
    figura.text(0.01, 0.925, "Azul: variables; naranja: interacciones elegidas por el EBM", fontsize=9,
                color=TINTA_SUAVE, va="top")
    eje.spines[["top", "right"]].set_visible(False)
    figura.tight_layout(rect=(0, 0, 1, 0.9))
    for extension in ("png", "pdf"):
        figura.savefig(carpeta_figuras / f"ebm_importancia.{extension}", dpi=300, facecolor="white")
    plt.close(figura)
    return tabla


def ebm(entorno, reloj):
    from etapa12_enfoques import variables_top_gain  # noqa: PLC0415
    etapa = entorno["etapa"]
    columnas = variables_top_gain(N_VARIABLES_EBM)
    estado = {}
    carpeta = CARPETA / "ebm"
    especificacion = (f"EBM (interpret ExplainableBoostingClassifier) sobre las {N_VARIABLES_EBM} variables de más "
                      f"gain de A, {PARAMETROS_EBM['interactions']} interacciones elegidas por el EBM, "
                      f"{PARAMETROS_EBM['outer_bags']} bolsas, early stopping")
    metadatos = ("Boosting", especificacion, "hiperparámetros fijos; interacciones elegidas por el EBM (FAST)", 1,
                 texto_hiperparametros(PARAMETROS_EBM), (), columnas)
    filas = evaluar_candidato_con_gate(
        f"{PREFIJO}ebm_top{N_VARIABLES_EBM}", lambda semilla: crear_ajuste_ebm(PARAMETROS_EBM, estado, carpeta),
        crear_preparador_columnas(columnas, estado), entorno, f"EBM sobre las {N_VARIABLES_EBM} variables top de A",
        metadatos, MINUTOS_EBM_ESTIMADOS)
    ruta_modelo = next((carpeta / f"ebm_{c}.joblib" for c in ("test", "holdout") if (carpeta / f"ebm_{c}.joblib").exists()),
                       None)
    if ruta_modelo is None:
        registrar_log(etapa, "sin modelo EBM guardado: no hay curvas")
        return filas
    tabla = graficar_ebm(ruta_modelo, carpeta_figuras=RUTA_FIGURAS / "prueba" if PRUEBA else RUTA_FIGURAS)
    registrar_log(etapa, f"curvas EBM desde {ruta_modelo.name}: {', '.join(tabla.head(5)['termino'])}")
    return filas


# ---- politica: subgrupos, pesos de bienestar y curva de cobertura ----------------

def ruta_rscript():
    """Rscript del PATH o de la instalación estándar de Windows."""
    encontrado = shutil.which("Rscript")
    if encontrado:
        return encontrado
    candidatos = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"))
    return str(candidatos[-1]) if candidatos else None


def predicciones_r(nombre):
    """Probabilidad de holdout y umbral de un candidato de R (stores/modelos/<nombre>.rds),
    exportados una vez con Rscript a etapa15/r/<nombre>_holdout.csv."""
    ruta = CARPETA / "r" / f"{nombre}_holdout.csv"
    if not ruta.exists():
        rscript = ruta_rscript()
        if rscript is None:
            raise RuntimeError("no se encontró Rscript para exportar las predicciones de R")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        rds = (RAIZ / "stores" / "modelos" / f"{nombre}.rds").as_posix()
        codigo = (f"r <- readRDS('{rds}'); h <- as.data.frame(r$prediccion_holdout)[, c('id', 'probabilidad')]; "
                  f"h$umbral <- r$umbral; write.csv(h, '{ruta.as_posix()}', row.names = FALSE)")
        subprocess.run([rscript, "-e", codigo], check=True, capture_output=True, text=True)
    tabla = pd.read_csv(ruta).set_index("id")
    return tabla["probabilidad"], float(tabla["umbral"].iloc[0])


def ajustar_encuesta_corta(variables_objetivo, particion, verdad):
    """Logit de la encuesta corta (etapa 3b) con las variables dadas: umbral con la
    OOF de los 5 folds y probabilidad en holdout con el ajuste sobre el 80 %.

    Devuelve: (probabilidad de holdout indexada por id, umbral, F1 OOF).
    """
    from etapa3_encuesta_corta import BLOQUES_CANDIDATOS  # noqa: PLC0415
    from funciones.seleccion import ajustar_logit, evaluar_mascara, mapear_columnas, preparar_division  # noqa: PLC0415
    from funciones.variables import construir_grupos  # noqa: PLC0415
    hogar = cargar_hogares()
    bloques = construir_grupos(cargar_personas(), hogar)
    variables = pd.concat([bloques[b] for b in BLOQUES_CANDIDATOS], axis=1)
    matriz = preparar_matriz(variables).astype(np.float64)
    mapa = mapear_columnas(variables, matriz)
    nombres = list(variables.columns)
    faltan = set(variables_objetivo) - set(nombres)
    assert not faltan, f"variables de la encuesta corta que no existen: {faltan}"
    mascara = np.isin(nombres, variables_objetivo)
    tabla = particion.set_index("id")
    divisiones = [preparar_division(matriz, tabla.index[tabla["fold_id"].isin([g for g in range(1, 6) if g != k])],
                                    tabla.index[tabla["fold_id"] == k], verdad) for k in range(1, 6)]
    f1_oof, umbral = evaluar_mascara(mascara, divisiones, mapa)
    ids_80 = tabla.index[tabla["fold_id"].notna()]
    ids_holdout = tabla.index[(tabla["base"] == "train") & tabla["fold_id"].isna()]
    division = preparar_division(matriz, ids_80, ids_holdout, verdad)
    columnas = sorted(c for v in np.flatnonzero(mascara) for c in mapa[v])
    probabilidad = pd.Series(ajustar_logit(division, columnas), index=ids_holdout, name="probabilidad")
    return probabilidad, umbral, f1_oof


def graficar_errores_subgrupo(tabla, nombre_modelo):
    """Barras de subcobertura y filtración por grupo (panel izquierdo) y por dominio
    (panel derecho), con la línea del total y los grupos marcados en naranja."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    total = tabla[tabla["dimension"] == "total"].iloc[0]
    izquierda = tabla[tabla["dimension"].isin(DIMENSIONES_BARRAS)].copy()
    izquierda["etiqueta"] = [f"{ETIQUETAS_DIMENSION[d]}: {g}" for d, g in zip(izquierda["dimension"], izquierda["grupo"])]
    derecha = tabla[tabla["dimension"] == "dominio"].sort_values("subcobertura", ascending=False)
    figura, (eje_a, eje_b) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.15]})
    for eje, datos, etiquetas in ((eje_a, izquierda, izquierda["etiqueta"]), (eje_b, derecha, derecha["grupo"])):
        posiciones = np.arange(len(datos))
        colores = [NARANJA if m else AZUL for m in datos["marca_subcobertura"]]
        eje.barh(posiciones - 0.2, datos["subcobertura"], height=0.4, color=colores, label="Subcobertura (pobres excluidos)")
        eje.barh(posiciones + 0.2, datos["filtracion"], height=0.4, color=GRIS, label="Filtración (no pobres incluidos)")
        eje.axvline(total["subcobertura"], color=TINTA_SUAVE, linewidth=1, linestyle="--")
        eje.axvline(FACTOR_MARCA * total["subcobertura"], color=NARANJA, linewidth=1, linestyle=":")
        eje.set_yticks(posiciones)
        eje.set_yticklabels(etiquetas, fontsize=8.5)
        eje.invert_yaxis()  # el primer grupo arriba
        eje.set_xlabel("Proporción", color=TINTA)
        eje.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        eje.grid(axis="x", color="#e8eaed", linewidth=0.8)
        eje.spines[["top", "right"]].set_visible(False)
    manijas, textos = eje_a.get_legend_handles_labels()
    figura.legend(manijas, textos, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figura.text(0.01, 0.975, f"¿A quién se equivoca el modelo? Errores de {nombre_modelo} por subgrupo (holdout)",
                fontsize=13, fontweight="bold", color=TINTA, va="top")
    figura.text(0.01, 0.94, f"Línea punteada gris: subcobertura total ({total['subcobertura']:.1%}); naranja: grupos "
                f"con subcobertura > {FACTOR_MARCA:g} veces el total; línea naranja: ese límite", fontsize=9.5,
                color=TINTA_SUAVE, va="top")
    figura.tight_layout(rect=(0, 0.04, 1, 0.92))
    for extension in ("png", "pdf"):
        figura.savefig(RUTA_FIGURAS / f"errores_subgrupo_{nombre_modelo}.{extension}", dpi=300, facecolor="white")
    plt.close(figura)


def graficar_curva_cobertura(curvas, puntos, tasa_pobreza):
    """Curvas de % de pobres cubiertos vs. % de hogares atendidos, con el punto de
    operación (umbral OOF) de cada modelo."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    colores = {BASE_CSV: AZUL, "n3_logit": NARANJA, ENCUESTA_CORTA: "#34a853"}
    etiquetas = {BASE_CSV: "n13_mezcla_final (LightGBM + redes)", "n3_logit": "n3_logit (48 variables)",
                 ENCUESTA_CORTA: "Encuesta corta: Logit de 8 variables"}
    figura, eje = plt.subplots(figsize=(8.5, 6))
    eje.plot([0, 1], [0, 1], color=GRIS, linewidth=1, linestyle="--", label="Al azar")
    eje.plot([0, tasa_pobreza, 1], [0, 1, 1], color=TINTA_SUAVE, linewidth=1, linestyle=":", label="Focalización perfecta")
    for orden, (nombre, curva) in enumerate(curvas.items()):
        eje.plot(curva["share_atendida"], curva["share_pobres_cubiertos"], color=colores[nombre], linewidth=2,
                 label=etiquetas[nombre])
        atendida, cubiertos = puntos[nombre]
        eje.plot(atendida, cubiertos, marker="o", markersize=7, color=colores[nombre], zorder=5)
        eje.annotate(f"umbral OOF: {atendida:.0%} atendidos, {cubiertos:.0%} cubiertos", (atendida, cubiertos),
                     textcoords="offset points", xytext=(16, -10 - 18 * orden), fontsize=8.5, color=colores[nombre],
                     bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"},
                     arrowprops={"arrowstyle": "-", "color": colores[nombre], "linewidth": 0.8})
    eje.set_xlabel("Hogares atendidos (% del holdout, de mayor a menor probabilidad)", color=TINTA)
    eje.set_ylabel("Pobres cubiertos (% de los pobres del holdout)", color=TINTA)
    eje.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    eje.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1.02)
    figura.text(0.01, 0.97, "¿Cuántos pobres cubre cada modelo al ampliar el programa?", fontsize=13,
                fontweight="bold", color=TINTA, va="top")
    figura.text(0.01, 0.925, f"Curva de cobertura en holdout (20 %); tasa de pobreza {tasa_pobreza:.1%}.\n"
                "Al bajar el umbral se atiende a más hogares a cambio de más filtración", fontsize=9.5,
                color=TINTA_SUAVE, va="top", linespacing=1.4)
    eje.grid(color="#e8eaed", linewidth=0.8)
    eje.spines[["top", "right"]].set_visible(False)
    eje.legend(loc="lower right", frameon=False, fontsize=9)
    figura.tight_layout(rect=(0, 0, 1, 0.87))
    for extension in ("png", "pdf"):
        figura.savefig(RUTA_FIGURAS / f"curva_cobertura.{extension}", dpi=300, facecolor="white")
    plt.close(figura)


def politica(entorno, reloj):
    etapa, particion, verdad = entorno["etapa"], entorno["particion"], entorno["verdad"]
    n13 = resultado_desde_predicciones(BASE_CSV, verdad)
    prob_n13 = n13["prediccion_holdout"]
    ids = prob_n13.index
    y = verdad.loc[ids].astype(int)
    clase_n13 = (prob_n13 >= n13["umbral"]).astype(int)

    # 5. Errores por subgrupo
    hogar = cargar_hogares()
    personas = cargar_personas()
    grupos = grupos_de_hogares(hogar.loc[ids], personas[personas["id"].isin(ids)])
    errores = errores_por_subgrupo(y, clase_n13, grupos, FACTOR_MARCA)
    errores.to_csv(RUTA_TABLAS / f"{TABLAS}_errores_subgrupo.csv", index=False)
    graficar_errores_subgrupo(errores, BASE_CSV)
    marcados = errores[errores["marca_subcobertura"]]
    registrar_log(etapa, f"errores por subgrupo: {len(errores) - 1} grupos, {len(marcados)} con subcobertura > "
                         f"{FACTOR_MARCA} x total ({', '.join(marcados['grupo'].astype(str))})")
    del personas
    gc.collect()

    # 6. Pesos de bienestar: n13, los Logit de R y la encuesta corta de 8 variables
    razon = np.exp(cargar_ingreso_hogar()).loc[ids]
    modelos = {BASE_CSV: (prob_n13, n13["umbral"], f"mezcla final (LightGBM + redes), umbral OOF {n13['umbral']:.4f}")}
    for nombre, descripcion in MODELOS_R.items():
        prob_r, umbral_r = predicciones_r(nombre)
        modelos[nombre] = (prob_r.loc[ids], umbral_r, f"{descripcion}, umbral OOF {umbral_r:.4f}")
    prob_corta, umbral_corta, f1_oof_corta = ajustar_encuesta_corta(VARIABLES_ENCUESTA_CORTA, particion, verdad)
    modelos[ENCUESTA_CORTA] = (prob_corta.loc[ids], umbral_corta,
                               f"Logit de 8 variables de la frontera (F1 OOF {f1_oof_corta:.4f}), umbral OOF "
                               f"{umbral_corta:.4f}")
    registrar_log(etapa, f"encuesta corta de 8 variables reajustada: F1 OOF {f1_oof_corta:.4f}")
    filas = []
    for nombre, (prob, umbral, descripcion) in modelos.items():
        clase = (prob >= umbral).astype(int)
        filas.append({"candidato": nombre, "descripcion": descripcion, "umbral": umbral,
                      **calcular_metricas(y, clase), **pesos_bienestar(y, clase, razon)})
    bienestar = pd.DataFrame(filas)
    bienestar.to_csv(RUTA_TABLAS / f"{TABLAS}_pesos_bienestar.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(bienestar.drop(columns=["descripcion"]).round(4).to_string(index=False))

    # 7. Curva de cobertura: n13, n3_logit y la encuesta corta
    curvas, puntos, tabla_curvas = {}, {}, []
    for nombre in (BASE_CSV, "n3_logit", ENCUESTA_CORTA):
        prob, umbral, _ = modelos[nombre]
        curvas[nombre] = curva_cobertura(y, prob)
        puntos[nombre] = punto_operacion(y, prob, umbral)
        tabla_curvas.append(curvas[nombre].assign(candidato=nombre))
    tabla_curvas = pd.concat(tabla_curvas)[["candidato", "share_atendida", "share_pobres_cubiertos"]]
    tabla_curvas.to_csv(RUTA_TABLAS / f"{TABLAS}_curva_cobertura.csv", index=False)
    pd.DataFrame([{"candidato": n, "umbral": modelos[n][1], "share_atendida": a, "share_pobres_cubiertos": c}
                  for n, (a, c) in puntos.items()]).to_csv(RUTA_TABLAS / f"{TABLAS}_curva_cobertura_puntos.csv",
                                                           index=False)
    graficar_curva_cobertura(curvas, puntos, float(y.mean()))
    registrar_log(etapa, "curva de cobertura y figuras guardadas")
    return bienestar


def cierre(entorno, reloj):
    return harness().cierre(entorno, reloj)


# ---- Orquestación -------------------------------------------------------------

EXPERIMENTOS = {"fold": fold, "pseudo": pseudo, "fex": fex, "ebm": ebm, "politica": politica, "cierre": cierre}


def main(experimento, minutos, sin_test=False, prueba=False):
    from etapa12_enfoques import correr_seguro  # noqa: PLC0415
    global CON_TEST
    CON_TEST = not sin_test
    if prueba:
        configurar_prueba()
    CARPETA.mkdir(parents=True, exist_ok=True)
    etapa = f"etapa15_{experimento}" + ("_prueba" if prueba else "")
    # El reloj se guarda en disco: un relanzamiento (salida por memoria) sigue contando
    reloj = Presupuesto(minutos, CARPETA / f"inicio_{experimento}.txt")
    registrar_log(etapa, f"inicio ({minutos:.0f} min{', sin test' if sin_test else ''})")
    bitacora.LIMITE_PROCESO_GB = LIMITE_PROCESO_GB
    iniciar_vigilante_memoria(etapa)
    desactivar = threading.Event()
    iniciar_corte_duro(etapa, reloj, desactivar)
    particion = cargar_particion()
    entorno = {"particion": particion, "verdad": particion.set_index("id")["pobre"], "reloj": reloj, "etapa": etapa}
    salida = correr_seguro(etapa, experimento, lambda: EXPERIMENTOS[experimento](entorno, reloj))
    desactivar.set()
    registrar_log(etapa, f"fin ({reloj.transcurrido():.1f} min){'' if salida is not None else ' con falla'}")


def todo(minutos_total=MINUTOS_TOTAL, sin_test=False, prueba=False):
    """Los seis experimentos en orden, cada uno en un subproceso con su tope (recortado
    al tiempo global que quede), relanzando hasta RELANZAMIENTOS veces el que salga
    por memoria; al terminar escribe etapa15/fin.txt."""
    if prueba:
        configurar_prueba()
    CARPETA.mkdir(parents=True, exist_ok=True)
    reloj = Presupuesto(minutos_total, CARPETA / "inicio_todo.txt")
    registrar_log("etapa15_todo", f"inicio ({minutos_total} min en total)")
    for experimento in EXPERIMENTOS:
        minutos = min(MINUTOS[experimento], reloj.restante())
        if minutos < 1:
            registrar_log("etapa15_todo", f"{experimento}: sin tiempo global, se omite")
            continue
        for intento in range(1, RELANZAMIENTOS + 1):
            comando = [sys.executable, str(Path(__file__).resolve()), experimento, "--minutos", f"{minutos:.1f}"]
            comando += (["--sin-test"] if sin_test else []) + (["--prueba"] if prueba else [])
            with open(RUTA_LOGS / f"etapa15_{experimento}_salida.txt", "a", encoding="utf-8") as salida, \
                    open(RUTA_LOGS / f"etapa15_{experimento}_error.txt", "a", encoding="utf-8") as error:
                codigo = subprocess.call(comando, stdout=salida, stderr=error, cwd=RAIZ)
            registrar_log("etapa15_todo", f"{experimento} intento {intento}: código {codigo}")
            if codigo != CODIGO_SALIDA_MEMORIA:
                break
            time.sleep(30)  # deja que el sistema libere memoria antes de retomar
    registrar_log("etapa15_todo", f"fin ({reloj.transcurrido():.1f} min)")
    (CARPETA / "fin.txt").write_text(datetime.now().isoformat(), encoding="utf-8")


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("experimento", choices=list(MINUTOS) + ["todo"])
    analizador.add_argument("--minutos", type=float, default=None, help="tope de minutos (por defecto el de la etapa)")
    analizador.add_argument("--sin-test", action="store_true",
                            help="cierra con OOF y holdout los candidatos que ya tienen parciales y no entrena ninguno")
    analizador.add_argument("--prueba", action="store_true", help="miniatura para probar el flujo")
    args = analizador.parse_args()
    if args.experimento == "todo":
        todo(args.minutos or MINUTOS_TOTAL, args.sin_test, args.prueba)
    else:
        main(args.experimento, args.minutos or MINUTOS[args.experimento], args.sin_test, args.prueba)
