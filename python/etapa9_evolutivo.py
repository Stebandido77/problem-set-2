"""Etapa 9: bagging de árboles evolutivos (algoritmo genético con cruce de
subárboles) frente a Random Forest, CART y la mezcla con LightGBM.

Base de comparación: n7_a_agregados_bag5 (F1 CV 0,7748, holdout 0,7806), que
sigue siendo el mejor por holdout al cerrar la etapa 8. Mismos folds, holdout,
umbral OOF y OOF anidada de los auxiliares de ingreso.

  0. Variables: en cada contexto, las 60 de mayor gain de un LightGBM n6 (una
     semilla) ajustado solo con el entrenamiento del contexto sobre la base de A
     (incluye los auxiliares laboral, total y fuentes). GA, Random Forest y CART
     usan exactamente esas columnas.
  1. Métrica: la aptitud de cada árbol es el F1 de la clase pobre con el umbral
     óptimo de buscar_umbral_f1(), sobre las filas out-of-bag de su bolsa, menos
     una penalización leve por profundidad (funciones/evolutivo.py; verificada
     en tests/test_etapa9.py contra el pipeline de Python y de R).
  2. GA-bagging: B bolsas estratificadas de 25 mil hogares del entrenamiento del
     contexto; en cada una un genético sobre árboles de profundidad <= 7; la
     predicción promedia los 5 mejores árboles de cada bolsa. Primero una
     configuración chica en cv_1 para medir tiempos y luego la escala que cabe
     en el presupuesto (decisión guardada en decision_escala.json).
  3. Comparación: Random Forest y CART con la misma base de variables
     (hiperparámetros por grilla con una validación interna del entrenamiento del
     contexto) y la mezcla LightGBM (base) + GA-bagging con el peso elegido solo
     en OOF. Correlación de las probabilidades OOF del GA con las de LightGBM.
  4. Tabla etapa9_candidatos.csv; CSV de Kaggle solo si un candidato sube al
     menos 0,002 en CV y en holdout sobre la base (máximo uno).

Reanudable (variables, matrices, cada bolsa del genético y cada fold se guardan
al terminar) y con el vigilante de memoria. Un solo proceso pesado; tope de 3 horas.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa9_evolutivo.py
"""

import gc
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from configuracion import N_HILOS, PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa4_leaderboard import (RUTA_LEADERBOARD_EXTRA, anexar_filas_extra, armar_fila, formatear_como_r,
                                texto_hiperparametros)
from etapa7_exploracion import AUXILIARES_BASE, PESOS_MEZCLA, RUTA_ETAPA7, correr_seguro, fila_tabla
from funciones.anidado import definir_contextos, guardar_parquet_atomico
from funciones.bases import armar_matriz, cargar_insumos
from funciones.bitacora import Presupuesto, iniciar_vigilante_memoria, registrar_log
from funciones.evolutivo import (GeneticoArboles, aptitud_f1, calcular_cortes, discretizar, muestra_estratificada,
                                 predecir_arboles)
from funciones.metricas import buscar_umbral_f1
from funciones.modelado import (ajustar_lightgbm, borrar_parciales, crear_preparador, evaluar_modelo,
                                guardar_predicciones, resultado_desde_predicciones)
from funciones.submission import guardar_submission

BASE = "n7_a_agregados_bag5"
MINUTOS_TOTALES = 180
RESERVA_MIN = 15                      # cierre, mezcla, tabla y margen
RUTA_ETAPA9 = RUTA_NIVEL7 / "etapa9"
N_VARIABLES = 60
MEJORA_MINIMA = 0.002

# Genético: configuración chica para medir y techo de la escala
N_CORTES = 32
TAMANO_BOLSA = 25_000
CONFIG_GA = {"profundidad_maxima": 7, "prob_cruce": 0.8, "prob_mutacion": 0.6, "torneo": 3, "elite": 2,
             "penalizacion": 0.0005, "fuerza_suavizado": 20, "mejores": 5}
CHICA = {"bolsas": 10, "poblacion": 30, "generaciones": 10}
POBLACION_ESCALA = 150
GENERACIONES_RANGO = (30, 200)
BOLSAS_MAXIMO = 20

# Random Forest y CART: grillas con validación interna (25% del entrenamiento del contexto)
GRILLA_CART = [{"max_depth": d, "min_samples_leaf": m} for d in (4, 5, 6, 7, 8) for m in (25, 50, 100, 200, 400)]
GRILLA_RF = [{"max_features": f, "min_samples_leaf": m} for f in ("sqrt", 0.33) for m in (5, 20)]
ARBOLES_RF_BUSQUEDA, ARBOLES_RF = 150, 300
FRACCION_VALIDACION = 0.25

NOMBRES = {"ga": "n9_ga_bagging", "rf": "n9_rf_top60", "cart": "n9_cart_top60", "mezcla": "n9_mezcla_lgb_ga"}


# ---- Utilidades ---------------------------------------------------------------

def escribir_json(datos, ruta):
    """JSON con escritura atómica."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(datos, indent=1), encoding="utf-8")
    temporal.replace(ruta)


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def anotar_tiempo(nombre, minutos):
    """Minutos de un candidato en etapa9/tiempos.json (para reportar al retomar)."""
    ruta = RUTA_ETAPA9 / "tiempos.json"
    tiempos = leer_json(ruta) or {}
    tiempos[nombre] = minutos
    escribir_json(tiempos, ruta)


def evaluar_o_cargar(nombre, evaluar, verdad):
    """Resultado desde las predicciones si ya terminó; si no, evaluar() y anota el tiempo."""
    if (RUTA_NIVEL7 / f"predicciones_{nombre}.parquet").exists():
        registrar_log(nombre, "desde predicciones guardadas")
        resultado = resultado_desde_predicciones(nombre, verdad)
        resultado["tiempo_min"] = (leer_json(RUTA_ETAPA9 / "tiempos.json") or {}).get(nombre, np.nan)
        return resultado
    resultado = evaluar()
    anotar_tiempo(nombre, resultado["tiempo_min"])
    return resultado


# ---- 0. Variables por contexto ------------------------------------------------

def variables_top_gain(contexto, x_ent, y_ent, obligatorias=()):
    """Las N_VARIABLES de mayor gain de LightGBM n6 (una semilla) en el
    entrenamiento del contexto, con las obligatorias (las columnas de los
    auxiliares de ingreso) siempre adentro.

    Devuelve: diccionario {variables, ganancia} (caché en etapa9/variables/).
    """
    ruta = RUTA_ETAPA9 / "variables" / f"{contexto}.json"
    if ruta.exists():
        return leer_json(ruta)
    inicio = time.time()
    modelo = ajustar_lightgbm(x_ent, y_ent, PARAMETROS_N6, SEMILLA)
    ganancia = pd.Series(modelo.feature_importance("gain"), index=x_ent.columns).sort_values(ascending=False)
    obligatorias = list(obligatorias)
    elegidas = [c for c in ganancia.index if c not in obligatorias]
    elegidas = obligatorias + elegidas[:N_VARIABLES - len(obligatorias)]
    elegidas = sorted(elegidas, key=lambda c: -ganancia[c])
    salida = {"variables": elegidas, "ganancia": [float(ganancia[c]) for c in elegidas],
              "share_gain": float(ganancia[elegidas].sum() / ganancia.sum())}
    escribir_json(salida, ruta)
    registrar_log("etapa9_variables", f"{contexto}: {len(elegidas)} variables, "
                                      f"{100 * salida['share_gain']:.1f}% del gain", inicio)
    del modelo
    gc.collect()
    return salida


def crear_preparador_top(preparar_completo, estado):
    """preparar() con las variables top del contexto; guarda las matrices
    reducidas en parquet para que GA, RF y CART no rearmen la base completa.
    Anota el contexto en estado["contexto"] para la caché por bolsa del GA y
    lee de estado["auxiliares"] las columnas que siempre entran.
    """
    def preparar(contexto):
        estado["contexto"] = contexto
        carpeta = RUTA_ETAPA9 / "matrices"
        rutas = {p: carpeta / f"{contexto}_{p}.parquet" for p in ("ent", "eval")}
        if all(r.exists() for r in rutas.values()):
            x_ent = pd.read_parquet(rutas["ent"])
            y_ent = x_ent.pop("pobre").to_numpy()
            return x_ent, y_ent, pd.read_parquet(rutas["eval"])
        x_ent, y_ent, x_eval = preparar_completo(contexto)
        variables = variables_top_gain(contexto, x_ent, y_ent, estado["auxiliares"])["variables"]
        x_ent, x_eval = x_ent[variables].astype(np.float32), x_eval[variables].astype(np.float32)
        carpeta.mkdir(parents=True, exist_ok=True)
        guardar_parquet_atomico(x_ent.assign(pobre=y_ent), rutas["ent"])
        guardar_parquet_atomico(x_eval, rutas["eval"])
        gc.collect()
        return x_ent, y_ent, x_eval

    return preparar


def preparar_todos(preparar):
    """Arma (o lee) las matrices reducidas de los 7 contextos."""
    for contexto in definir_contextos():
        preparar(contexto)
        gc.collect()


# ---- 2. GA-bagging ------------------------------------------------------------

def guardar_bolsa(salida, ruta):
    """Escribe los árboles de una bolsa en .npz con escritura atómica."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(buffer, variable=np.stack([a[0] for a in salida["arboles"]]),
             umbral=np.stack([a[1] for a in salida["arboles"]]),
             p_hoja=np.stack([a[2] for a in salida["arboles"]]),
             aptitud=np.array(salida["aptitud"]), f1=np.array(salida["f1"]),
             historia=np.array(salida["historia"]), segundos=np.array(salida["segundos"]),
             evaluaciones=np.array(salida["evaluaciones"]))
    temporal = ruta.with_suffix(".tmp")
    temporal.write_bytes(buffer.getvalue())
    temporal.replace(ruta)


def leer_bolsa(ruta):
    """Lee una bolsa guardada con guardar_bolsa()."""
    with np.load(ruta) as datos:
        salida = {k: datos[k] for k in datos.files}
    salida["arboles"] = list(zip(salida["variable"], salida["umbral"], salida["p_hoja"]))
    return salida


def crear_ajuste_ga(nombre, config, estado):
    """Función ajustar() del GA-bagging para evaluar_modelo().

    Cada bolsa se guarda al terminar en etapa9/ga/<nombre>/<contexto>/, y su semilla
    depende solo del contexto y del número de bolsa, así que retomar da lo mismo.
    Argumentos: nombre = candidato; config = CONFIG_GA + bolsas, poblacion y
        generaciones; estado = diccionario que llena crear_preparador_top().
    Devuelve: función(x_ent, y_ent, x_eval) -> (probabilidad, veces que se usa
        cada variable en los árboles elegidos).
    """
    contextos = list(definir_contextos())

    def ajustar(x_ent, y_ent, x_eval):
        contexto = estado["contexto"]
        ganancia = variables_top_gain(contexto, None, None)
        pesos = np.sqrt(pd.Series(ganancia["ganancia"], index=ganancia["variables"])[x_ent.columns].to_numpy())
        cortes = calcular_cortes(x_ent, N_CORTES)
        xd_ent, xd_eval = discretizar(x_ent, cortes), discretizar(x_eval, cortes)
        n_umbrales = [len(c) for c in cortes]
        probabilidad, usos = np.zeros(len(x_eval)), np.zeros(x_ent.shape[1])
        for b in range(config["bolsas"]):
            ruta = RUTA_ETAPA9 / "ga" / nombre / contexto / f"bolsa_{b:02d}.npz"
            if ruta.exists():
                salida = leer_bolsa(ruta)
            else:
                inicio = time.time()
                rng = np.random.default_rng(SEMILLA + 1000 * contextos.index(contexto) + b)
                bolsa = muestra_estratificada(y_ent, TAMANO_BOLSA, rng)
                oob = np.setdiff1d(np.arange(len(y_ent)), bolsa)
                genetico = GeneticoArboles(xd_ent[bolsa], y_ent[bolsa], xd_ent[oob], y_ent[oob], n_umbrales,
                                           pesos, config, int(rng.integers(2**31)))
                salida = genetico.correr()
                salida["segundos"] = time.time() - inicio
                salida["evaluaciones"] = genetico.n_evaluaciones
                guardar_bolsa(salida, ruta)
                registrar_log(nombre, f"{contexto} bolsa {b + 1}/{config['bolsas']}: F1 OOB "
                                      f"{max(salida['f1']):.4f} ({salida['segundos']:.0f} s)")
                del genetico
            probabilidad += predecir_arboles(salida["arboles"], xd_eval) / config["bolsas"]
            for variable, _, _ in salida["arboles"]:
                np.add.at(usos, variable[variable >= 0], 1)
        return probabilidad, pd.Series(usos, index=x_ent.columns)

    return ajustar


def medir_y_escalar(preparar, estado, reloj):
    """Corre la configuración chica en cv_1 y decide la escala que cabe en el
    presupuesto que queda. La decisión se guarda para que un reinicio no cambie.

    Devuelve: diccionario de configuración del GA completo.
    """
    ruta = RUTA_ETAPA9 / "decision_escala.json"
    decision = leer_json(ruta)
    if decision:
        return decision["config"]
    config_chica = {**CONFIG_GA, **CHICA}
    x_ent, y_ent, x_eval = preparar("cv_1")
    inicio = time.time()
    crear_ajuste_ga("n9_ga_chica", config_chica, estado)(x_ent, y_ent, x_eval)
    segundos = time.time() - inicio
    bolsas = [leer_bolsa(p) for p in sorted((RUTA_ETAPA9 / "ga" / "n9_ga_chica" / "cv_1").glob("*.npz"))]
    evaluaciones = sum(int(b["evaluaciones"]) for b in bolsas)
    segundos_eval = sum(float(b["segundos"]) for b in bolsas) / evaluaciones
    # Evaluaciones nuevas por generación (sin repetir árboles ya evaluados), medidas en la chica
    fraccion_nuevas = evaluaciones / (len(bolsas) * CHICA["poblacion"] * (CHICA["generaciones"] + 1))

    # El reloj ya cuenta variables, CART, RF y la configuración chica
    disponible = (MINUTOS_TOTALES - RESERVA_MIN - reloj.transcurrido()) * 60

    def por_bolsa(poblacion, generaciones):
        return poblacion * (generaciones + 1) * fraccion_nuevas * segundos_eval

    # Cada árbol recorre todo el entrenamiento del contexto (bolsa + OOB): holdout
    # y test cuestan más que cv_1 en proporción a sus filas
    filas = {c: len(pd.read_parquet(RUTA_ETAPA9 / "matrices" / f"{c}_ent.parquet", columns=["pobre"]))
             for c in definir_contextos()}
    contextos_equivalentes = sum(filas.values()) / filas["cv_1"]
    bolsas_n, poblacion = CHICA["bolsas"], POBLACION_ESCALA
    generaciones = int(disponible / (contextos_equivalentes * bolsas_n * por_bolsa(poblacion, 0))) - 1
    generaciones = int(np.clip(generaciones, *GENERACIONES_RANGO))
    if generaciones == GENERACIONES_RANGO[1]:
        bolsas_n = int(np.clip(disponible / (contextos_equivalentes * por_bolsa(poblacion, generaciones)),
                               CHICA["bolsas"], BOLSAS_MAXIMO))
    config = {**CONFIG_GA, "bolsas": bolsas_n, "poblacion": poblacion, "generaciones": generaciones}
    estimado = contextos_equivalentes * bolsas_n * por_bolsa(poblacion, generaciones) / 60
    decision = {"config": config, "segundos_chica_cv_1": segundos, "segundos_por_evaluacion": segundos_eval,
                "fraccion_evaluaciones_nuevas": fraccion_nuevas, "minutos_disponibles": disponible / 60,
                "minutos_estimados": estimado,
                "f1_oob_chica_por_bolsa": [float(max(b["f1"])) for b in bolsas]}
    escribir_json(decision, ruta)
    registrar_log("etapa9_escala", f"chica cv_1 {segundos:.0f} s, {1000 * segundos_eval:.1f} ms/árbol -> "
                                   f"B {bolsas_n}, población {poblacion}, {generaciones} gen. (~{estimado:.0f} min)")
    return config


# ---- 3. Random Forest y CART --------------------------------------------------

def elegir_por_validacion(crear, grilla, x_ent, y_ent, contexto):
    """Mejor configuración de la grilla por F1 (umbral óptimo) en una validación
    interna estratificada del entrenamiento del contexto (nunca su evaluación).

    Devuelve: (configuración, tabla de la búsqueda).
    """
    rng = np.random.default_rng(SEMILLA)
    validacion = muestra_estratificada(y_ent, int(FRACCION_VALIDACION * len(y_ent)), rng)
    entrenamiento = np.setdiff1d(np.arange(len(y_ent)), validacion)
    x_ent = x_ent.to_numpy(np.float32)
    filas = []
    for parametros in grilla:
        modelo = crear(parametros).fit(x_ent[entrenamiento], y_ent[entrenamiento])
        f1 = buscar_umbral_f1(y_ent[validacion], modelo.predict_proba(x_ent[validacion])[:, 1])[1]
        filas.append({"contexto": contexto, **parametros, "f1_validacion": f1})
    tabla = pd.DataFrame(filas)
    mejor = grilla[int(tabla["f1_validacion"].to_numpy().argmax())]
    return mejor, tabla


def crear_ajuste_sklearn(clave, crear_busqueda, crear_final, grilla, estado):
    """Función ajustar() de RF o CART con su grilla por contexto.

    Las búsquedas de cada contexto se acumulan en estado["busquedas"][clave].
    """
    def ajustar(x_ent, y_ent, x_eval):
        mejor, tabla = elegir_por_validacion(crear_busqueda, grilla, x_ent, y_ent, estado["contexto"])
        estado.setdefault("busquedas", {}).setdefault(clave, []).append(tabla)
        modelo = crear_final(mejor).fit(x_ent.to_numpy(np.float32), y_ent)
        probabilidad = modelo.predict_proba(x_eval.to_numpy(np.float32))[:, 1]
        importancia = pd.Series(modelo.feature_importances_, index=x_ent.columns)
        del modelo
        gc.collect()
        return probabilidad, importancia

    return ajustar


def crear_cart(parametros):
    return DecisionTreeClassifier(**parametros, random_state=SEMILLA)


def crear_rf(arboles):
    return lambda parametros: RandomForestClassifier(n_estimators=arboles, **parametros, n_jobs=N_HILOS,
                                                     random_state=SEMILLA)


# ---- 3. Mezcla y correlación --------------------------------------------------

def mezclar_con_base(nombre, nombre_ga, verdad):
    """(1 - w) * LightGBM base + w * GA con w elegido por F1 OOF (grilla de la etapa 7).

    Devuelve: (resultado, peso del GA).
    """
    a = resultado_desde_predicciones(BASE, verdad)
    b = resultado_desde_predicciones(nombre_ga, verdad)
    oof_b = b["oof"]["probabilidad"].reindex(a["oof"].index)
    f1_por_peso = {w: aptitud_f1(a["oof"]["verdad"], (1 - w) * a["oof"]["probabilidad"] + w * oof_b)[1]
                   for w in PESOS_MEZCLA}
    peso = max(f1_por_peso, key=f1_por_peso.get)

    def combinar(serie_a, serie_b):
        return ((1 - peso) * serie_a + peso * serie_b.reindex(serie_a.index)).rename("probabilidad")

    resultado = {"nombre": nombre,
                 "oof": pd.DataFrame({"probabilidad": combinar(a["oof"]["probabilidad"], b["oof"]["probabilidad"]),
                                      "fold": a["oof"]["fold"]}),
                 "prediccion_holdout": combinar(a["prediccion_holdout"], b["prediccion_holdout"]),
                 "prediccion_test": combinar(a["prediccion_test"], b["prediccion_test"])}
    guardar_predicciones(resultado)
    pd.DataFrame({"peso_ga": list(f1_por_peso), "f1_oof": list(f1_por_peso.values())}) \
        .to_csv(RUTA_TABLAS / "etapa9_mezcla_pesos.csv", index=False)
    registrar_log(nombre, f"peso GA {peso:.2f} (F1 OOF {f1_por_peso[peso]:.4f})")
    return resultado_desde_predicciones(nombre, verdad), float(peso)


def correlacion_oof(resultados):
    """Correlación de Pearson y de Spearman de las probabilidades OOF de cada
    candidato con las de LightGBM (base)."""
    base = resultados[BASE]["oof"]["probabilidad"]
    filas = []
    for nombre, resultado in resultados.items():
        if nombre == BASE:
            continue
        otra = resultado["oof"]["probabilidad"].reindex(base.index)
        filas.append({"candidato": nombre, "pearson_con_lightgbm": base.corr(otra),
                      "spearman_con_lightgbm": base.corr(otra, method="spearman")})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / "etapa9_correlacion_oof.csv", index=False)
    return tabla


def historia_ga(nombre):
    """Mejor aptitud por generación de cada bolsa y contexto (para la figura de evolución)."""
    filas = []
    for ruta in sorted((RUTA_ETAPA9 / "ga" / nombre).glob("*/*.npz")):
        bolsa = leer_bolsa(ruta)
        for generacion, aptitud in enumerate(bolsa["historia"]):
            filas.append({"contexto": ruta.parent.name, "bolsa": int(ruta.stem.split("_")[1]),
                          "generacion": generacion, "mejor_aptitud": float(aptitud)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / "etapa9_ga_historia.csv", index=False)
    return tabla


# ---- 4. Tabla, CSV y leaderboard ----------------------------------------------

def cerrar(entorno, resultados, descripciones):
    """Tabla de candidatos, CSV del mejor que cumple la regla y leaderboard."""
    verdad, base = entorno["verdad"], resultados[BASE]
    filas = []
    for nombre, resultado in resultados.items():
        fila = fila_tabla(resultado, verdad, descripciones[nombre]["descripcion"], base)
        fila["cumple_regla_csv"] = bool(fila["delta_cv"] >= MEJORA_MINIMA and fila["delta_holdout"] >= MEJORA_MINIMA)
        filas.append(fila)
    tabla = pd.DataFrame(filas).drop(columns=["supera_base"])
    tabla.to_csv(RUTA_TABLAS / "etapa9_candidatos.csv", index=False)

    cumplen = tabla[tabla["cumple_regla_csv"]].sort_values("f1_holdout", ascending=False)
    elegido = cumplen["candidato"].iloc[0] if len(cumplen) else None
    registrar_log("etapa9", f"candidatos que cumplen la regla: {list(cumplen['candidato'])} -> CSV: {elegido}")
    ids_test = entorno["particion"].loc[entorno["particion"]["base"] == "test", "id"]

    filas_lb = []
    for nombre, descripcion in descripciones.items():
        if not nombre.startswith("n9_"):
            continue
        archivo = None
        if nombre == elegido:
            ruta, tasa = guardar_submission(resultados[nombre], f"{nombre}.csv", ids_test,
                                            descripcion["especificacion"] + " | umbral OOF",
                                            algoritmo=descripcion["algoritmo"])
            archivo = f"nivel7/{ruta.name}"
            registrar_log("etapa9", f"submission {ruta.name} tasa {tasa:.4f}")
        filas_lb.append(armar_fila(nombre, descripcion["algoritmo"], descripcion["especificacion"],
                                   descripcion["variables"], descripcion["busqueda"],
                                   descripcion["n_configuraciones"], descripcion["hiperparametros"],
                                   resultados[nombre].get("tiempo_min", pd.NA), archivo, verdad))
    # Se agregan a las filas de las etapas anteriores (no las reemplazan)
    extra = pd.read_csv(RUTA_LEADERBOARD_EXTRA, dtype=str, keep_default_na=False)
    nuevas = formatear_como_r(pd.DataFrame(filas_lb)).astype(str).replace({"<NA>": "NA", "nan": "NA"})
    extra = pd.concat([extra[~extra["candidato"].isin(nuevas["candidato"])], nuevas[extra.columns]],
                      ignore_index=True)
    extra.to_csv(RUTA_LEADERBOARD_EXTRA, index=False)
    anexar_filas_extra()
    return tabla, elegido


def main():
    RUTA_ETAPA9.mkdir(parents=True, exist_ok=True)
    reloj = Presupuesto(MINUTOS_TOTALES)
    registrar_log("etapa9", "inicio")
    iniciar_vigilante_memoria("etapa9")
    insumos = cargar_insumos(auxiliar_total=True, auxiliar_fuentes=True)
    particion = insumos["particion"]
    verdad = particion.set_index("id")["pobre"]
    entorno = {"particion": particion, "verdad": verdad}
    # Columnas de los auxiliares de ingreso (laboral, total y fuentes): siempre entran
    estado = {"auxiliares": [c for a in AUXILIARES_BASE for c in insumos["auxiliares"][a]["cv_1"].columns]}

    # 0. Variables y matrices reducidas (la base completa se arma una vez por contexto)
    matriz_a = armar_matriz(insumos).join(pd.read_parquet(RUTA_ETAPA7 / "agregados_personas.parquet"))
    preparar_completo = crear_preparador(matriz_a, particion, [insumos["auxiliares"][a] for a in AUXILIARES_BASE])
    preparar = crear_preparador_top(preparar_completo, estado)
    preparar_todos(preparar)
    del matriz_a, preparar_completo, insumos
    gc.collect()
    registrar_log("etapa9", f"variables y matrices listas ({reloj.transcurrido():.1f} min)")
    variables_holdout = variables_top_gain("holdout", None, None)["variables"]

    resultados = {BASE: resultado_desde_predicciones(BASE, verdad)}
    resultados[BASE]["tiempo_min"] = np.nan
    descripciones = {BASE: {"descripcion": "base: A (LightGBM n6, bagging 5 semillas, todas las variables)"}}

    # 3a. CART y Random Forest con las mismas variables (antes del GA, para conocer su costo)
    inicio_sk = time.time()
    especificaciones_sk = {
        "cart": ("CART", crear_cart, crear_cart, GRILLA_CART, "CART (sklearn) con las 60 variables del GA",
                 "max_depth {4..8} x min_samples_leaf {25, 50, 100, 200, 400}"),
        "rf": ("Random Forest", crear_rf(ARBOLES_RF_BUSQUEDA), crear_rf(ARBOLES_RF), GRILLA_RF,
               f"Random Forest ({ARBOLES_RF} árboles) con las 60 variables del GA",
               f"max_features {{sqrt, 0.33}} x min_samples_leaf {{5, 20}} ({ARBOLES_RF_BUSQUEDA} árboles)"),
    }
    for clave, (algoritmo, crear_busqueda, crear_final, grilla, especificacion, rango) in especificaciones_sk.items():
        nombre = NOMBRES[clave]
        resultado = correr_seguro("etapa9", nombre, lambda: evaluar_o_cargar(nombre, lambda: evaluar_modelo(
            nombre, preparar, crear_ajuste_sklearn(clave, crear_busqueda, crear_final, grilla, estado), particion,
            con_test=True, reanudable=True), verdad))
        if resultado is None:
            continue
        resultados[nombre] = resultado
        if estado.get("busquedas", {}).get(clave):
            pd.concat(estado["busquedas"][clave]).to_csv(RUTA_TABLAS / f"etapa9_busqueda_{clave}.csv", index=False)
        descripciones[nombre] = {
            "descripcion": especificacion, "algoritmo": algoritmo, "variables": variables_holdout,
            "especificacion": f"{especificacion} (top gain de LightGBM n6 por contexto)",
            "busqueda": f"grilla {rango}; F1 con umbral óptimo en una validación interna estratificada del "
                        f"{int(100 * FRACCION_VALIDACION)}% del entrenamiento de cada contexto",
            "n_configuraciones": len(grilla), "hiperparametros": "elegidos por contexto (ver tabla de búsqueda)"}
    registrar_log("etapa9", f"CART y RF listos ({(time.time() - inicio_sk) / 60:.1f} min)")

    # 2. GA-bagging: medición chica y escala
    config = correr_seguro("etapa9", "escala", lambda: medir_y_escalar(preparar, estado, reloj))
    nombre_ga = NOMBRES["ga"]
    if config is not None:
        ga = correr_seguro("etapa9", nombre_ga, lambda: evaluar_o_cargar(nombre_ga, lambda: evaluar_modelo(
            nombre_ga, preparar, crear_ajuste_ga(nombre_ga, config, estado), particion, con_test=True,
            reanudable=True), verdad))
        if ga is not None:
            resultados[nombre_ga] = ga
            historia_ga(nombre_ga)
            descripciones[nombre_ga] = {
                "descripcion": f"GA-bagging (B {config['bolsas']}, población {config['poblacion']}, "
                               f"{config['generaciones']} generaciones, profundidad <= "
                               f"{config['profundidad_maxima']})",
                "algoritmo": "Random Forest", "variables": variables_holdout,
                "especificacion": (f"Bagging de árboles evolutivos: {config['bolsas']} bolsas estratificadas de "
                                   f"{TAMANO_BOLSA} hogares; genético con cruce de subárboles y mutación; "
                                   f"aptitud = F1 OOB con umbral óptimo - {config['penalizacion']} x profundidad; "
                                   f"promedio de los {config['mejores']} mejores árboles por bolsa"),
                "busqueda": ("configuración chica (B 10, población 30, 10 generaciones) en cv_1 para medir "
                             "tiempos y escala al presupuesto de 3 horas; operadores e hiperparámetros del GA fijos"),
                "n_configuraciones": 1, "hiperparametros": texto_hiperparametros(config)}

    # 3b. Mezcla con LightGBM y correlación
    if nombre_ga in resultados:
        salida = correr_seguro("etapa9", NOMBRES["mezcla"],
                               lambda: mezclar_con_base(NOMBRES["mezcla"], nombre_ga, verdad))
        if salida is not None:
            resultados[NOMBRES["mezcla"]], peso = salida
            descripciones[NOMBRES["mezcla"]] = {
                "descripcion": f"mezcla LightGBM base {1 - peso:.2f} + GA-bagging {peso:.2f}",
                "algoritmo": "Boosting", "variables": variables_holdout,
                "especificacion": f"Promedio ponderado LightGBM (A) {1 - peso:.2f} + GA-bagging {peso:.2f}",
                "busqueda": "peso del GA en {0, 0.05, ..., 1} por F1 OOF; umbral OOF",
                "n_configuraciones": len(PESOS_MEZCLA), "hiperparametros": f"peso_ga = {peso:.2f}"}
    correlacion = correlacion_oof(resultados)

    tabla, elegido = cerrar(entorno, resultados, descripciones)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(tabla[["candidato", "f1_cv", "f1_cv_sd", "f1_holdout", "precision_holdout", "recall_holdout",
                     "f1_banda_08_12", "tiempo_min", "delta_cv", "delta_holdout", "cumple_regla_csv"]].round(4))
        print(correlacion.round(4))
    for nombre in NOMBRES.values():
        borrar_parciales(nombre)
    registrar_log("etapa9", f"fin: CSV {elegido} ({reloj.transcurrido():.1f} min)")


if __name__ == "__main__":
    main()
