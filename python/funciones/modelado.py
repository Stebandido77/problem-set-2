"""Evaluación de un candidato LightGBM del nivel 7 con el mismo protocolo que
evaluar_candidato() en R: predicciones OOF en los 5 folds del 80%, umbral
elegido en OOF, F1 en holdout (reajuste sobre el 80%) y, si se pide, reajuste
sobre el 100% del train para predecir el test."""

import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from configuracion import CIUDADES_PEOR_F1, N_HILOS, RUTA_NIVEL7, SEMILLA
from funciones.anidado import definir_contextos, guardar_parquet_atomico
from funciones.bitacora import registrar_log
from funciones.datos import imputar_mediana
from funciones.metricas import buscar_umbral_f1, calcular_metricas, metricas_por_fold


def ajustar_lightgbm(x_entrenamiento, y_entrenamiento, parametros, semilla=SEMILLA):
    """Ajusta un LightGBM binario con hilos internos.

    Argumentos: x_entrenamiento, y_entrenamiento = datos; parametros = diccionario
        de LightGBM (num_iterations incluido); semilla = semilla del motor.
    Devuelve: booster ajustado.
    """
    parametros = {**parametros, "objective": "binary", "num_threads": N_HILOS,
                  "seed": semilla, "verbose": -1}
    datos = lgb.Dataset(x_entrenamiento, label=y_entrenamiento, free_raw_data=True)
    return lgb.train(parametros, datos)


def armar_contexto(matriz, dinamicas, contexto, ids):
    """Une la matriz fija con las variables de auxiliares de un contexto.

    Argumentos: matriz = DataFrame fijo indexado por id; dinamicas = lista de
        diccionarios {contexto: DataFrame} de predecir_anidado(); contexto = nombre;
        ids = hogares a devolver.
    Devuelve: DataFrame con las filas ids.
    """
    partes = [matriz.loc[ids]]
    for variables in dinamicas:
        partes.append(variables[contexto].reindex(ids).astype(np.float32))
    return pd.concat(partes, axis=1)


def evaluar_candidato(nombre, matriz, particion, parametros, dinamicas=(), con_test=False,
                      semillas=(SEMILLA,), guardar=True, reanudable=False):
    """CV + umbral OOF + holdout (+ test) de un LightGBM con parámetros fijos.

    Argumentos:
      nombre     = identificador del candidato (n7_...)
      matriz     = predictores fijos en dummies, indexados por id (train + test)
      particion  = cargar_particion()
      parametros = hiperparámetros de LightGBM
      dinamicas  = variables de auxiliares por contexto (predecir_anidado())
      con_test   = si True reajusta sobre el 100% y predice el test
      semillas   = varias semillas promedian las probabilidades (seed bagging)
      guardar    = escribe las predicciones en stores/procesados/nivel7/
      reanudable = guarda la predicción de cada fold (y holdout y test) al
                   terminarlo en parciales/<nombre>/ y, si ya existe, la lee en
                   vez de reajustar; quien llama la borra con borrar_parciales()
                   cuando ya no la necesita
    Devuelve: diccionario con métricas, umbral y predicciones.
    """
    inicio = time.time()
    registrar_log(nombre, "inicio")
    tabla = particion.set_index("id")
    grupos_contexto = definir_contextos()
    y = tabla["pobre"]

    ruta_parcial = RUTA_NIVEL7 / "parciales" / nombre

    def ajustar_predecir(contexto):
        parcial = ruta_parcial / f"{contexto}.parquet"
        if reanudable and parcial.exists():
            registrar_log(nombre, f"{contexto} desde caché parcial")
            return pd.read_parquet(parcial)["probabilidad"], None
        entrenamiento, evaluacion = grupos_contexto[contexto]
        ids_ent = tabla.index[en_grupos(tabla, entrenamiento)]
        ids_eval = tabla.index[en_grupos(tabla, evaluacion)]
        x_ent = armar_contexto(matriz, dinamicas, contexto, ids_ent)
        x_eval = armar_contexto(matriz, dinamicas, contexto, ids_eval)
        x_ent, x_eval = imputar_mediana(x_ent, x_eval)
        probabilidad = np.zeros(len(ids_eval))
        for semilla in semillas:
            modelo = ajustar_lightgbm(x_ent, y.loc[ids_ent].to_numpy(), parametros, semilla)
            probabilidad += modelo.predict(x_eval) / len(semillas)
        probabilidad = pd.Series(probabilidad, index=ids_eval, name="probabilidad")
        if reanudable:
            ruta_parcial.mkdir(parents=True, exist_ok=True)
            guardar_parquet_atomico(probabilidad.to_frame(), parcial)
        return probabilidad, modelo

    # 1. OOF en los 5 folds del 80%
    oof = []
    for k in range(1, 6):
        probabilidad, _ = ajustar_predecir(f"cv_{k}")
        oof.append(pd.DataFrame({"probabilidad": probabilidad, "fold": k}))
        registrar_log(nombre, f"fin fold {k}")
    oof = pd.concat(oof)
    oof["verdad"] = y.loc[oof.index].astype(int)

    # 2. Umbral solo con OOF
    umbral, _ = buscar_umbral_f1(oof["verdad"], oof["probabilidad"])
    metricas_cv = metricas_por_fold(oof["verdad"], oof["probabilidad"], oof["fold"], umbral)

    # 3. Holdout con el umbral OOF fijo
    prob_holdout, modelo_holdout = ajustar_predecir("holdout")
    verdad_holdout = y.loc[prob_holdout.index].astype(int)
    metricas_holdout = calcular_metricas(verdad_holdout, (prob_holdout >= umbral).astype(int))
    registrar_log(nombre, "fin holdout")

    resultado = {
        "nombre": nombre, "umbral": umbral, **metricas_cv,
        **{f"{k}_holdout": v for k, v in metricas_holdout.items()},
        "oof": oof, "prediccion_holdout": prob_holdout,
        # Si el holdout salió de la caché parcial no hay modelo para la importancia
        "importancia": (pd.Series(modelo_holdout.feature_importance("gain"),
                                  index=modelo_holdout.feature_name())
                        if modelo_holdout is not None else pd.Series(dtype=float)),
    }

    # 4. Reajuste sobre el 100% y test
    if con_test:
        prob_test, _ = ajustar_predecir("test")
        resultado["prediccion_test"] = prob_test
        registrar_log(nombre, "fin test")

    resultado["tiempo_min"] = (time.time() - inicio) / 60
    if guardar:
        guardar_predicciones(resultado)
    registrar_log(nombre, f"F1 CV {metricas_cv['f1_cv']:.4f} | holdout {metricas_holdout['f1']:.4f}", inicio)
    return resultado


def borrar_parciales(nombre):
    """Borra la caché parcial (por fold) de un candidato evaluado con reanudable=True."""
    shutil.rmtree(RUTA_NIVEL7 / "parciales" / nombre, ignore_errors=True)


def en_grupos(tabla, grupos):
    """Máscara de hogares cuyo grupo (fold 1-5, 6 holdout, 0 test) está en grupos."""
    grupo = tabla["fold_id"].astype("float").fillna(-1).astype(int).to_numpy().copy()
    grupo[tabla["holdout"].to_numpy()] = 6
    grupo[(tabla["base"] == "test").to_numpy()] = 0
    return np.isin(grupo, grupos)


def guardar_predicciones(resultado):
    """Escribe OOF, holdout y test de un candidato en stores/procesados/nivel7/."""
    partes = [
        resultado["oof"].assign(tipo="oof")[["tipo", "fold", "probabilidad"]],
        resultado["prediccion_holdout"].rename("probabilidad").to_frame().assign(tipo="holdout", fold=pd.NA),
    ]
    if "prediccion_test" in resultado:
        partes.append(resultado["prediccion_test"].rename("probabilidad").to_frame().assign(tipo="test", fold=pd.NA))
    salida = pd.concat(partes).rename_axis("id").reset_index()
    salida["fold"] = salida["fold"].astype("Int64")
    salida.to_parquet(RUTA_NIVEL7 / f"predicciones_{resultado['nombre']}.parquet", index=False)


def f1_subgrupos(resultado, hogar, particion):
    """F1 en hogares de 1-2 personas y en las 5 ciudades con peor F1 de n6, en
    OOF y en holdout, con el umbral OOF del candidato.

    Argumentos: resultado = evaluar_candidato(); hogar = cargar_hogares();
        particion = cargar_particion().
    Devuelve: diccionario de F1 por subgrupo.
    """
    verdad = particion.set_index("id")["pobre"]
    salida = {}
    for tipo, prob in (("oof", resultado["oof"]["probabilidad"]),
                       ("holdout", resultado["prediccion_holdout"])):
        base = hogar.loc[prob.index, ["n_personas", "dominio"]]
        clase = (prob >= resultado["umbral"]).astype(int)
        for etiqueta, mascara in (("1_2_personas", base["n_personas"] <= 2),
                                  ("5_ciudades", base["dominio"].isin(CIUDADES_PEOR_F1))):
            salida[f"f1_{etiqueta}_{tipo}"] = calcular_metricas(
                verdad.loc[prob.index[mascara]].astype(int), clase[mascara])["f1"]
    return salida
