"""Predicciones fuera de fold anidadas para variables que salen de un modelo
auxiliar (ingreso predicho), sin fuga hacia el modelo principal.

Grupos de hogares: folds 1-5 del 80% (los de R), 6 = holdout, 0 = test.
Contextos donde se ajusta el modelo principal:
  cv_k     entrena con los folds != k y evalúa en el fold k
  holdout  entrena con el 80% (folds 1-5) y evalúa en el holdout
  test     entrena con el 100% del train (grupos 1-6) y predice el test
Dentro de cada contexto:
  - las filas de evaluación reciben la predicción de un auxiliar entrenado con
    todo el entrenamiento del contexto;
  - las filas de entrenamiento reciben una predicción OOF interna: la del grupo j
    sale de un auxiliar entrenado con el entrenamiento del contexto menos j.
Así ningún hogar recibe una predicción de un modelo entrenado con él, y en cv_k
ningún auxiliar usado (ni para entrenar ni para evaluar) vio el fold k.
Cada auxiliar se identifica por el conjunto de grupos con que se entrenó; los
que se repiten entre contextos se ajustan una sola vez (23 ajustes en total).
"""

import shutil

import pandas as pd

from configuracion import RUTA_NIVEL7
from funciones.bitacora import registrar_log

GRUPOS_80 = (1, 2, 3, 4, 5)
GRUPO_HOLDOUT = 6
GRUPO_TEST = 0


def asignar_grupos(particion):
    """Grupo de cada hogar: fold_id en el 80%, 6 en holdout, 0 en test.

    Argumentos: particion = cargar_particion().
    Devuelve: Serie de enteros indexada por id.
    """
    grupo = particion["fold_id"].astype("float").fillna(-1).astype(int)
    grupo[particion["holdout"].to_numpy()] = GRUPO_HOLDOUT
    grupo[(particion["base"] == "test").to_numpy()] = GRUPO_TEST
    return pd.Series(grupo.to_numpy(), index=particion["id"].to_numpy(), name="grupo")


def definir_contextos():
    """Grupos de entrenamiento y de evaluación de cada contexto del modelo principal.

    Devuelve: diccionario {contexto: (grupos_entrenamiento, grupos_evaluacion)}.
    """
    contextos = {f"cv_{k}": (tuple(g for g in GRUPOS_80 if g != k), (k,)) for k in GRUPOS_80}
    contextos["holdout"] = (GRUPOS_80, (GRUPO_HOLDOUT,))
    contextos["test"] = (GRUPOS_80 + (GRUPO_HOLDOUT,), (GRUPO_TEST,))
    return contextos


def planear_ajustes(contextos):
    """Qué auxiliar (conjunto de grupos) predice a qué grupo en cada contexto.

    Devuelve: lista de filas (contexto, rol, grupo_predicho, grupos_modelo).
    """
    plan = []
    for contexto, (entrenamiento, evaluacion) in contextos.items():
        for g in evaluacion:
            plan.append((contexto, "evaluacion", g, frozenset(entrenamiento)))
        for j in entrenamiento:
            plan.append((contexto, "entrenamiento", j, frozenset(entrenamiento) - {j}))
    return plan


def guardar_parquet_atomico(tabla, ruta):
    """Escribe un parquet en un temporal y lo renombra, para que un corte a mitad
    de escritura nunca deje un archivo corrupto con el nombre definitivo.

    Argumentos: tabla = DataFrame; ruta = destino.
    """
    temporal = ruta.with_suffix(".tmp")
    tabla.to_parquet(temporal)
    temporal.replace(ruta)


def predecir_anidado(ajustar_predecir, grupos, nombre, contextos=None, ruta_parcial=None):
    """Corre el plan de auxiliares y arma las variables de cada contexto.

    Argumentos:
      ajustar_predecir = función(ids_entrenamiento, ids_prediccion) -> DataFrame de
                         variables indexado por id de hogar (solo ids_prediccion)
      grupos   = asignar_grupos(particion)
      nombre   = etiqueta para el log (None no escribe en el log)
      contextos = definir_contextos() por defecto
      ruta_parcial = carpeta donde se guarda cada ajuste al terminarlo; si el
                     ajuste ya está ahí se lee en vez de recalcularlo (reanudar
                     tras un corte). None no guarda nada.
    Devuelve: (variables, registro)
      variables = {contexto: DataFrame indexado por id con entrenamiento + evaluación}
      registro  = DataFrame (contexto, rol, id, grupo, grupos_modelo) para auditar fugas
    """
    contextos = contextos or definir_contextos()
    plan = planear_ajustes(contextos)

    # Un ajuste por conjunto distinto; predice todos los grupos que el plan le pide
    pedidos = {}
    for _, _, g, modelo in plan:
        pedidos.setdefault(modelo, set()).add(g)

    predicciones = {}
    for i, (modelo, grupos_pedidos) in enumerate(sorted(pedidos.items(), key=lambda x: sorted(x[0]))):
        ids_entrenamiento = grupos.index[grupos.isin(modelo)]
        ids_prediccion = grupos.index[grupos.isin(grupos_pedidos)]
        assert not set(grupos_pedidos) & set(modelo), "un auxiliar predice un grupo que vio"
        parcial = None
        if ruta_parcial is not None:
            etiqueta = "-".join(str(g) for g in sorted(modelo))
            parcial = ruta_parcial / f"modelo_{etiqueta}__pred_{'-'.join(str(g) for g in sorted(grupos_pedidos))}.parquet"
        if parcial is not None and parcial.exists():
            predicciones[modelo] = pd.read_parquet(parcial)
            evento = "desde caché parcial"
        else:
            predicciones[modelo] = ajustar_predecir(ids_entrenamiento, ids_prediccion)
            if parcial is not None:
                ruta_parcial.mkdir(parents=True, exist_ok=True)
                guardar_parquet_atomico(predicciones[modelo], parcial)
            evento = "listo"
        if nombre:
            registrar_log(nombre, f"auxiliar {i + 1}/{len(pedidos)} grupos {sorted(modelo)} {evento}")

    variables, registro = {}, []
    for contexto in contextos:
        partes = []
        for ctx, rol, g, modelo in plan:
            if ctx != contexto:
                continue
            ids = grupos.index[grupos == g]
            parte = predicciones[modelo].reindex(ids)
            partes.append(parte)
            registro.append(pd.DataFrame({
                "contexto": contexto, "rol": rol, "id": ids, "grupo": g,
                "grupos_modelo": ",".join(str(x) for x in sorted(modelo)),
            }))
        variables[contexto] = pd.concat(partes)
    return variables, pd.concat(registro, ignore_index=True)


def verificar_sin_fuga(registro, grupos, contextos=None):
    """Falla si algún hogar recibe una predicción de un auxiliar entrenado con él,
    o si en cv_k algún auxiliar usado vio el fold k.

    Argumentos: registro = salida de predecir_anidado(); grupos = asignar_grupos().
    Devuelve: True si todo está bien (si no, AssertionError).
    """
    contextos = contextos or definir_contextos()
    grupos_modelo = registro["grupos_modelo"].str.split(",").map(lambda x: {int(v) for v in x})
    grupo_real = grupos.reindex(registro["id"]).to_numpy()
    assert (registro["grupo"].to_numpy() == grupo_real).all(), "grupo del registro no coincide con la partición"
    propio = [g in m for g, m in zip(grupo_real, grupos_modelo)]
    assert not any(propio), f"{sum(propio)} hogares reciben predicción de un auxiliar entrenado con ellos"
    for contexto, (_, evaluacion) in contextos.items():
        usados = grupos_modelo[registro["contexto"] == contexto]
        for g in evaluacion:
            assert not any(g in m for m in usados), f"en {contexto} un auxiliar vio el grupo {g}"
    # Cada hogar del contexto aparece exactamente una vez
    for contexto, (entrenamiento, evaluacion) in contextos.items():
        ids = registro.loc[registro["contexto"] == contexto, "id"]
        esperados = grupos.isin(set(entrenamiento) | set(evaluacion)).sum()
        assert len(ids) == esperados and ids.is_unique, f"{contexto}: hogares faltantes o repetidos"
    return True


def crear_perezoso(crear):
    """Envuelve la construcción de un auxiliar para que solo ocurra si hace falta.

    Construir un auxiliar puede costar memoria y tiempo (p. ej. la matriz de
    personas con contexto del hogar); si la caché anidada ya existe,
    cargar_o_calcular_anidado() nunca llama a ajustar_predecir y no se construye.
    Argumentos: crear = función sin argumentos que devuelve ajustar_predecir.
    Devuelve: función(ids_entrenamiento, ids_prediccion) equivalente.
    """
    construido = []

    def ajustar_predecir(ids_entrenamiento, ids_prediccion):
        if not construido:
            construido.append(crear())
        return construido[0](ids_entrenamiento, ids_prediccion)

    return ajustar_predecir


def cargar_o_calcular_anidado(nombre, ajustar_predecir, grupos):
    """Variables de un auxiliar por contexto, desde caché si ya se calcularon.

    Argumentos: nombre = etiqueta del auxiliar; ajustar_predecir = función del
        auxiliar; grupos = asignar_grupos().
    Devuelve: diccionario {contexto: DataFrame}. Verifica que no haya fuga.
    """
    ruta = RUTA_NIVEL7 / f"anidado_{nombre}.parquet"
    ruta_registro = RUTA_NIVEL7 / f"registro_{nombre}.parquet"
    if ruta.exists() and ruta_registro.exists():
        tabla = pd.read_parquet(ruta)
        registro = pd.read_parquet(ruta_registro)
    else:
        # Cada ajuste se guarda al terminar en parciales/<nombre>/; si el proceso
        # se corta, la siguiente corrida retoma desde el último ajuste guardado
        ruta_parcial = RUTA_NIVEL7 / "parciales" / nombre
        variables, registro = predecir_anidado(ajustar_predecir, grupos, nombre, ruta_parcial=ruta_parcial)
        tabla = pd.concat([v.assign(contexto=c) for c, v in variables.items()]).rename_axis("id").reset_index()
        guardar_parquet_atomico(registro, ruta_registro)
        guardar_parquet_atomico(tabla, ruta)
        shutil.rmtree(ruta_parcial, ignore_errors=True)
    verificar_sin_fuga(registro, grupos)
    return {c: g.drop(columns="contexto").set_index("id") for c, g in tabla.groupby("contexto")}
