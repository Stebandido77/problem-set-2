"""Análisis de política del clasificador (etapa 15): errores por subgrupo,
pesos de bienestar (brecha de pobreza y distancia a la línea), curva de
cobertura y F1 ponderado por factores de expansión. Todo se calcula sobre
predicciones ya guardadas; nada reentrena.

Definiciones (las mismas del resto del pipeline):
  subcobertura = pobres no incluidos / pobres (1 - recall);
  filtración   = no pobres incluidos / incluidos (1 - precisión).
"""

import numpy as np
import pandas as pd

EDAD_ADULTO_MAYOR = 65  # el mismo corte que n_adultos_mayores en la limpieza de R
CORTES_TAMANO = [0, 1, 2, 4, np.inf]
ETIQUETAS_TAMANO = ["1", "2", "3-4", "5+"]


def grupos_de_hogares(hogar, personas):
    """Dimensiones de subgrupo de cada hogar.

    Argumentos: hogar = cargar_hogares() (indexado por id, con clase, n_personas,
        jefe_mujer y dominio); personas = cargar_personas() (id, edad,
        recibe_pension; NA en la pensión cuenta como no la recibe).
    Devuelve: DataFrame indexado por id con zona, tamano, jefatura,
        mayor_sin_pension ("con" si algún miembro de 65+ no recibe pensión) y dominio.
    """
    mayores = personas[(personas["edad"] >= EDAD_ADULTO_MAYOR) & (personas["recibe_pension"].fillna(0) == 0)]
    con_mayor = hogar.index.isin(mayores["id"].unique())
    return pd.DataFrame({
        "zona": hogar["clase"].astype(str),
        "tamano": pd.cut(hogar["n_personas"], CORTES_TAMANO, labels=ETIQUETAS_TAMANO).astype(str),
        "jefatura": np.where(hogar["jefe_mujer"] == 1, "femenina", "masculina"),
        "mayor_sin_pension": np.where(con_mayor, "con", "sin"),
        "dominio": hogar["dominio"].astype(str),
    }, index=hogar.index)


def errores_por_subgrupo(verdad, prediccion, grupos, factor=1.5):
    """Subcobertura y filtración por dimensión y grupo, con una fila total.

    Argumentos: verdad, prediccion = Series 0/1 alineadas con grupos; grupos =
        grupos_de_hogares() (solo las columnas que se quieran); factor = un grupo
        se marca si su subcobertura supera factor x la subcobertura total.
    Devuelve: DataFrame con dimension, grupo, n_hogares, n_pobres,
        n_predichos_pobres, n_fn, n_fp, subcobertura, filtracion, share_hogares,
        share_pobres y marca_subcobertura (NaN en las tasas sin denominador).
    """
    verdad = pd.Series(verdad, index=grupos.index).astype(int)
    prediccion = pd.Series(prediccion, index=grupos.index).astype(int)

    def resumen(mascara):
        v, p = verdad[mascara], prediccion[mascara]
        pobres, predichos = int((v == 1).sum()), int((p == 1).sum())
        fn, fp = int(((v == 1) & (p == 0)).sum()), int(((v == 0) & (p == 1)).sum())
        return {"n_hogares": int(mascara.sum()), "n_pobres": pobres, "n_predichos_pobres": predichos,
                "n_fn": fn, "n_fp": fp, "subcobertura": fn / pobres if pobres else np.nan,
                "filtracion": fp / predichos if predichos else np.nan}

    todos = np.ones(len(grupos), bool)
    total = resumen(todos)
    filas = [{"dimension": "total", "grupo": "todos", **total}]
    for dimension in grupos.columns:
        for grupo in sorted(pd.unique(grupos[dimension])):
            filas.append({"dimension": dimension, "grupo": grupo, **resumen((grupos[dimension] == grupo).to_numpy())})
    tabla = pd.DataFrame(filas)
    tabla["share_hogares"] = tabla["n_hogares"] / total["n_hogares"]
    tabla["share_pobres"] = tabla["n_pobres"] / total["n_pobres"] if total["n_pobres"] else np.nan
    tabla["marca_subcobertura"] = tabla["subcobertura"] > factor * total["subcobertura"]
    return tabla


def pesos_bienestar(verdad, prediccion, razon, cerca=0.2, extrema=0.5):
    """Subcobertura y filtración ponderadas por bienestar.

    Cada pobre excluido pesa su brecha de pobreza (1 - Ingpcug/Lp) y cada no
    pobre incluido pesa su distancia a la línea (Ingpcug/Lp - 1); las tasas
    ponderadas dividen por la masa total de brecha de los pobres y de distancia
    de los no pobres.
    Argumentos: verdad, prediccion = 0/1; razon = Ingpcug / Lp; cerca = un hogar
        está cerca de la línea si |razon - 1| <= cerca; extrema = brecha mínima
        de la pobreza extrema (razon < 1 - extrema).
    Devuelve: diccionario con conteos, tasas simples y ponderadas, brechas y
        distancias medias y la proporción de errores cerca de la línea.
    """
    verdad, prediccion, razon = (np.asarray(a, dtype=float) for a in (verdad, prediccion, razon))
    pobre, incluido = verdad == 1, prediccion == 1
    brecha, distancia = np.clip(1 - razon, 0, None), np.clip(razon - 1, 0, None)
    excluidos, cubiertos, filtrados = pobre & ~incluido, pobre & incluido, ~pobre & incluido

    def media(valores, mascara):
        return float(valores[mascara].mean()) if mascara.any() else np.nan

    def proporcion(mascara, condicion):
        return float(condicion[mascara].mean()) if mascara.any() else np.nan

    return {
        "n_pobres": int(pobre.sum()), "n_excluidos": int(excluidos.sum()),
        "subcobertura": excluidos.sum() / pobre.sum() if pobre.any() else np.nan,
        "subcobertura_ponderada": brecha[excluidos].sum() / brecha[pobre].sum() if brecha[pobre].sum() else np.nan,
        "brecha_media_pobres": media(brecha, pobre), "brecha_media_excluidos": media(brecha, excluidos),
        "brecha_media_cubiertos": media(brecha, cubiertos),
        "excluidos_cerca_linea": proporcion(excluidos, razon >= 1 - cerca),
        "excluidos_pobreza_extrema": proporcion(excluidos, razon < 1 - extrema),
        "n_incluidos": int(incluido.sum()), "n_filtrados": int(filtrados.sum()),
        "filtracion": filtrados.sum() / incluido.sum() if incluido.any() else np.nan,
        "filtracion_ponderada": (distancia[filtrados].sum() / distancia[~pobre].sum()
                                 if distancia[~pobre].sum() else np.nan),
        "distancia_media_no_pobres": media(distancia, ~pobre), "distancia_media_incluidos": media(distancia, filtrados),
        "incluidos_cerca_linea": proporcion(filtrados, razon <= 1 + cerca),
    }


def curva_cobertura(verdad, probabilidad, puntos=None):
    """% de pobres cubiertos al atender el x % de hogares de mayor probabilidad.

    Argumentos: verdad = 0/1; probabilidad = P(pobre); puntos = proporciones de
        hogares atendidos (por defecto 1 %, 2 %, ..., 100 %).
    Devuelve: DataFrame con share_atendida y share_pobres_cubiertos.
    """
    verdad, probabilidad = np.asarray(verdad, dtype=float), np.asarray(probabilidad, dtype=float)
    puntos = np.arange(0.01, 1.0001, 0.01) if puntos is None else np.asarray(puntos, dtype=float)
    orden = np.argsort(-probabilidad, kind="stable")
    acumulado = np.cumsum(verdad[orden]) / verdad.sum()
    posiciones = np.clip(np.round(puntos * len(verdad)).astype(int), 1, len(verdad))
    return pd.DataFrame({"share_atendida": puntos, "share_pobres_cubiertos": acumulado[posiciones - 1]})


def punto_operacion(verdad, probabilidad, umbral):
    """(share de hogares atendidos, share de pobres cubiertos) con un umbral fijo."""
    verdad, probabilidad = np.asarray(verdad, dtype=float), np.asarray(probabilidad, dtype=float)
    incluido = probabilidad >= umbral
    return float(incluido.mean()), float(((verdad == 1) & incluido).sum() / (verdad == 1).sum())


def f1_ponderado(verdad, prediccion, pesos):
    """F1 de la clase pobre con cada hogar contando su peso (p. ej. Fex_c)."""
    verdad, prediccion, pesos = (np.asarray(a, dtype=float) for a in (verdad, prediccion, pesos))
    vp = pesos[(verdad == 1) & (prediccion == 1)].sum()
    fp = pesos[(verdad == 0) & (prediccion == 1)].sum()
    fn = pesos[(verdad == 1) & (prediccion == 0)].sum()
    return 0.0 if vp == 0 else float(2 * vp / (2 * vp + fp + fn))
