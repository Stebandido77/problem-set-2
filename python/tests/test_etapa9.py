"""Pruebas de la etapa 9: la aptitud del genético es exactamente el F1 del
pipeline (mismo F1 y mismo umbral), se calcula solo con las filas out-of-bag y
los operadores de cruce y mutación siempre dejan árboles válidos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import funciones.evolutivo as evolutivo
from configuracion import RUTA_NIVEL7
from funciones.datos import cargar_particion
from funciones.evolutivo import (HOJA, VACIO, GeneticoArboles, aptitud_f1, calcular_cortes, crear_vacio,
                                 discretizar, muestra_estratificada, partir, predecir_arboles,
                                 profundidad_arbol, profundidad_nodo, rutear, trasplantar)
from funciones.metricas import buscar_umbral_f1, calcular_metricas

CONFIG_PRUEBA = {"poblacion": 30, "generaciones": 15, "profundidad_maxima": 4, "prob_cruce": 0.8,
                 "prob_mutacion": 0.5, "torneo": 3, "elite": 2, "penalizacion": 0.0005,
                 "fuerza_suavizado": 10, "mejores": 3}


def datos_sinteticos(n=4000, semilla=1):
    """Pobre si x0 alto y x1 bajo (con ruido); x2 y x3 son ruido. 25% pobres aprox."""
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=["x0", "x1", "x2", "x3"])
    y = ((x["x0"] > 0.2) & (x["x1"] < 0.3) | (rng.random(n) < 0.05)).astype(int).to_numpy()
    return x, y


def armar_genetico(x, y, config=CONFIG_PRUEBA, semilla=0):
    cortes = calcular_cortes(x, 16)
    xd = discretizar(x, cortes)
    bolsa = muestra_estratificada(y, len(y) // 2, np.random.default_rng(semilla))
    oob = np.setdiff1d(np.arange(len(y)), bolsa)
    genetico = GeneticoArboles(xd[bolsa], y[bolsa], xd[oob], y[oob], [len(c) for c in cortes],
                               np.ones(x.shape[1]), config, semilla)
    return genetico, xd, bolsa, oob


def es_valido(variable, profundidad_maxima):
    """La raíz existe, los internos tienen dos hijos existentes, las hojas ninguno."""
    if variable[0] == VACIO:
        return False
    for i, v in enumerate(variable):
        hijos = [2 * i + 1, 2 * i + 2]
        if v >= 0:
            if profundidad_nodo(i) >= profundidad_maxima or any(variable[h] == VACIO for h in hijos):
                return False
        elif hijos[0] < len(variable) and any(variable[h] != VACIO for h in hijos):
            return False
    return True


# ---- 1. La métrica ------------------------------------------------------------

def test_aptitud_igual_al_pipeline_en_oof_de_n7_v3():
    """Mismo F1 y mismo umbral que buscar_umbral_f1() sobre la OOF guardada de n7_v3."""
    ruta = RUTA_NIVEL7 / "predicciones_n7_v3_fuentes_bag5.parquet"
    if not ruta.exists():
        pytest.skip("faltan las predicciones de n7_v3 (datos locales)")
    verdad = cargar_particion().set_index("id")["pobre"]
    oof = pd.read_parquet(ruta).query("tipo == 'oof'").set_index("id")["probabilidad"]
    y = verdad.loc[oof.index].astype(int).to_numpy()
    assert aptitud_f1(y, oof.to_numpy()) == buscar_umbral_f1(y, oof.to_numpy())


def test_aptitud_es_f1_con_umbral_optimo_y_no_accuracy():
    """Con 10% de pobres, una sola hoja (probabilidad constante) da accuracy 0,9 si
    se predice todo no pobre, pero su aptitud es el F1 de marcar a todos pobres."""
    y = np.r_[np.ones(100, int), np.zeros(900, int)]
    umbral, f1 = aptitud_f1(y, np.full(1000, 0.1))
    assert f1 == pytest.approx(2 * 100 / (2 * 100 + 900))
    # Con un predictor que separa algo, el F1 sale del umbral óptimo, no de 0,5
    rng = np.random.default_rng(0)
    probabilidad = np.clip(0.1 + 0.2 * y + rng.normal(0, 0.05, 1000), 0, 1)
    umbral, f1 = aptitud_f1(y, probabilidad)
    f1_05 = calcular_metricas(y, (probabilidad >= 0.5).astype(int))["f1"]
    assert umbral < 0.5 and f1 > 0.9 and f1_05 == 0.0


def test_aptitud_del_arbol_solo_usa_out_of_bag(monkeypatch):
    """Cada llamada a la métrica recibe exactamente las etiquetas OOB de la bolsa."""
    x, y = datos_sinteticos()
    genetico, xd, bolsa, oob = armar_genetico(x, y)
    llamadas = []

    def espia(verdad, probabilidad):
        llamadas.append((np.asarray(verdad).copy(), len(probabilidad)))
        return buscar_umbral_f1(verdad, probabilidad)

    monkeypatch.setattr(evolutivo, "aptitud_f1", espia)
    genetico.config = {**CONFIG_PRUEBA, "generaciones": 2}
    genetico.correr()
    assert llamadas
    for verdad, n in llamadas:
        assert n == len(oob) and np.array_equal(verdad, y[oob])
    assert len(np.intersect1d(bolsa, oob)) == 0


def test_aptitud_del_arbol_coincide_con_el_pipeline():
    """evaluar() = buscar_umbral_f1() de las probabilidades OOB - penalización."""
    x, y = datos_sinteticos()
    genetico, xd, bolsa, oob = armar_genetico(x, y)
    arbol = genetico.arbol_aleatorio()
    aptitud, f1 = genetico.evaluar(arbol)
    p_hoja = genetico.ajustar_hojas(arbol)
    esperado = buscar_umbral_f1(y[oob], p_hoja[rutear(*arbol, xd[oob])])[1]
    assert f1 == esperado
    assert aptitud == pytest.approx(esperado - CONFIG_PRUEBA["penalizacion"] * profundidad_arbol(arbol[0]))


# ---- 2. Árboles y operadores --------------------------------------------------

def test_rutear_respeta_los_cortes():
    x = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)[::-1]})
    cortes = calcular_cortes(x, 4)
    xd = discretizar(x, cortes)
    variable, umbral = crear_vacio(3)
    partir(variable, umbral, 0, 0, 1)
    hojas = rutear(variable, umbral, xd)
    assert np.array_equal(hojas == 1, x["a"].to_numpy() <= cortes[0][1])


def test_trasplantar_trunca_a_la_profundidad_maxima():
    profundo = crear_vacio(3)
    for i in (0, 1, 3):
        partir(*profundo, i, 0, 0)
    receptor = crear_vacio(3)
    for i in (0, 2):
        partir(*receptor, i, 1, 0)
    trasplantar(receptor, 6, profundo, 0, 3)  # nodo de profundidad 2 recibe un árbol de 3 niveles
    assert es_valido(receptor[0], 3)
    assert profundidad_arbol(receptor[0]) == 3
    assert receptor[0][6] == 0 and receptor[0][13] == HOJA


def test_cruce_y_mutacion_dejan_arboles_validos():
    x, y = datos_sinteticos()
    genetico, *_ = armar_genetico(x, y)
    poblacion = [genetico.arbol_aleatorio() for _ in range(40)]
    for _ in range(300):
        a, b = genetico.rng.choice(len(poblacion), 2, replace=False)
        hijo, hija = genetico.cruzar(poblacion[a], poblacion[b])
        hijo = genetico.mutar(hijo)
        for arbol in (hijo, hija):
            assert es_valido(arbol[0], CONFIG_PRUEBA["profundidad_maxima"])
            internos = np.flatnonzero(arbol[0] >= 0)
            assert all(arbol[1][i] < genetico.n_umbrales[arbol[0][i]] for i in internos)
        poblacion[a], poblacion[b] = hijo, hija


def test_genetico_mejora_y_encuentra_la_senal():
    x, y = datos_sinteticos()
    genetico, xd, bolsa, oob = armar_genetico(x, y)
    salida = genetico.correr()
    assert salida["historia"][-1] >= salida["historia"][0]
    assert max(salida["f1"]) > 0.8
    usadas = {int(v) for variable, _, _ in salida["arboles"] for v in variable[variable >= 0]}
    assert {0, 1} <= usadas
    probabilidad = predecir_arboles(salida["arboles"], xd)
    assert probabilidad.shape == (len(y),) and ((probabilidad >= 0) & (probabilidad <= 1)).all()


def test_muestra_estratificada_conserva_la_tasa():
    y = np.r_[np.ones(3000, int), np.zeros(7000, int)]
    posiciones = muestra_estratificada(y, 2000, np.random.default_rng(0))
    assert len(posiciones) == 2000 and len(np.unique(posiciones)) == 2000
    assert y[posiciones].mean() == pytest.approx(0.3, abs=0.001)


# ---- 3. GA-bagging reanudable -------------------------------------------------

def test_ga_bagging_reanuda_con_el_mismo_resultado(tmp_path, monkeypatch):
    """Cada bolsa queda en disco; leerla o recalcularla (misma semilla) da lo mismo."""
    import etapa9_evolutivo as etapa9  # noqa: PLC0415 (importa la etapa completa)
    x, y = datos_sinteticos()
    monkeypatch.setattr(etapa9, "RUTA_ETAPA9", tmp_path)
    monkeypatch.setattr(etapa9, "TAMANO_BOLSA", 1500)
    monkeypatch.setattr(etapa9, "registrar_log", lambda *a, **k: None)
    monkeypatch.setattr(etapa9, "variables_top_gain",
                        lambda *a: {"variables": list(x.columns), "ganancia": [4.0, 3.0, 2.0, 1.0]})
    ajustar = etapa9.crear_ajuste_ga("prueba", {**CONFIG_PRUEBA, "bolsas": 2}, {"contexto": "cv_1"})
    entrenamiento, evaluacion = x.iloc[:3000], x.iloc[3000:]
    primera, usos = ajustar(entrenamiento, y[:3000], evaluacion)
    assert len(list((tmp_path / "ga" / "prueba" / "cv_1").glob("*.npz"))) == 2
    assert usos.sum() > 0 and list(usos.index) == list(x.columns)
    desde_cache, _ = ajustar(entrenamiento, y[:3000], evaluacion)
    np.testing.assert_array_equal(primera, desde_cache)
    for ruta in (tmp_path / "ga" / "prueba" / "cv_1").glob("*.npz"):
        ruta.unlink()
    recalculada, _ = ajustar(entrenamiento, y[:3000], evaluacion)
    np.testing.assert_allclose(primera, recalculada)
    assert buscar_umbral_f1(y[3000:], primera)[1] > 0.7
