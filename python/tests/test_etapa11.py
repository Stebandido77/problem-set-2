"""Pruebas de la etapa 11: la regla del arriendo imputado, la reconciliación de
los auxiliares desde la caché y el auxiliar de estrato."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from configuracion import RUTA_NIVEL7, RUTA_PROCESADOS
from funciones.auxiliares import (ESTRATOS, agregar_ingreso_hogar, arriendo_imputado_hogar, crear_auxiliar_estrato,
                                  derivar_per_capita, resumir_probabilidades_estrato)
from funciones.bases import SUFIJO_AUXILIAR, reconciliar_auxiliares


def hogar_de_juguete():
    """Un hogar por tenencia, con arriendo estimado 500.000 y cuota 200.000 donde aplica."""
    return pd.DataFrame({
        "tenencia": pd.Categorical(["propia_pagada", "propia_pagando", "arriendo", "usufructo",
                                    "posesion_sin_titulo", "otra", "propia_pagando"]),
        "arriendo": [500000.0, 500000.0, 300000.0, 400000.0, 450000.0, 100000.0, np.nan],
        "P5100": [np.nan, 200000.0, np.nan, np.nan, np.nan, np.nan, 900000.0],
        "personas_ug": [2, 1, 3, 2, 1, 1, 2],
        "linea_pobreza": [300000.0] * 7,
    }, index=pd.Index([f"h{i}" for i in range(7)], name="id"))


def test_regla_del_arriendo_imputado_por_tenencia():
    imputado = arriendo_imputado_hogar(hogar_de_juguete())
    esperado = [500000, 300000, 0, 400000, 0, 0, 0]  # pagando: estimado - cuota, nunca negativo
    assert imputado.tolist() == esperado
    assert imputado.index.equals(hogar_de_juguete().index)


def test_agregar_ingreso_hogar_con_y_sin_arriendo():
    hogar = hogar_de_juguete()
    ids = ["h0", "h0", "h2"]
    sin = agregar_ingreso_hogar(ids, [100000, 200000, 600000], [True, True, True], hogar, arriendo_imputado=False)
    con = agregar_ingreso_hogar(ids, [100000, 200000, 600000], [True, True, True], hogar, arriendo_imputado=True)
    assert list(sin.columns) == ["ingreso_pred_total", "log_ingreso_pred_pc", "log_ingreso_pred_pc_lp", "ingreso_pred_max_lp"]
    # El total guardado sigue siendo solo ingresos personales; cambia el per cápita
    assert sin.loc["h0", "ingreso_pred_total"] == con.loc["h0", "ingreso_pred_total"] == 300000
    assert np.isclose(con.loc["h0", "log_ingreso_pred_pc_lp"], np.log((800000 / 2 + 1) / 300000))
    assert np.isclose(sin.loc["h0", "log_ingreso_pred_pc_lp"], np.log((300000 / 2 + 1) / 300000))
    # El arrendatario no recibe imputación: igual con y sin
    pd.testing.assert_series_equal(sin.loc["h2"], con.loc["h2"])


def test_reconciliar_sin_arriendo_reproduce_la_cache():
    """derivar_per_capita(arriendo_imputado=False) devuelve exactamente las columnas
    que guardó la caché de cada auxiliar (misma fórmula), y con True las cambia
    solo en hogares con arriendo imputado."""
    from funciones.datos import cargar_hogares
    rutas = {"laboral": "anidado_ingreso_personas.parquet", "fuentes": "anidado_ingreso_fuentes_v3.parquet"}
    if not all((RUTA_NIVEL7 / r).exists() for r in rutas.values()):
        pytest.skip("faltan las cachés de los auxiliares")
    hogar = cargar_hogares()
    imputado = arriendo_imputado_hogar(hogar)
    for nombre, ruta in rutas.items():
        tabla = pd.read_parquet(RUTA_NIVEL7 / ruta)
        tabla = tabla[tabla["contexto"] == "holdout"].drop(columns="contexto").set_index("id")
        sufijo = SUFIJO_AUXILIAR[nombre]
        sin = reconciliar_auxiliares({nombre: {"holdout": tabla}}, hogar, arriendo_imputado=False)[nombre]["holdout"]
        pd.testing.assert_frame_equal(sin, tabla)
        con = derivar_per_capita(tabla, hogar, sufijo, arriendo_imputado=True)
        cambia = con[f"log_ingreso_pred_pc_lp{sufijo}"] != tabla[f"log_ingreso_pred_pc_lp{sufijo}"]
        assert cambia.equals(imputado.reindex(tabla.index).gt(0))
        assert (con.loc[cambia, f"log_ingreso_pred_pc_lp{sufijo}"] > tabla.loc[cambia, f"log_ingreso_pred_pc_lp{sufijo}"]).all()


def test_resumen_de_probabilidades_de_estrato():
    probabilidades = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0.5, 0.5, 0, 0, 0, 0], [1 / 6] * 6])
    salida = resumir_probabilidades_estrato(probabilidades, ["a", "b", "c", "d"])
    assert salida["estrato_esperado"].tolist() == pytest.approx([1, 6, 1.5, 3.5])
    assert salida["p_estrato_bajo"].tolist() == pytest.approx([1, 0, 1, 1 / 3])
    assert salida.index.name == "id"


def test_auxiliar_de_estrato_solo_devuelve_los_hogares_pedidos():
    """El auxiliar predice únicamente ids_prediccion, no usa el estrato como
    predictor (no está en la matriz) y sus salidas quedan en rango."""
    rng = np.random.default_rng(2026)
    n = 600
    ids = pd.Index([f"h{i}" for i in range(n)], name="id")
    estrato = pd.Series(rng.integers(1, 7, n), index=ids)
    x = pd.DataFrame({"x1": estrato + rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)}, index=ids).astype(np.float32)
    ajustar = crear_auxiliar_estrato(x, estrato, {"objective": "multiclass", "num_class": len(ESTRATOS),
                                                  "num_iterations": 20, "num_leaves": 7, "min_data_in_leaf": 5,
                                                  "verbose": -1})
    salida = ajustar(ids[:400], ids[400:])
    assert list(salida.index) == list(ids[400:]) and list(salida.columns) == ["estrato_esperado", "p_estrato_bajo"]
    assert salida["estrato_esperado"].between(1, 6).all() and salida["p_estrato_bajo"].between(0, 1).all()
    assert salida["estrato_esperado"].corr(estrato[ids[400:]]) > 0.5


def test_estrato_exportado_es_valido():
    ruta = RUTA_PROCESADOS / "estrato_hogar_train.parquet"
    if not ruta.exists():
        pytest.skip("falta estrato_hogar_train.parquet (corre 11_exportar_python.R)")
    from funciones.datos import cargar_estrato_hogar, cargar_particion
    estrato = cargar_estrato_hogar()
    particion = cargar_particion()
    assert estrato.isin(ESTRATOS).all() and estrato.index.is_unique
    assert set(estrato.index) == set(particion.loc[particion["base"] == "train", "id"])
