"""Pruebas de la etapa 15: LightGBM con pesos por fila, selección y agregado de
pseudoetiquetas sin tocar la evaluación, errores por subgrupo, pesos de
bienestar, curva de cobertura, F1 ponderado, EBM y la revisión del ensamble por
fold."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from funciones.modelado import crear_ajuste_lightgbm
from funciones.politica import (curva_cobertura, errores_por_subgrupo, f1_ponderado, grupos_de_hogares,
                                pesos_bienestar, punto_operacion)

PARAMETROS_JUGUETE = {"num_iterations": 40, "learning_rate": 0.2, "num_leaves": 7, "min_data_in_leaf": 5}


def datos_de_juguete(n, semilla):
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["ingreso", "edad", "cuartos"],
                     index=pd.Index([f"h{semilla}_{i}" for i in range(n)], name="id"))
    y = ((x["ingreso"] + 0.5 * x["cuartos"] + rng.normal(scale=0.5, size=n)) < 0).astype(int).to_numpy()
    return x, y


def test_lightgbm_con_columna_de_peso_la_usa_como_peso_y_no_como_variable():
    x_ent, y_ent = datos_de_juguete(800, 0)
    x_eval, _ = datos_de_juguete(200, 1)
    sin_peso, _ = crear_ajuste_lightgbm(PARAMETROS_JUGUETE)(x_ent, y_ent, x_eval)
    pesos_unitarios = x_ent.assign(peso_fila=1.0)
    con_unitarios, importancia = crear_ajuste_lightgbm(PARAMETROS_JUGUETE, columna_peso="peso_fila")(
        pesos_unitarios, y_ent, x_eval)
    assert np.allclose(sin_peso, con_unitarios)
    assert "peso_fila" not in importancia.index and set(importancia.index) == set(x_ent.columns)
    # Pesar 10 veces más a los pobres sube la probabilidad media predicha
    mas_pobres = x_ent.assign(peso_fila=np.where(y_ent == 1, 10.0, 1.0))
    con_pesos, _ = crear_ajuste_lightgbm(PARAMETROS_JUGUETE, columna_peso="peso_fila")(mas_pobres, y_ent, x_eval)
    assert con_pesos.mean() > sin_peso.mean() + 0.05
    # La columna de peso también puede venir en x_eval (se ignora) y no se modifica la entrada
    con_eval, _ = crear_ajuste_lightgbm(PARAMETROS_JUGUETE, columna_peso="peso_fila")(
        pesos_unitarios, y_ent, x_eval.assign(peso_fila=3.0))
    assert np.allclose(con_eval, con_unitarios) and "peso_fila" in pesos_unitarios.columns


def test_seleccionar_y_agregar_pseudoetiquetas_respeta_los_umbrales_y_el_peso():
    from etapa15_modelos_politica import agregar_pseudo, seleccionar_pseudo
    probabilidad = pd.Series([0.99, 0.95, 0.5, 0.05, 0.01, 0.949], index=pd.Index(list("abcdef"), name="id"))
    etiquetas = seleccionar_pseudo(probabilidad, bajo=0.05, alto=0.95)
    assert etiquetas.to_dict() == {"a": 1, "b": 1, "d": 0, "e": 0}
    x_ent, y_ent = datos_de_juguete(10, 2)
    x_test, _ = datos_de_juguete(6, 3)
    x_test.index = pd.Index(list("abcdef"), name="id")
    x_nuevo, y_nuevo = agregar_pseudo(x_ent, y_ent, x_test, etiquetas, peso=0.5)
    assert len(x_nuevo) == 14 and len(y_nuevo) == 14
    assert list(x_nuevo["peso_fila"]) == [1.0] * 10 + [0.5] * 4
    assert list(y_nuevo[10:]) == [1, 1, 0, 0] and list(x_nuevo.index[10:]) == list("abde")
    assert list(x_nuevo.columns[:-1]) == list(x_ent.columns) and "peso_fila" not in x_ent.columns


def hogares_de_juguete():
    hogar = pd.DataFrame({
        "clase": pd.Categorical(["urbano", "urbano", "rural", "rural", "urbano", "urbano"]),
        "n_personas": [1, 2, 3, 4, 5, 9], "jefe_mujer": [1, 0, 1, 0, 0, 1],
        "dominio": pd.Categorical(["BOGOTA", "BOGOTA", "RESTO RURAL", "RESTO RURAL", "CALI", "CALI"]),
    }, index=pd.Index([f"h{i}" for i in range(6)], name="id"))
    personas = pd.DataFrame({"id": ["h0", "h1", "h1", "h2", "h3", "h5", "h5"],
                             "edad": [70, 30, 68, 40, 66, 80, 20],
                             "recibe_pension": [1, 0, 0, 0, np.nan, 0, 0]})
    return hogar, personas


def test_grupos_de_hogares_arma_las_cinco_dimensiones():
    hogar, personas = hogares_de_juguete()
    grupos = grupos_de_hogares(hogar, personas)
    assert list(grupos.columns) == ["zona", "tamano", "jefatura", "mayor_sin_pension", "dominio"]
    assert list(grupos["tamano"]) == ["1", "2", "3-4", "3-4", "5+", "5+"]
    assert list(grupos["jefatura"]) == ["femenina", "masculina", "femenina", "masculina", "masculina", "femenina"]
    # h0 tiene un mayor con pensión; h1, h3 (pensión NA = no recibe) y h5 tienen un mayor sin pensión
    assert list(grupos["mayor_sin_pension"]) == ["sin", "con", "sin", "con", "sin", "con"]
    assert list(grupos["zona"]) == ["urbano", "urbano", "rural", "rural", "urbano", "urbano"]


def test_errores_por_subgrupo_calcula_subcobertura_y_filtracion_y_marca_los_graves():
    hogar, personas = hogares_de_juguete()
    grupos = grupos_de_hogares(hogar, personas)[["zona"]]
    verdad = pd.Series([1, 1, 1, 1, 0, 0], index=hogar.index)
    prediccion = pd.Series([1, 0, 0, 0, 1, 0], index=hogar.index)
    tabla = errores_por_subgrupo(verdad, prediccion, grupos, factor=1.5).set_index(["dimension", "grupo"])
    total = tabla.loc[("total", "todos")]
    assert total["n_pobres"] == 4 and total["n_fn"] == 3 and total["n_fp"] == 1
    assert total["subcobertura"] == pytest.approx(0.75) and total["filtracion"] == pytest.approx(0.5)
    urbano, rural = tabla.loc[("zona", "urbano")], tabla.loc[("zona", "rural")]
    assert urbano["subcobertura"] == pytest.approx(0.5) and rural["subcobertura"] == pytest.approx(1.0)
    assert not urbano["marca_subcobertura"] and not rural["marca_subcobertura"]  # 1,0 < 1,5 x 0,75
    assert urbano["filtracion"] == pytest.approx(0.5) and np.isnan(rural["filtracion"])  # rural sin predichos
    tabla2 = errores_por_subgrupo(verdad, prediccion, grupos, factor=1.2).set_index(["dimension", "grupo"])
    assert tabla2.loc[("zona", "rural"), "marca_subcobertura"]


def test_pesos_de_bienestar_ponderan_por_brecha_y_distancia_a_la_linea():
    verdad = np.array([1, 1, 1, 0, 0, 0])
    prediccion = np.array([1, 1, 0, 1, 0, 0])
    razon = np.array([0.2, 0.9, 0.95, 1.1, 1.5, 3.0])  # Ingpcug / Lp
    r = pesos_bienestar(verdad, prediccion, razon)
    assert r["subcobertura"] == pytest.approx(1 / 3)
    assert r["subcobertura_ponderada"] == pytest.approx(0.05 / (0.8 + 0.1 + 0.05))
    assert r["brecha_media_excluidos"] == pytest.approx(0.05) and r["brecha_media_pobres"] == pytest.approx(0.95 / 3)
    assert r["excluidos_cerca_linea"] == pytest.approx(1.0)
    assert r["filtracion"] == pytest.approx(1 / 3)
    assert r["filtracion_ponderada"] == pytest.approx(0.1 / (0.1 + 0.5 + 2.0))
    assert r["distancia_media_incluidos"] == pytest.approx(0.1) and r["incluidos_cerca_linea"] == pytest.approx(1.0)


def test_curva_de_cobertura_crece_hasta_cubrir_a_todos_los_pobres():
    rng = np.random.default_rng(5)
    verdad = (rng.random(500) < 0.3).astype(int)
    probabilidad = np.clip(verdad * 0.6 + rng.random(500) * 0.5, 0, 1)
    curva = curva_cobertura(verdad, probabilidad, puntos=np.arange(0.1, 1.01, 0.1))
    assert list(curva.columns) == ["share_atendida", "share_pobres_cubiertos"]
    assert (np.diff(curva["share_pobres_cubiertos"]) >= 0).all() and curva["share_pobres_cubiertos"].iloc[-1] == 1.0
    perfecta = curva_cobertura(verdad, verdad.astype(float), puntos=np.array([verdad.mean()]))
    assert perfecta["share_pobres_cubiertos"].iloc[0] == pytest.approx(1.0)
    atendida, cubiertos = punto_operacion(verdad, probabilidad, 0.5)
    assert atendida == pytest.approx((probabilidad >= 0.5).mean())
    assert cubiertos == pytest.approx(((probabilidad >= 0.5) & (verdad == 1)).sum() / verdad.sum())


def test_f1_ponderado_coincide_con_replicar_filas():
    verdad = np.array([1, 1, 0, 0, 1])
    prediccion = np.array([1, 0, 1, 0, 1])
    pesos = np.array([1, 2, 3, 1, 1])
    from funciones.metricas import calcular_metricas
    assert f1_ponderado(verdad, prediccion, np.ones(5)) == pytest.approx(calcular_metricas(verdad, prediccion)["f1"])
    replicado = calcular_metricas(np.repeat(verdad, pesos), np.repeat(prediccion, pesos))["f1"]
    assert f1_ponderado(verdad, prediccion, pesos) == pytest.approx(replicado)
    assert f1_ponderado(verdad, prediccion, pesos / pesos.sum()) == pytest.approx(replicado)  # invariante a la escala


def test_ajuste_ebm_devuelve_probabilidades_e_importancia_y_guarda_el_modelo(tmp_path):
    from etapa15_modelos_politica import crear_ajuste_ebm
    x_ent, y_ent = datos_de_juguete(400, 6)
    x_eval, y_eval = datos_de_juguete(100, 7)
    estado = {"contexto": "holdout"}
    ajustar = crear_ajuste_ebm({"outer_bags": 2, "interactions": 1, "max_rounds": 60}, estado, tmp_path)
    probabilidad, importancia = ajustar(x_ent, y_ent, x_eval)
    assert probabilidad.shape == (100,) and ((probabilidad >= 0) & (probabilidad <= 1)).all()
    assert set(importancia.index) <= set(x_ent.columns) and (importancia >= 0).all()
    assert ((probabilidad >= 0.5).astype(int) == y_eval).mean() > 0.75
    assert (tmp_path / "ebm_holdout.joblib").exists()
    estado["contexto"] = "cv_1"
    ajustar(x_ent, y_ent, x_eval)
    assert not (tmp_path / "ebm_cv_1.joblib").exists()  # solo se guardan holdout y test


def test_ensamble_por_fold_detecta_que_no_hay_predicciones_guardadas(tmp_path):
    from etapa15_modelos_politica import predicciones_test_por_fold
    assert predicciones_test_por_fold("n7_a_agregados_bag5", tmp_path) is None
    carpeta = tmp_path / "test_por_fold"
    carpeta.mkdir()
    for k in range(1, 6):
        pd.DataFrame({"probabilidad": [0.1 * k, 0.2 * k]}, index=pd.Index(["t1", "t2"], name="id")).to_parquet(
            carpeta / f"n7_a_agregados_bag5_cv_{k}.parquet")
    tabla = predicciones_test_por_fold("n7_a_agregados_bag5", tmp_path)
    assert list(tabla.columns) == [f"cv_{k}" for k in range(1, 6)] and tabla.loc["t2", "cv_3"] == pytest.approx(0.6)
    shutil.rmtree(carpeta)
