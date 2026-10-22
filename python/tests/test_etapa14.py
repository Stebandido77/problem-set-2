"""Pruebas de la etapa 14: la aptitud F1 de las fórmulas ignora las filas fuera de
la submuestra y toma la mejor orientación, las fórmulas se ajustan solo con el
entrenamiento del contexto, la colonia funciona sin máscara binaria, la
decodificación de RealMLP respeta los rangos, XGBoost se ajusta con el protocolo y
la regla del CSV exige superar a la base en CV y en holdout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from configuracion import PARAMETROS_N6_XGBOOST
from etapa14_busquedas import (CONFIGURACION_REALMLP_ETAPA10, codificar_abc, decodificar_abc,
                               supera_en_ambos)
from funciones.abejas import ColoniaAbejas
from funciones.formulas import aptitud_f1_umbral, crear_transformador, formular_contexto, formulas_de
from funciones.modelado import crear_ajuste_xgboost


def test_aptitud_f1_ignora_filas_sin_peso_y_toma_la_mejor_orientacion():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    # Las seis primeras filas separan perfecto con la orientación invertida (bajo = pobre);
    # las dos últimas, con peso 0, arruinarían el F1 si se tuvieran en cuenta
    y_pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.95, 0.05])
    w = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
    assert aptitud_f1_umbral(y, y_pred, w) == pytest.approx(1.0)
    assert aptitud_f1_umbral(y, -y_pred, w) == pytest.approx(1.0)
    assert aptitud_f1_umbral(y, y_pred, np.ones(8)) < 1.0
    assert aptitud_f1_umbral(y, np.full(8, 0.3), w) == 0.0


def datos_de_juguete(n, semilla):
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=["ingreso", "edad", "cuartos", "ruido"],
                     index=pd.Index([f"h{semilla}_{i}" for i in range(n)], name="id"))
    y = ((x["ingreso"] * x["edad"] + x["cuartos"]) < 0).astype(int).to_numpy()
    return x, y


def test_formulas_se_ajustan_con_el_entrenamiento_y_se_aplican_a_ambas_matrices():
    x_ent, y_ent = datos_de_juguete(400, 0)
    x_eval, _ = datos_de_juguete(120, 1)
    transformador = crear_transformador(list(x_ent.columns), n_formulas=3, poblacion=100, generaciones=2,
                                        n_jobs=1, fraccion_filas=0.5)
    tabla, formulas = formular_contexto(x_ent, y_ent, x_eval, transformador)
    assert list(tabla.columns) == ["formula_1", "formula_2", "formula_3"]
    assert tabla.index.equals(x_ent.index.append(x_eval.index))
    assert np.isfinite(tabla.to_numpy()).all()
    assert len(formulas) == 3 and formulas == formulas_de(transformador)
    assert all(0 <= f["aptitud"] <= 1 and f["longitud"] >= 1 for f in formulas)
    assert any(any(c in f["formula"] for c in x_ent.columns) for f in formulas)


def test_cli_de_formulas_corre_en_un_proceso_aparte_con_varios_workers(tmp_path):
    import subprocess
    x_ent, y_ent = datos_de_juguete(300, 4)
    x_eval, _ = datos_de_juguete(80, 5)
    x_ent.assign(pobre=y_ent).to_parquet(tmp_path / "ent.parquet")
    x_eval.to_parquet(tmp_path / "eval.parquet")
    comando = [sys.executable, "-m", "funciones.formulas", "--entrenamiento", str(tmp_path / "ent.parquet"),
               "--evaluacion", str(tmp_path / "eval.parquet"), "--salida", str(tmp_path / "formulas.parquet"),
               "--formulas", str(tmp_path / "formulas.json"), "--n-formulas", "2", "--poblacion", "60",
               "--generaciones", "2", "--fraccion-filas", "0.5", "--semilla", "1", "--n-jobs", "2",
               "--columnas", *x_ent.columns]
    proceso = subprocess.run(comando, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proceso.returncode == 0, proceso.stderr[-800:]
    tabla = pd.read_parquet(tmp_path / "formulas.parquet")
    assert tabla.shape == (380, 2) and list(tabla.columns) == ["formula_1", "formula_2"]
    assert tabla.index.equals(x_ent.index.append(x_eval.index))
    import json
    formulas = json.loads((tmp_path / "formulas.json").read_text(encoding="utf-8"))
    assert len(formulas) == 2 and all({"formula", "aptitud", "longitud"} <= set(f) for f in formulas)


def test_decodificar_abc_cubre_los_extremos_y_recupera_la_configuracion_de_la_etapa_10():
    minimo = decodificar_abc(np.zeros(6))
    assert minimo["hidden_sizes"] == [128, 128] and minimo["p_drop"] == 0.0
    assert minimo["lr"] == pytest.approx(0.005) and minimo["wd"] == pytest.approx(1e-4)
    assert minimo["num_emb_type"] == "none"
    maximo = decodificar_abc(np.ones(6))
    assert maximo["hidden_sizes"] == [512] * 4 and maximo["p_drop"] == pytest.approx(0.4)
    assert maximo["lr"] == pytest.approx(0.08) and maximo["wd"] == pytest.approx(0.05)
    assert maximo["num_emb_type"] == "pbld"
    z = codificar_abc(CONFIGURACION_REALMLP_ETAPA10)
    assert ((0 <= z) & (z <= 1)).all()
    recuperada = decodificar_abc(z)
    for clave, valor in CONFIGURACION_REALMLP_ETAPA10.items():
        assert recuperada[clave] == pytest.approx(valor) if isinstance(valor, float) else recuperada[clave] == valor


def test_colonia_sin_mascara_binaria_se_mueve_solo_en_los_continuos(tmp_path):
    def aptitud(z):
        return float(-np.abs(np.asarray(z) - 0.3).sum()), {}

    colonia = ColoniaAbejas(aptitud, n_binarias=0, n_continuas=3, n_fuentes=3, limite=2, semilla=4,
                            ruta=tmp_path / "abc", iniciales=[np.array([0.9, 0.9, 0.9])], registrar=lambda _: None)
    inicial = colonia.mejor()[1]
    for _ in range(4):
        colonia.correr_ciclo()
    assert colonia.fuentes.shape == (3, 3) and ((colonia.fuentes >= 0) & (colonia.fuentes <= 1)).all()
    assert colonia.mejor()[1] >= inicial and len(colonia.mejor()[0]) == 3


def test_ajuste_xgboost_devuelve_probabilidades_e_importancia():
    x_ent, y_ent = datos_de_juguete(600, 2)
    x_eval, y_eval = datos_de_juguete(200, 3)
    ajustar = crear_ajuste_xgboost({**PARAMETROS_N6_XGBOOST, "num_boost_round": 30, "learning_rate": 0.3},
                                   semillas=(1, 2))
    probabilidad, importancia = ajustar(x_ent, y_ent, x_eval)
    assert probabilidad.shape == (200,) and ((probabilidad >= 0) & (probabilidad <= 1)).all()
    assert set(importancia.index) <= set(x_ent.columns) and (importancia >= 0).all()
    assert ((probabilidad >= 0.5).astype(int) == y_eval).mean() > 0.8


def test_regla_csv_exige_superar_a_la_base_en_cv_y_holdout():
    base = {"f1_cv": 0.7753, "f1_holdout": 0.7820}
    assert supera_en_ambos({"f1_cv": 0.7754, "f1_holdout": 0.7821}, base)
    assert not supera_en_ambos({"f1_cv": 0.7760, "f1_holdout": 0.7820}, base)
    assert not supera_en_ambos({"f1_cv": 0.7753, "f1_holdout": 0.7830}, base)


# ---- Cierre sin test (--sin-test) y carga perezosa de la aptitud ABC -----------

def contextos_de_juguete(tmp_path, monkeypatch, nombre):
    """Partición de juguete con los parciales de los 5 folds y del holdout de un
    candidato (como los deja un proceso que murió antes del contexto test)."""
    import etapa14_busquedas as etapa14
    from funciones import modelado
    monkeypatch.setattr(modelado, "RUTA_NIVEL7", tmp_path)
    monkeypatch.setattr(modelado, "registrar_log", lambda *a, **k: None)
    monkeypatch.setattr(etapa14, "RUTA_NIVEL7", tmp_path)
    monkeypatch.setattr(etapa14, "registrar_log", lambda *a, **k: None)
    monkeypatch.setitem(etapa14.MODO, "carpeta", tmp_path / "etapa14")
    rng = np.random.default_rng(0)
    ids = {f"cv_{k}": [f"h{k}_{i}" for i in range(20)] for k in range(1, 6)}
    ids["holdout"] = [f"g{i}" for i in range(50)]
    ids["test"] = [f"t{i}" for i in range(30)]
    particion = pd.DataFrame({"id": sum(ids.values(), []), "base": ["train"] * 150 + ["test"] * 30,
                              "pobre": list(rng.integers(0, 2, 150)) + [pd.NA] * 30})
    carpeta = tmp_path / "parciales" / nombre
    carpeta.mkdir(parents=True)
    for contexto in list(ids)[:-1]:
        (pd.Series(rng.random(len(ids[contexto])), index=pd.Index(ids[contexto], name="id"), name="probabilidad")
         .to_frame().to_parquet(carpeta / f"{contexto}.parquet"))
    from funciones.bitacora import Presupuesto
    entorno = {"particion": particion, "verdad": particion.set_index("id")["pobre"], "reloj": Presupuesto(60),
               "etapa": "prueba"}
    return etapa14, entorno, ids


def test_sin_test_cierra_un_candidato_desde_sus_parciales_sin_entrenar(tmp_path, monkeypatch):
    etapa14, entorno, _ = contextos_de_juguete(tmp_path, monkeypatch, "n14_realmlp_abc_s0")
    monkeypatch.setitem(etapa14.MODO, "con_test", False)
    llamadas = []
    terminadas = etapa14.correr_semillas("n14_realmlp_abc", lambda s: lambda *a: llamadas.append(("ajustar", s)),
                                         lambda c: llamadas.append(("preparar", c)), entorno)
    assert terminadas == ["n14_realmlp_abc_s0"] and llamadas == []
    predicciones = pd.read_parquet(tmp_path / "predicciones_n14_realmlp_abc_s0.parquet")
    assert set(predicciones["tipo"]) == {"oof", "holdout"} and len(predicciones) == 150
    assert (tmp_path / "parciales" / "n14_realmlp_abc_s0" / "holdout.parquet").exists()
    assert np.isfinite(etapa14.leer_json("tiempos")["n14_realmlp_abc_s0"])
    assert not (tmp_path / "predicciones_n14_realmlp_abc_s1.parquet").exists()


def test_con_test_completa_solo_el_test_de_un_candidato_cerrado_sin_test(tmp_path, monkeypatch):
    etapa14, entorno, ids = contextos_de_juguete(tmp_path, monkeypatch, "n14_realmlp_abc_s0")
    monkeypatch.setitem(etapa14.MODO, "con_test", False)
    etapa14.correr_semillas("n14_realmlp_abc", lambda s: None, lambda c: None, entorno, objetivo=1)
    monkeypatch.setitem(etapa14.MODO, "con_test", True)
    contextos = []

    def preparar(contexto):
        contextos.append(contexto)
        x = pd.DataFrame({"v": np.arange(len(ids[contexto]))}, index=pd.Index(ids[contexto], name="id"))
        return x, np.zeros(len(x)), x

    terminadas = etapa14.correr_semillas("n14_realmlp_abc", lambda s: lambda x_ent, y_ent, x_eval: (
        np.full(len(x_eval), 0.5), None), preparar, entorno, objetivo=1)
    assert terminadas == ["n14_realmlp_abc_s0"] and contextos == ["test"]
    predicciones = pd.read_parquet(tmp_path / "predicciones_n14_realmlp_abc_s0.parquet")
    assert set(predicciones["tipo"]) == {"oof", "holdout", "test"} and len(predicciones) == 180
    assert not (tmp_path / "parciales" / "n14_realmlp_abc_s0").exists()


def test_aptitud_abc_carga_los_contextos_solo_cuando_evalua(monkeypatch):
    import etapa14_busquedas as etapa14
    monkeypatch.setitem(etapa14.MODO, "fijos_realmlp", {"n_epochs": 1, "batch_size": 64})
    x_ent, y_ent = datos_de_juguete(240, 6)
    x_eval, y_eval = datos_de_juguete(60, 7)
    cargas = []

    def cargar():
        cargas.append(1)
        return {"cv_1": (x_ent, y_ent, x_eval)}

    aptitud = etapa14.crear_aptitud_abc(cargar, pd.Series(y_eval, index=x_eval.index))
    assert cargas == []
    valor, detalle = aptitud(etapa14.codificar_abc(CONFIGURACION_REALMLP_ETAPA10))
    assert cargas == [1] and 0 <= valor <= 1 and detalle["capas"] == 3 and "f1_cv_1" in detalle
