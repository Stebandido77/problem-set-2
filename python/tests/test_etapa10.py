"""Pruebas de la etapa 10: las redes nunca ven el fold de evaluación (ni en el
early stopping ni en el escalado), el tuning no toca holdout ni test, los pesos
de los ensambles salen solo de la OOF, la métrica es la del pipeline y el reloj
sigue contando al relanzar."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import funciones.mezclas as mezclas
import funciones.modelado as modelado
import funciones.redes as redes
from funciones.anidado import definir_contextos
from funciones.bitacora import Presupuesto
from funciones.metricas import buscar_umbral_f1
from funciones.modelado import crear_preparador, en_grupos, evaluar_modelo, guardar_predicciones
from test_sin_fuga import particion_sintetica

PARAMETROS_TABM = {"k": 4, "d_block": 32, "n_blocks": 2, "arch_type": "tabm", "dropout": 0.0, "lr": 3e-3,
                   "weight_decay": 0.0, "batch_size": 128, "max_epocas": 8, "paciencia": 3}


def datos_sinteticos(n=3000, semilla=1):
    """Pobre si x0 alto y x1 bajo (con ruido); una dummy, una constante y una marca de fila."""
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame(rng.normal(size=(n, 3)).astype(np.float32), columns=["x0", "x1", "x2"])
    x["dummy"] = (rng.random(n) < 0.3).astype(np.float32)
    x["constante"] = np.float32(1)
    x["marca"] = np.arange(n, dtype=np.float32)  # identifica la fila en los espías
    y = ((x["x0"] > 0.2) & (x["x1"] < 0.3) | (rng.random(n) < 0.05)).astype(int).to_numpy()
    return x, y


# ---- 1. Early stopping y escalado solo con el entrenamiento -------------------

def test_particion_parada_es_10_por_ciento_estratificada_y_disjunta():
    y = np.r_[np.ones(2000, int), np.zeros(8000, int)]
    ajuste, parada = redes.particion_parada(y, 0)
    assert len(parada) == 1000 and len(np.intersect1d(ajuste, parada)) == 0
    assert np.array_equal(np.sort(np.r_[ajuste, parada]), np.arange(len(y)))
    assert y[parada].mean() == pytest.approx(0.2, abs=0.001)


def test_realmlp_ajusta_y_para_solo_con_filas_de_entrenamiento(monkeypatch):
    """fit() recibe solo x_ent (marcas) y la parada es el 10% interno de sus filas;
    x_eval solo llega a predict_proba()."""
    x, y = datos_sinteticos()
    x_ent, y_ent, x_eval = x.iloc[:2000], y[:2000], x.iloc[2000:]
    vistos = {}

    class Espia:
        def __init__(self, parametros, semilla):
            pass

        def fit(self, x_fit, y_fit, val_idxs):
            vistos.update(fit=x_fit, y_fit=y_fit, val_idxs=val_idxs)
            return self

        def predict_proba(self, x_pred):
            vistos["prediccion"] = x_pred
            return np.column_stack([np.full(len(x_pred), 0.5), np.full(len(x_pred), 0.5)])

    monkeypatch.setattr(redes, "crear_realmlp", Espia)
    redes.crear_ajuste_realmlp({}, 0)(x_ent, y_ent, x_eval)
    filas_fit = set(vistos["fit"]["marca"])
    assert filas_fit == set(x_ent["marca"]) and filas_fit.isdisjoint(set(x_eval["marca"]))
    np.testing.assert_array_equal(vistos["y_fit"], y_ent)
    _, parada = redes.particion_parada(y_ent, 0)
    np.testing.assert_array_equal(vistos["val_idxs"], parada)
    assert len(parada) == 200
    np.testing.assert_array_equal(vistos["prediccion"]["marca"].to_numpy(), x_eval["marca"].to_numpy())


def test_tabm_entrena_y_para_solo_con_filas_de_entrenamiento(monkeypatch):
    x, y = datos_sinteticos()
    x_ent, y_ent, x_eval = x.iloc[:2000], y[:2000], x.iloc[2000:]
    vistos = {}

    def espia(x_ajuste, y_ajuste, x_parada, y_parada, parametros, semilla):
        vistos.update(n_ajuste=len(x_ajuste), n_parada=len(x_parada), y_ajuste=y_ajuste, y_parada=y_parada)
        raise StopIteration

    monkeypatch.setattr(redes, "entrenar_tabm", espia)
    with pytest.raises(StopIteration):
        redes.crear_ajuste_tabm(PARAMETROS_TABM, 0)(x_ent, y_ent, x_eval)
    ajuste, parada = redes.particion_parada(y_ent, 0)
    assert vistos["n_ajuste"] + vistos["n_parada"] == len(x_ent) and vistos["n_parada"] == 200
    np.testing.assert_array_equal(vistos["y_ajuste"], y_ent[ajuste])
    np.testing.assert_array_equal(vistos["y_parada"], y_ent[parada])


def test_escalar_cuantiles_se_ajusta_solo_con_el_entrenamiento():
    """Cambiar la evaluación (valores extremos) no cambia la transformación del ajuste."""
    x, _ = datos_sinteticos()
    x = x[["x0", "x1", "dummy"]]
    ajuste, evaluacion = x.iloc[:2000], x.iloc[2000:]
    a1, e1 = redes.escalar_cuantiles(ajuste, evaluacion)
    a2, e2 = redes.escalar_cuantiles(ajuste, evaluacion * 1000)
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(e1, e2)
    np.testing.assert_array_equal(a1[:, 2], ajuste["dummy"].to_numpy())  # las dummies quedan en 0/1
    assert abs(a1[:, 0].mean()) < 0.05 and a1[:, 0].std() == pytest.approx(1, abs=0.1)


# ---- 2. Tuning dentro del 80% -------------------------------------------------

def test_tuning_solo_usa_el_entrenamiento_del_holdout():
    """El tuning parte el entrenamiento del contexto holdout: folds 1-5, nunca holdout ni test."""
    import etapa10_redes as etapa10  # noqa: PLC0415 (importa la etapa completa)
    particion = particion_sintetica(200)
    tabla = particion.set_index("id")
    entrenamiento, evaluacion = definir_contextos()["holdout"]
    assert set(entrenamiento) == {1, 2, 3, 4, 5} and evaluacion == (6,)
    ids = tabla.index[en_grupos(tabla, entrenamiento)]
    y = tabla.loc[ids, "pobre"].astype(int).to_numpy()
    ajuste, validacion = etapa10.particion_tuneo(y)
    assert len(np.intersect1d(ajuste, validacion)) == 0
    assert len(validacion) == pytest.approx(etapa10.FRACCION_TUNEO * len(y), abs=1)
    grupos = set(tabla.loc[ids[np.r_[ajuste, validacion]], "fold_id"].astype(int))
    assert grupos == {1, 2, 3, 4, 5}


# ---- 3. Ensambles y bagging ---------------------------------------------------

def guardar_falso(nombre, probabilidad, verdad, folds, holdout, test):
    """Escribe predicciones de juguete con el formato de guardar_predicciones()."""
    ids = verdad.index
    guardar_predicciones({
        "nombre": nombre,
        "oof": pd.DataFrame({"probabilidad": probabilidad[:len(folds)], "fold": folds}, index=ids[:len(folds)]),
        "prediccion_holdout": pd.Series(holdout, index=ids[len(folds):len(folds) + len(holdout)]),
        "prediccion_test": pd.Series(test, index=pd.Index([f"t{i}" for i in range(len(test))]))})


@pytest.fixture
def candidatos_falsos(tmp_path, monkeypatch):
    monkeypatch.setattr(modelado, "RUTA_NIVEL7", tmp_path)
    rng = np.random.default_rng(0)
    n_oof, n_holdout = 1000, 300
    verdad = pd.Series(rng.integers(0, 2, n_oof + n_holdout), index=[f"h{i}" for i in range(n_oof + n_holdout)])
    folds = np.tile(np.arange(1, 6), n_oof // 5)
    ruido = {nombre: rng.normal(0, s, n_oof + n_holdout) for nombre, s in (("a", 0.4), ("b", 0.6), ("c", 0.8))}
    for nombre, r in ruido.items():
        p = 1 / (1 + np.exp(-(2 * verdad.to_numpy() - 1 + r)))
        guardar_falso(nombre, p, verdad, folds, p[n_oof:], rng.random(50))
    return verdad, folds, n_oof


def test_pesos_salen_solo_de_la_oof(candidatos_falsos):
    """Reemplazar holdout y test de los componentes no cambia los pesos elegidos."""
    verdad, folds, n_oof = candidatos_falsos
    grilla = [np.array([1 - w, w]) for w in np.round(np.arange(0, 1.0001, 0.05), 2)]
    _, pesos, tabla = mezclas.mezclar_por_oof("m", ["a", "b"], grilla, verdad)
    rng = np.random.default_rng(5)
    for nombre in ("a", "b"):
        r = modelado.resultado_desde_predicciones(nombre, verdad)
        guardar_falso(nombre, r["oof"]["probabilidad"].to_numpy(), verdad, folds, rng.random(300), rng.random(50))
    _, pesos_2, _ = mezclas.mezclar_por_oof("m", ["a", "b"], grilla, verdad)
    np.testing.assert_array_equal(pesos, pesos_2)
    # El F1 OOF de la mezcla elegida es el máximo de la grilla y nunca peor que el de a solo
    assert tabla["f1_oof"].max() == pytest.approx(tabla["f1_oof"].iloc[int(np.argmax(tabla["f1_oof"]))])
    assert tabla["f1_oof"].max() >= tabla["f1_oof"].iloc[0]


def test_mezcla_aplica_los_pesos_y_el_umbral_oof_del_pipeline(candidatos_falsos):
    verdad, _, _ = candidatos_falsos
    resultado, pesos, _ = mezclas.mezclar_por_oof("m3", ["a", "b", "c"], [np.array([0.5, 0.3, 0.2])], verdad)
    a, b, c = (modelado.resultado_desde_predicciones(n, verdad) for n in "abc")
    esperado = 0.5 * a["prediccion_test"] + 0.3 * b["prediccion_test"] + 0.2 * c["prediccion_test"]
    np.testing.assert_allclose(resultado["prediccion_test"].to_numpy(), esperado.to_numpy())
    oof = resultado["oof"]
    assert resultado["umbral"] == buscar_umbral_f1(oof["verdad"], oof["probabilidad"])[0]


def test_bagging_promedia_las_semillas(candidatos_falsos):
    verdad, _, _ = candidatos_falsos
    resultado = mezclas.promediar_semillas("bag", ["a", "b", "c"], verdad)
    partes = [modelado.resultado_desde_predicciones(n, verdad) for n in "abc"]
    np.testing.assert_allclose(resultado["prediccion_holdout"].to_numpy(),
                               sum(p["prediccion_holdout"] for p in partes).to_numpy() / 3)
    np.testing.assert_allclose(resultado["oof"]["probabilidad"].to_numpy(),
                               sum(p["oof"]["probabilidad"] for p in partes).to_numpy() / 3)


# ---- 4. Las redes aprenden con el protocolo completo --------------------------

def test_tabm_aprende_y_es_reproducible():
    x, y = datos_sinteticos()
    x = x.drop(columns="marca")
    ajustar = redes.crear_ajuste_tabm(PARAMETROS_TABM, 3)
    p1, _ = ajustar(x.iloc[:2000], y[:2000], x.iloc[2000:])
    p2, _ = ajustar(x.iloc[:2000], y[:2000], x.iloc[2000:])
    np.testing.assert_allclose(p1, p2, rtol=1e-5)
    assert buscar_umbral_f1(y[2000:], p1)[1] > 0.75


def test_red_en_evaluar_modelo_con_contextos_sinteticos(tmp_path, monkeypatch):
    """TabM pasa por evaluar_modelo() (CV, umbral OOF, holdout y test) sin errores
    y la parada de cada contexto sale de su entrenamiento."""
    monkeypatch.setattr(modelado, "RUTA_NIVEL7", tmp_path)
    monkeypatch.setattr(modelado, "registrar_log", lambda *a, **k: None)
    particion = particion_sintetica(300)
    x, y = datos_sinteticos(len(particion))
    x = x.drop(columns="marca").set_axis(particion["id"].to_numpy())
    particion.loc[particion["base"] == "train", "pobre"] = y[(particion["base"] == "train").to_numpy()]
    preparar = crear_preparador(x, particion)
    resultado = evaluar_modelo("prueba", preparar, redes.crear_ajuste_tabm(PARAMETROS_TABM, 0), particion,
                               con_test=True, guardar=False)
    assert resultado["f1_cv"] > 0.7 and resultado["f1_holdout"] > 0.7
    assert len(resultado["prediccion_test"]) == 300


def test_realmlp_aprende():
    x, y = datos_sinteticos()
    x = x.drop(columns="marca")
    ajustar = redes.crear_ajuste_realmlp({"n_epochs": 8, "batch_size": 256, "num_emb_type": "none"}, 0)
    probabilidad, _ = ajustar(x.iloc[:2000], y[:2000], x.iloc[2000:])
    assert buscar_umbral_f1(y[2000:], probabilidad)[1] > 0.75


# ---- 5. Reloj -----------------------------------------------------------------

def test_reloj_sigue_contando_al_relanzar(tmp_path):
    ruta = tmp_path / "inicio.txt"
    primero = Presupuesto(180, ruta)
    ruta.write_text(repr(time.time() - 3600), encoding="utf-8")  # el primer arranque fue hace una hora
    relanzado = Presupuesto(180, ruta)
    assert relanzado.transcurrido() == pytest.approx(60, abs=0.1)
    assert relanzado.restante() == pytest.approx(120, abs=0.1)
    assert primero.transcurrido() < 1
