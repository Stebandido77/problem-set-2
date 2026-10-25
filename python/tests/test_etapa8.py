"""Pruebas de la etapa 8: la búsqueda de hiperparámetros de los auxiliares nunca
ve la evaluación de su contexto, la OOF anidada por grupo de parámetros no filtra
y la grilla de pesos del promedio es un simplex."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import funciones.anidado as anidado
from etapa8_cierre import grilla_simplex
from funciones.anidado import asignar_grupos, definir_contextos, verificar_sin_fuga
from funciones.tuneo_auxiliar import (definir_grupos_tuneo, grupo_de_contexto, particion_interna,
                                      verificar_tuneo_sin_fuga)
from test_sin_fuga import particion_sintetica


def test_busqueda_de_cada_contexto_solo_usa_su_entrenamiento():
    assert verificar_tuneo_sin_fuga()
    contextos = definir_contextos()
    for k in range(1, 6):
        assert k not in definir_grupos_tuneo()[grupo_de_contexto(f"cv_{k}")]
    for contexto in ("holdout", "test"):
        assert 6 not in definir_grupos_tuneo()[grupo_de_contexto(contexto)]
    # La validación interna de la búsqueda es un fold del entrenamiento del grupo
    for grupo, folds in definir_grupos_tuneo().items():
        entrenamiento, validacion = particion_interna(folds)
        assert validacion in folds and validacion not in entrenamiento
        assert set(entrenamiento) | {validacion} == set(folds)
    assert set(contextos) == {"cv_1", "cv_2", "cv_3", "cv_4", "cv_5", "holdout", "test"}


def test_verificador_detecta_busqueda_con_fuga():
    contextos = definir_contextos()
    contextos["cv_1"] = ((1, 3, 4, 5), (2,))  # evalúa en el fold 2, que la búsqueda de cv_1 usa
    with pytest.raises(AssertionError):
        verificar_tuneo_sin_fuga(contextos)


def test_anidado_por_grupo_no_filtra_y_usa_el_auxiliar_de_su_grupo(tmp_path, monkeypatch):
    """El espía devuelve si el hogar estuvo en su entrenamiento y con qué grupo de
    parámetros se construyó; ningún hogar fue visto y cada contexto usa el suyo."""
    monkeypatch.setattr(anidado, "RUTA_NIVEL7", tmp_path)
    grupos = asignar_grupos(particion_sintetica())
    numero = {g: i for i, g in enumerate(sorted(definir_grupos_tuneo()))}

    def crear_ajuste(grupo):
        def espia(ids_entrenamiento, ids_prediccion):
            vistos = set(ids_entrenamiento)
            return pd.DataFrame({"visto": [float(i in vistos) for i in ids_prediccion],
                                 "grupo_parametros": float(numero[grupo])},
                                index=pd.Index(ids_prediccion, name="id"))
        return espia

    variables = anidado.cargar_o_calcular_anidado_por_grupo("espia", crear_ajuste, grupos, grupo_de_contexto)
    for contexto, tabla in variables.items():
        assert (tabla["visto"] == 0).all(), contexto
        assert (tabla["grupo_parametros"] == numero[grupo_de_contexto(contexto)]).all(), contexto
    registro = pd.read_parquet(tmp_path / "registro_espia.parquet")
    assert verificar_sin_fuga(registro, grupos)
    assert not (tmp_path / "parciales" / "espia").exists()


def test_grilla_simplex():
    grilla = grilla_simplex(4, 0.1)
    assert len(grilla) == 286
    assert all(np.isclose(w.sum(), 1) and (w >= 0).all() for w in grilla)
