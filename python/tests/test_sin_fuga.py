"""Ningún hogar recibe una predicción de un modelo auxiliar entrenado con él, y
en cada fold de CV ningún auxiliar usado vio ese fold."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from configuracion import RUTA_NIVEL7
from funciones.anidado import asignar_grupos, predecir_anidado, verificar_sin_fuga
from funciones.datos import cargar_particion


def particion_sintetica(n_por_grupo=20):
    """Partición de juguete con 5 folds, holdout y test."""
    filas = []
    for g in range(7):
        for i in range(n_por_grupo):
            filas.append({
                "id": f"h{g}_{i}", "pobre": i % 2 if g else pd.NA,
                "fold_id": g if 1 <= g <= 5 else pd.NA, "holdout": g == 6,
                "base": "test" if g == 0 else "train",
            })
    particion = pd.DataFrame(filas)
    particion["fold_id"] = particion["fold_id"].astype("Int64")
    return particion


def test_auxiliar_espia_nunca_predice_hogares_que_vio():
    """El auxiliar 'espía' devuelve 1 si el hogar a predecir estuvo en su entrenamiento."""
    grupos = asignar_grupos(particion_sintetica())

    def espia(ids_entrenamiento, ids_prediccion):
        vistos = set(ids_entrenamiento)
        return pd.DataFrame({"visto": [float(i in vistos) for i in ids_prediccion]},
                            index=pd.Index(ids_prediccion, name="id"))

    variables, registro = predecir_anidado(espia, grupos, None)
    assert all((v["visto"] == 0).all() for v in variables.values())
    assert verificar_sin_fuga(registro, grupos)


def test_verificador_detecta_una_fuga():
    grupos = asignar_grupos(particion_sintetica())
    _, registro = predecir_anidado(
        lambda ent, pred: pd.DataFrame({"x": np.zeros(len(pred))}, index=pd.Index(pred, name="id")),
        grupos, None)
    registro.loc[registro.index[0], "grupos_modelo"] += f",{registro.loc[registro.index[0], 'grupo']}"
    with pytest.raises(AssertionError):
        verificar_sin_fuga(registro, grupos)


@pytest.mark.parametrize("nombre", ["ingreso_personas", "ingreso_hogar", "ingreso_total_personas"])
def test_registro_real_sin_fuga(nombre):
    ruta = RUTA_NIVEL7 / f"registro_{nombre}.parquet"
    if not ruta.exists():
        pytest.skip(f"{ruta.name} todavía no existe (corre etapa1_ablacion.py)")
    assert verificar_sin_fuga(pd.read_parquet(ruta), asignar_grupos(cargar_particion()))
