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


def test_reanudar_desde_parciales_no_cambia_ni_filtra(tmp_path):
    """Una corrida cortada y retomada desde parciales/ da lo mismo que una entera,
    y el auxiliar espía sigue sin predecir hogares que vio."""
    grupos = asignar_grupos(particion_sintetica())
    llamadas = []

    def espia(ids_entrenamiento, ids_prediccion):
        llamadas.append(1)
        if len(llamadas) == 5:
            raise KeyboardInterrupt("corte simulado")
        vistos = set(ids_entrenamiento)
        return pd.DataFrame({"visto": [float(i in vistos) for i in ids_prediccion]},
                            index=pd.Index(ids_prediccion, name="id"))

    with pytest.raises(KeyboardInterrupt):
        predecir_anidado(espia, grupos, None, ruta_parcial=tmp_path)
    assert len(list(tmp_path.glob("*.parquet"))) == 4
    assert not list(tmp_path.glob("*.tmp")), "quedó un parcial a medio escribir"

    variables, registro = predecir_anidado(espia, grupos, None, ruta_parcial=tmp_path)
    completas, _ = predecir_anidado(espia, grupos, None)
    for contexto, tabla in variables.items():
        assert (tabla["visto"] == 0).all()
        pd.testing.assert_frame_equal(tabla, completas[contexto])
    assert verificar_sin_fuga(registro, grupos)


def test_contexto_de_hogar_del_auxiliar_v3_no_trae_ingresos():
    """Las variables de hogar que se pegan a las personas en el auxiliar v3 no
    incluyen ninguna columna de ingreso (que solo existe en train)."""
    from funciones.datos import cargar_hogares, cargar_personas
    from funciones.variables import construir_grupos
    try:
        personas, hogar = cargar_personas(), cargar_hogares()
    except FileNotFoundError:
        pytest.skip("faltan las bases procesadas del nivel 7")
    bloques = construir_grupos(personas, hogar)
    columnas = set(bloques["nivel_3"].columns) | set(bloques["laborales"].columns)
    ingresos = set(pd.read_parquet(RUTA_NIVEL7.parent / "ingreso_personas_train.parquet").columns)
    ingresos |= set(pd.read_parquet(RUTA_NIVEL7.parent / "ingreso_hogar_train.parquet").columns)
    ingresos -= {"id", "Orden"}
    assert not columnas & ingresos, f"columnas de ingreso en el contexto: {columnas & ingresos}"
    assert not [c for c in columnas if c.lower().startswith(("ing", "log_ingreso"))]


@pytest.mark.parametrize("nombre", ["ingreso_personas", "ingreso_hogar", "ingreso_total_personas",
                                    "ingreso_fuentes_v3", "ingreso_cuantiles_v3"])
def test_registro_real_sin_fuga(nombre):
    ruta = RUTA_NIVEL7 / f"registro_{nombre}.parquet"
    if not ruta.exists():
        pytest.skip(f"{ruta.name} todavía no existe (corre etapa1_ablacion.py)")
    assert verificar_sin_fuga(pd.read_parquet(ruta), asignar_grupos(cargar_particion()))
