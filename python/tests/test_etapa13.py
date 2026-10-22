"""Pruebas de la etapa 13: la mezcla final solo combina candidatos con test, la
grilla de pesos incluye a la base sola (la mezcla nunca pierde contra ella en OOF)
y holdout y test reciben la misma combinación lineal que la OOF."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from etapa13_mezcla_final import BASE, COMPONENTES, PASO_SIMPLEX, componentes_con_test, grilla_pesos, texto_pesos
from funciones.mezclas import combinar_predicciones


def resultado_de_juguete(nombre, semilla, con_test):
    rng = np.random.default_rng(semilla)
    ids_oof = [f"h{i}" for i in range(20)]
    ids_holdout = [f"v{i}" for i in range(5)]
    salida = {"nombre": nombre,
              "oof": pd.DataFrame({"probabilidad": rng.uniform(size=20), "fold": np.repeat([1, 2, 3, 4, 5], 4),
                                   "verdad": rng.integers(0, 2, 20)}, index=pd.Index(ids_oof, name="id")),
              "prediccion_holdout": pd.Series(rng.uniform(size=5), index=pd.Index(ids_holdout, name="id"))}
    if con_test:
        salida["prediccion_test"] = pd.Series(rng.uniform(size=3), index=pd.Index(["t0", "t1", "t2"], name="id"))
    return salida


def test_componentes_con_test_deja_fuera_a_los_que_no_tienen_prediccion_de_test():
    resultados = {"a": resultado_de_juguete("a", 0, True), "b": resultado_de_juguete("b", 1, False),
                  "c": resultado_de_juguete("c", 2, True)}
    assert componentes_con_test(resultados) == ["a", "c"]


def test_la_base_encabeza_los_componentes_y_todos_tienen_descripcion():
    assert list(COMPONENTES)[0] == BASE
    assert all(isinstance(d, str) and d for d in COMPONENTES.values())


def test_grilla_de_pesos_es_un_simplex_que_incluye_a_la_base_sola():
    for n in (3, 4):
        grilla = grilla_pesos(n)
        assert all(len(w) == n and w.min() >= 0 and w.sum() == pytest.approx(1) for w in grilla)
        assert any(np.array_equal(w, np.eye(n)[0]) for w in grilla), "la base sola debe estar en la grilla"
        assert len({tuple(np.round(w / PASO_SIMPLEX).astype(int)) for w in grilla}) == len(grilla)


def test_texto_pesos_enumera_cada_componente_con_dos_decimales():
    assert texto_pesos(["a", "b", "c"], np.array([0.85, 0.0, 0.15])) == "a 0.85 + b 0.00 + c 0.15"


def test_holdout_y_test_reciben_la_misma_combinacion_que_la_oof():
    resultados = [resultado_de_juguete("a", 0, True), resultado_de_juguete("b", 1, True)]
    pesos = np.array([0.7, 0.3])
    mezcla = combinar_predicciones("m", resultados, pesos)
    for clave, serie_de in (("oof", lambda r: r["oof"]["probabilidad"]),
                            ("prediccion_holdout", lambda r: r["prediccion_holdout"]),
                            ("prediccion_test", lambda r: r["prediccion_test"])):
        esperado = 0.7 * serie_de(resultados[0]) + 0.3 * serie_de(resultados[1])
        obtenido = mezcla[clave]["probabilidad"] if clave == "oof" else mezcla[clave]
        assert np.allclose(obtenido.to_numpy(), esperado.to_numpy())
    assert (mezcla["oof"]["fold"] == resultados[0]["oof"]["fold"]).all()
