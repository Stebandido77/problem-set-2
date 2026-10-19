"""Pruebas de la etapa 7: agregados por rol, ausencia de fuga en la preparación
de contextos y en el filtro de agregados, y la colonia de abejas."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from funciones.abejas import ColoniaAbejas
from funciones.agregados import agregar_por_rol, asignar_rol
from funciones.anidado import definir_contextos
from funciones.modelado import crear_preparador, en_grupos
from test_sin_fuga import particion_sintetica


def test_agregados_por_rol_calculan_lo_esperado():
    """Hogar a: jefe 40 años, cónyuge 35, hijo 10; hogar b: jefe solo de 70."""
    ids = pd.Series(["a", "a", "a", "b"])
    edad = pd.DataFrame({"P6040": [40.0, 35.0, 10.0, 70.0]})
    sexo = pd.DataFrame({"P6020_1": [1.0, 0.0, 1.0, 0.0], "P6020_2": [0.0, 1.0, 0.0, 1.0]})
    rol = asignar_rol(np.array([1, 2, 3, 1]))
    hogares = pd.Index(["a", "b"])

    jefe = agregar_por_rol(edad[rol == "jefe"], sexo[rol == "jefe"], ids[rol == "jefe"], "jefe", hogares)
    assert jefe.loc["a", "ag_jefe_P6040_valor"] == 40 and jefe.loc["b", "ag_jefe_P6020_2"] == 1

    todos = agregar_por_rol(edad, sexo, ids, "todos", hogares)
    assert todos.loc["a", "ag_todos_P6040_media"] == 85 / 3
    assert todos.loc["a", "ag_todos_P6040_suma"] == 85 and todos.loc["b", "ag_todos_P6040_desv"] == 0
    assert todos.loc["a", "ag_todos_P6020_1_conteo"] == 2
    assert np.isclose(todos.loc["a", "ag_todos_P6020_1_prop"], 2 / 3)

    filas = rol == "resto"
    resto = agregar_por_rol(edad[filas], sexo[filas], ids[filas], "resto", hogares)
    assert resto.loc["a", "ag_resto_P6040_max"] == 10
    assert resto.loc["b", "ag_resto_P6020_1_conteo"] == 0          # sin resto: conteo 0
    assert np.isnan(resto.loc["b", "ag_resto_P6020_1_prop"])       # y proporción sin definir


def matriz_sintetica(particion):
    """Matriz de juguete indexada por id, con una columna base y un agregado."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"base": rng.random(len(particion)), "ag_todos_x_media": rng.random(len(particion))},
                        index=pd.Index(particion["id"], name="id"))


def test_preparador_nunca_mezcla_entrenamiento_y_evaluacion():
    particion = particion_sintetica()
    preparar = crear_preparador(matriz_sintetica(particion), particion)
    tabla = particion.set_index("id")
    for contexto, (entrenamiento, evaluacion) in definir_contextos().items():
        x_ent, _, x_eval = preparar(contexto)
        assert not set(x_ent.index) & set(x_eval.index), contexto
        assert set(x_ent.index) == set(tabla.index[en_grupos(tabla, entrenamiento)])
        assert set(x_eval.index) == set(tabla.index[en_grupos(tabla, evaluacion)])
        assert not (tabla.loc[x_ent.index, "base"] == "test").any(), "el test entra a un entrenamiento"


def test_filtro_de_agregados_solo_ve_el_entrenamiento(monkeypatch):
    import etapa7_exploracion as etapa7

    particion = particion_sintetica()
    vistos = {}

    def espia(contexto, x_ent, y_ent, k):
        vistos[contexto] = set(x_ent.index)
        return ["ag_todos_x_media"]

    monkeypatch.setattr(etapa7, "agregados_filtrados", espia)
    preparar = etapa7.crear_filtro(crear_preparador(matriz_sintetica(particion), particion), k=1)
    for contexto in definir_contextos():
        _, _, x_eval = preparar(contexto)
        assert not vistos[contexto] & set(x_eval.index), f"el filtro vio la evaluación en {contexto}"
        assert list(x_eval.columns) == ["base", "ag_todos_x_media"]


def aptitud_juguete(objetivo):
    """Aptitud = coincidencia con una máscara objetivo + cercanía del parámetro a 0,7."""
    def aptitud(z):
        z = np.asarray(z)
        return float((z[:len(objetivo)] == objetivo).mean() - abs(z[-1] - 0.7)), {}
    return aptitud


def test_colonia_mejora_y_se_reanuda_igual(tmp_path):
    objetivo = (np.arange(12) % 3 == 0).astype(float)
    argumentos = dict(n_binarias=12, n_continuas=1, n_fuentes=4, limite=2, semilla=1,
                      bits_por_movimiento=2, registrar=lambda _: None)

    seguida = ColoniaAbejas(aptitud_juguete(objetivo), ruta=tmp_path / "seguida", **argumentos)
    inicial = seguida.mejor()[1]
    for _ in range(6):
        seguida.correr_ciclo()
    assert seguida.mejor()[1] >= inicial

    cortada = ColoniaAbejas(aptitud_juguete(objetivo), ruta=tmp_path / "cortada", **argumentos)
    for _ in range(4):
        cortada.correr_ciclo()
    retomada = ColoniaAbejas(aptitud_juguete(objetivo), ruta=tmp_path / "cortada", **argumentos)
    assert retomada.ciclo == 4
    for _ in range(2):
        retomada.correr_ciclo()
    np.testing.assert_array_equal(retomada.fuentes, seguida.fuentes)
    assert retomada.mejor()[1] == seguida.mejor()[1]
