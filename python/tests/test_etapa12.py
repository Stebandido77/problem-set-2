"""Pruebas de la etapa 12: resumen de cuantiles, vecinos fuera de fold, base cruda
y red de conjuntos (invariante a permutaciones y sin ver las filas a predecir)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from etapa12_enfoques import (ALFAS, K_VECINOS, VARIABLES_CUANTILES, construir_base_cruda, resumir_cuantiles,
                              tasas_vecinos)
from funciones.conjuntos import Conjuntos, RedConjuntos, crear_ajuste_conjuntos, preparar_personas_conjuntos


def test_resumen_de_cuantiles_interpola_la_probabilidad_bajo_la_linea():
    alfas = np.asarray(ALFAS)
    # Hogar 1: todos los cuantiles sobre 0 -> P = 0; hogar 2: todos bajo 0 -> P = 1;
    # hogar 3: el 0 cae justo en el cuantil 0,5 -> P = 0,5; hogar 4: cuantiles cruzados
    q1 = alfas * 2 + 0.1
    q2 = alfas * 2 - 3
    q3 = alfas - 0.5
    q4 = q3[::-1]
    salida = resumir_cuantiles(np.vstack([q1, q2, q3, q4]))
    assert salida.shape == (4, len(VARIABLES_CUANTILES))
    assert salida[:, 0].tolist() == pytest.approx([0.0, 1.0, 0.5, 0.5])
    assert salida[2, 1] == pytest.approx(0.0) and salida[3, 1] == pytest.approx(0.0)
    assert salida[2, 2] == pytest.approx(0.5) and salida[3, 2] == pytest.approx(0.5)


def test_tasas_de_vecinos_no_usan_la_fila_consultada():
    rng = np.random.default_rng(0)
    x_ref = rng.normal(size=(1000, 3)).astype(np.float32)
    y_ref = (x_ref[:, 0] > 0).astype(int)
    x_consulta = np.array([[3.0, 0, 0], [-3.0, 0, 0]], dtype=np.float32)
    tasas = tasas_vecinos(x_ref, y_ref, x_consulta)
    assert tasas.shape == (2, len(K_VECINOS))
    assert tasas[0, 0] == 1.0 and tasas[1, 0] == 0.0
    assert (tasas[0] >= tasas[1]).all()


def test_base_cruda_por_rol():
    personas = pd.DataFrame({"id": ["a", "a", "a", "b"], "P6050": [1, 2, 3, 1], "P6040": [40, 38, 10, 70],
                             "Oc": [1, 1, np.nan, np.nan], "Pet": [1, 1, 1, 1]})
    columnas = ["clase", "dominio", "departamento", "cuartos", "dormitorios", "tenencia", "n_personas", "personas_ug",
                "linea_pobreza", "arriendo", "P5100", "Li", "Fex_c", "Fex_dpto"]
    hogar = pd.DataFrame({c: [1.0, 2.0] for c in columnas}, index=pd.Index(["a", "b"], name="id"))
    base = construir_base_cruda(personas, hogar)
    assert base.loc["a", "jefe_P6040"] == 40 and base.loc["a", "conyuge_P6040"] == 38
    assert base.loc["a", "resto_P6040_max"] == 10 and base.loc["a", "todos_P6040_media"] == pytest.approx(88 / 3)
    assert np.isnan(base.loc["b", "conyuge_P6040"]) and base.loc["b", "n_conyuge"] == 0
    assert not any(c.startswith(("log_", "prop_", "ingreso")) for c in base.columns)


def conjuntos_de_juguete(n_hogares=300, semilla=0):
    rng = np.random.default_rng(semilla)
    tamanos = rng.integers(1, 6, n_hogares)
    ids = np.repeat([f"h{i}" for i in range(n_hogares)], tamanos)
    personas = pd.DataFrame({"id": ids, "edad": rng.normal(size=len(ids)), "ocupado": rng.integers(0, 2, len(ids)),
                             "posicion": rng.choice(["a", "b"], len(ids)), "P6800": rng.normal(size=len(ids))})
    personas.loc[personas.index[:7], "P6800"] = np.nan
    x_personas, ids_persona = preparar_personas_conjuntos(personas)
    x_hogar = pd.DataFrame({"z": rng.normal(size=n_hogares)}, index=pd.Index([f"h{i}" for i in range(n_hogares)], name="id"))
    y = pd.Series((personas.groupby("id")["ocupado"].mean().reindex(x_hogar.index) < 0.5).astype(int).to_numpy(),
                  index=x_hogar.index)
    return Conjuntos(x_personas, ids_persona, x_hogar), y


def test_red_de_conjuntos_es_invariante_a_permutaciones():
    conjuntos, _ = conjuntos_de_juguete()
    ids = conjuntos.x_hogar.index[:8]
    estadisticas = conjuntos.estadisticas(ids)
    personas, mascara, hogar = conjuntos.lote(ids, *estadisticas)
    modelo = RedConjuntos(personas.shape[-1], hogar.shape[-1], 16, 32, 0.0).eval()
    import torch
    with torch.no_grad():
        salida = modelo(personas, mascara, hogar)
        permutacion = torch.randperm(personas.shape[1])
        salida_permutada = modelo(personas[:, permutacion], mascara[:, permutacion], hogar)
    assert torch.allclose(salida, salida_permutada, atol=1e-5)
    assert mascara.sum(1).tolist() == conjuntos.conteo.loc[ids].tolist()


def test_ajuste_de_conjuntos_aprende_y_solo_predice_los_ids_pedidos():
    from funciones.conjuntos import PARAMETROS_CONJUNTOS
    conjuntos, y = conjuntos_de_juguete(600)
    ids = conjuntos.x_hogar.index
    x_ent, x_eval = pd.DataFrame(index=ids[:450]), pd.DataFrame(index=ids[450:])
    ajustar = crear_ajuste_conjuntos(conjuntos, {**PARAMETROS_CONJUNTOS, "max_epocas": 15, "lote": 64, "d_oculto": 16,
                                                 "d_salida": 32}, semilla=1)
    probabilidad, _ = ajustar(x_ent, y.loc[x_ent.index].to_numpy(), x_eval)
    assert probabilidad.shape == (150,) and np.all((probabilidad >= 0) & (probabilidad <= 1))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y.loc[x_eval.index], probabilidad) > 0.8
