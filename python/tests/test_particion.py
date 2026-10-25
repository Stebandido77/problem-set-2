"""La partición que usa Python es exactamente la de R: mismos hogares por fold,
misma tasa de pobreza y mismos folds que las OOF de n6_lightgbm."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from configuracion import RUTA_PROCESADOS
from funciones.datos import cargar_particion


def resumir_particion_python(particion):
    """Conteos por fold/holdout calculados en Python, con el formato de R."""
    train = particion[particion["base"] == "train"].copy()
    train["grupo"] = np.where(train["holdout"], "holdout",
                              "fold_" + train["fold_id"].astype("string"))
    return (train.groupby("grupo")
            .agg(hogares=("id", "size"), pobres=("pobre", "sum"), tasa_pobreza=("pobre", "mean"))
            .reset_index())


def test_conteos_y_tasa_coinciden_con_r():
    python = resumir_particion_python(cargar_particion())
    r = pd.read_csv(RUTA_PROCESADOS / "resumen_particion_r.csv")
    unido = r.merge(python, on="grupo", suffixes=("_r", "_py"), validate="one_to_one")
    assert len(unido) == 6
    assert (unido["hogares_r"] == unido["hogares_py"]).all()
    assert (unido["pobres_r"] == unido["pobres_py"]).all()
    assert np.allclose(unido["tasa_pobreza_r"], unido["tasa_pobreza_py"])


def test_holdout_y_folds_disjuntos():
    particion = cargar_particion()
    train = particion[particion["base"] == "train"]
    assert particion["id"].is_unique
    assert train.loc[train["holdout"], "fold_id"].isna().all()
    assert train.loc[~train["holdout"], "fold_id"].notna().all()
    assert len(particion[particion["base"] == "test"]) == 66168


def test_folds_coinciden_con_oof_de_n6():
    particion = cargar_particion().set_index("id")
    n6 = pd.read_parquet(RUTA_PROCESADOS / "n6_lightgbm_predicciones.parquet")
    oof = n6[n6["tipo"] == "oof"]
    fold_r = oof["fold"].str.replace("Fold", "").astype(int).to_numpy()
    assert (fold_r == particion.loc[oof["id"], "fold_id"].astype(int).to_numpy()).all()
    holdout = n6.loc[n6["tipo"] == "holdout", "id"]
    assert set(holdout) == set(particion.index[particion["holdout"]])
