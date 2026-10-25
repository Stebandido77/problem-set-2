"""Etapa 2, paso 0: CSV checkpoint de a4 (hiperparámetros de n6, umbral OOF,
reajuste sobre el 100% del train) y chequeo de que test se parezca a OOF/holdout.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa2_checkpoint.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import PARAMETROS_N6, RUTA_TABLAS
from funciones.bases import armar_matriz, cargar_insumos
from funciones.bitacora import registrar_log
from funciones.modelado import evaluar_candidato, f1_subgrupos
from funciones.submission import guardar_submission

VARIABLE_CLAVE = "log_ingreso_pred_pc_lp"


def describir(serie, nombre):
    """Media y percentiles 10/50/90 de una variable en una muestra."""
    return {"muestra": nombre, "n": serie.notna().sum(), "media": serie.mean(),
            "p10": serie.quantile(0.1), "p50": serie.quantile(0.5), "p90": serie.quantile(0.9)}


def main():
    insumos = cargar_insumos()
    particion, grupos = insumos["particion"], insumos["grupos"]
    aux = insumos["auxiliares"]["laboral"]

    resultado = evaluar_candidato("n7_a4_checkpoint", armar_matriz(insumos), particion,
                                  PARAMETROS_N6, [aux], con_test=True)
    ids_test = particion.loc[particion["base"] == "test", "id"]
    ruta, tasa_test = guardar_submission(
        resultado, "checkpoint_a4.csv", ids_test,
        "LightGBM (hiperparámetros de n6) | nivel 3 + v2 (laborales, perceptor, pequeños) "
        "+ auxiliar de ingreso laboral con OOF anidada | umbral OOF")
    registrar_log("n7_a4_checkpoint", f"submission {ruta.name} tasa {tasa_test:.4f}")

    # Distribución de la variable clave en cada muestra, tal como la ve el modelo
    holdout, test = aux["holdout"], aux["test"]
    distribucion = pd.DataFrame([
        describir(holdout.loc[grupos.reindex(holdout.index).between(1, 5).to_numpy(), VARIABLE_CLAVE],
                  "OOF del 80% (entrenamiento para holdout)"),
        describir(holdout.loc[(grupos.reindex(holdout.index) == 6).to_numpy(), VARIABLE_CLAVE], "holdout"),
        describir(test.loc[grupos.reindex(test.index).between(1, 6).to_numpy(), VARIABLE_CLAVE],
                  "OOF del 100% (entrenamiento para test)"),
        describir(test.loc[(grupos.reindex(test.index) == 0).to_numpy(), VARIABLE_CLAVE], "test"),
    ])
    umbral = resultado["umbral"]
    tasas = pd.DataFrame([
        {"muestra": "OOF", "tasa_pobres_predicha": (resultado["oof"]["probabilidad"] >= umbral).mean(),
         "probabilidad_media": resultado["oof"]["probabilidad"].mean()},
        {"muestra": "holdout", "tasa_pobres_predicha": (resultado["prediccion_holdout"] >= umbral).mean(),
         "probabilidad_media": resultado["prediccion_holdout"].mean()},
        {"muestra": "test", "tasa_pobres_predicha": tasa_test,
         "probabilidad_media": resultado["prediccion_test"].mean()},
    ])
    distribucion.to_csv(RUTA_TABLAS / "checkpoint_a4_distribucion.csv", index=False)
    tasas.to_csv(RUTA_TABLAS / "checkpoint_a4_tasas.csv", index=False)

    print(distribucion.round(4).to_string(index=False))
    print(tasas.round(4).to_string(index=False))
    print({k: round(v, 4) for k, v in f1_subgrupos(resultado, insumos["hogar"], particion).items()})
    print(f"umbral {umbral:.4f} | F1 CV {resultado['f1_cv']:.4f} | holdout {resultado['f1_holdout']:.4f}")


if __name__ == "__main__":
    main()
