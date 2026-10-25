"""Submissions del nivel 7: validación (la misma de validar_submission() en R,
más un chequeo de tasa de pobres) y registro en la bitácora de Kaggle."""

from datetime import date

import numpy as np
import pandas as pd

from configuracion import RAIZ, RUTA_SUBMISSIONS

N_FILAS_TEST = 66168
RUTA_BITACORA = RAIZ / "stores" / "submissions" / "bitacora_kaggle.csv"
# Rango aceptable de la tasa de pobres predicha en test (la del train es 0,200)
RANGO_TASA_POBRES = (0.15, 0.32)


def validar_submission(submission, ids_test):
    """Revisa formato y plausibilidad de una submission.

    Argumentos: submission = DataFrame id, pobre; ids_test = ids del test.
    Devuelve: la tasa de pobres predicha (AssertionError si algo no cumple).
    """
    assert list(submission.columns) == ["id", "pobre"], "las columnas deben ser id y pobre"
    assert len(submission) == N_FILAS_TEST, "número de filas distinto al test"
    assert set(submission["id"]) == set(ids_test), "los ids no coinciden con el test"
    assert submission["id"].is_unique, "hay ids duplicados"
    assert not submission.isna().any().any(), "hay NA en la submission"
    assert submission["pobre"].isin([0, 1]).all() and submission["pobre"].dtype.kind == "i", "clase no es entera 0/1"
    tasa = submission["pobre"].mean()
    assert RANGO_TASA_POBRES[0] <= tasa <= RANGO_TASA_POBRES[1], f"tasa de pobres fuera de rango: {tasa:.3f}"
    return tasa


def guardar_submission(resultado, nombre_archivo, ids_test, especificacion, registrar=True,
                       algoritmo="Boosting"):
    """Escribe el CSV de Kaggle de un candidato y lo agrega a la bitácora.

    Argumentos: resultado = evaluar_candidato(con_test=True); nombre_archivo = CSV
        dentro de stores/submissions/nivel7/; ids_test = ids del test;
        especificacion = descripción corta; registrar = agrega fila a la bitácora
        con f1_kaggle vacía (se llena a mano al subir); algoritmo = nombre del
        algoritmo en la bitácora (lista del curso).
    Devuelve: (ruta del CSV, tasa de pobres predicha).
    """
    clases = (resultado["prediccion_test"] >= resultado["umbral"]).astype(np.int64)
    submission = pd.DataFrame({"id": clases.index.astype(str), "pobre": clases.to_numpy()})
    tasa = validar_submission(submission, ids_test)
    RUTA_SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_SUBMISSIONS / nombre_archivo
    submission.to_csv(ruta, index=False)

    if registrar:
        bitacora = pd.read_csv(RUTA_BITACORA, dtype=str)
        fila = {
            "fecha": date.today().isoformat(), "archivo": f"nivel7/{nombre_archivo}",
            "nivel": "7", "candidato": resultado["nombre"], "algoritmo": algoritmo,
            "especificacion": especificacion,
            "f1_cv": f"{resultado['f1_cv']:.4f}", "f1_holdout": f"{resultado['f1_holdout']:.4f}",
        }
        bitacora = bitacora[bitacora["archivo"] != fila["archivo"]]  # no duplica si se regenera
        bitacora = pd.concat([bitacora, pd.DataFrame([fila])], ignore_index=True)
        bitacora.to_csv(RUTA_BITACORA, index=False)
    return ruta, tasa
