"""Etapa 2, variantes baratas sobre a4 con los hiperparámetros de n6:
  v1  oficio del perceptor (2 dígitos) como categórica nativa de LightGBM en
      vez de dummies (máximo 10 min)
  v2  auxiliar de ingreso individual TOTAL (Ingtot), en reemplazo y junto al
      laboral, con la misma OOF anidada y el mismo chequeo de fuga (máximo 15 min)

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa2_variantes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import PARAMETROS_N6, RUTA_TABLAS
from funciones.bases import armar_matriz, cargar_insumos
from funciones.bitacora import Presupuesto, registrar_log
from funciones.modelado import evaluar_candidato, f1_subgrupos

NATIVAS_V1 = ["perceptor_oficio"]


def fila(resultado, descripcion, matriz, dinamicas, insumos):
    """Fila de la tabla de variantes."""
    n_columnas = matriz.shape[1] + sum(next(iter(d.values())).shape[1] for d in dinamicas)
    return {"candidato": resultado["nombre"], "descripcion": descripcion, "n_columnas": n_columnas,
            "umbral": resultado["umbral"], "f1_cv": resultado["f1_cv"], "f1_cv_sd": resultado["f1_cv_sd"],
            "f1_holdout": resultado["f1_holdout"],
            **f1_subgrupos(resultado, insumos["hogar"], insumos["particion"])}


def main():
    insumos = cargar_insumos(auxiliar_total=False)
    particion, laboral = insumos["particion"], insumos["auxiliares"]["laboral"]
    filas = []
    ruta = RUTA_TABLAS / "variantes_etapa2.csv"

    # v1: oficio nativo
    reloj = Presupuesto(10)
    matriz_v1 = armar_matriz(insumos, nativas=NATIVAS_V1)
    resultado = evaluar_candidato("n7_v1_oficio_nativo", matriz_v1, particion, PARAMETROS_N6, [laboral])
    filas.append(fila(resultado, "a4 con perceptor_oficio nativo", matriz_v1, [laboral], insumos))
    pd.DataFrame(filas).to_csv(ruta, index=False)
    registrar_log("etapa2_variantes", f"v1 en {reloj.transcurrido():.1f} min")

    # v2: auxiliar de ingreso total (la OOF anidada verifica la fuga al cargar)
    reloj = Presupuesto(15)
    insumos = cargar_insumos(auxiliar_total=True)
    total = insumos["auxiliares"]["total"]
    matriz = armar_matriz(insumos)
    for nombre, descripcion, dinamicas in (
        ("n7_v2_total_reemplaza", "a4 con auxiliar total en vez del laboral", [total]),
        ("n7_v2_total_suma", "a4 con auxiliar laboral + auxiliar total", [laboral, total]),
    ):
        if reloj.agotado():
            registrar_log("etapa2_variantes", f"tiempo agotado antes de {nombre}")
            break
        resultado = evaluar_candidato(nombre, matriz, particion, PARAMETROS_N6, dinamicas)
        filas.append(fila(resultado, descripcion, matriz, dinamicas, insumos))
        pd.DataFrame(filas).to_csv(ruta, index=False)
    registrar_log("etapa2_variantes", f"v2 en {reloj.transcurrido():.1f} min")
    print(pd.DataFrame(filas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
