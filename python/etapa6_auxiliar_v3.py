"""Etapa 6: auxiliar de ingreso v3, priorizando el ingreso TOTAL (la brecha que
mostró el oráculo de la etapa 5).

  1. Dos partes por fuente (laboral, pensiones, arriendos, transferencias, otros):
     P(recibe) x E[ingreso | recibe], a nivel persona con el contexto del hogar,
     agregado a la unidad de gasto. Misma OOF anidada que los auxiliares v2.
  2. Si sobra tiempo: regresión por cuantiles de log(Ingpcug / Lp) por hogar.
Ablación con los hiperparámetros de n6 y bagging de 5 semillas contra
n7_n6param_a4_total_bag5 (0,7760), con el F1 en la banda [0,8; 1,2] de Lp.
Solo se genera CSV si el holdout supera 0,780.

Es reanudable: cada ajuste auxiliar y cada fold de cada candidato se guarda al
terminar en stores/procesados/nivel7/parciales/; si el proceso se corta, volver a
correrlo retoma desde ahí. Los parciales de los candidatos se borran al final.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa6_auxiliar_v3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS
from etapa2_optuna import SEMILLAS_BAGGING
from etapa5_techo import bandas_linea
from funciones.anidado import cargar_o_calcular_anidado
from funciones.auxiliares import crear_auxiliar_cuantiles, crear_auxiliar_fuentes
from funciones.bases import armar_matriz, cargar_insumos
from funciones.bitacora import Presupuesto, registrar_log
from funciones.datos import cargar_ingreso_fuentes, cargar_ingreso_hogar, cargar_personas
from funciones.metricas import buscar_umbral_f1
from funciones.modelado import borrar_parciales, evaluar_candidato, f1_subgrupos
from funciones.submission import guardar_submission

BASE = "n7_n6param_a4_total_bag5"
F1_BASE = 0.7760
F1_MINIMO_CSV = 0.780
MINUTOS = 75
MINUTOS_POR_CANDIDATO = 10   # bag5 con reajuste sobre el 100% (medido en la etapa 2)
BANDA = "dentro [0.8; 1.2]"


def f1_banda(verdad, prob_holdout, umbral):
    """F1 del holdout dentro y fuera de la banda [0,8; 1,2] de Lp."""
    tabla = bandas_linea(verdad.loc[prob_holdout.index].astype(int), prob_holdout, umbral).set_index("banda")
    return {"f1_banda_08_12": tabla.loc[BANDA, "f1"], "f1_fuera_banda_08_12": tabla.loc["fuera [0.8; 1.2]", "f1"]}


def fila_base(verdad):
    """Fila del candidato de referencia desde sus predicciones guardadas."""
    predicciones = pd.read_parquet(RUTA_NIVEL7 / f"predicciones_{BASE}.parquet").set_index("id")
    oof = predicciones[predicciones["tipo"] == "oof"]["probabilidad"]
    holdout = predicciones[predicciones["tipo"] == "holdout"]["probabilidad"]
    umbral, _ = buscar_umbral_f1(verdad.loc[oof.index], oof)
    lider = pd.read_csv(RUTA_TABLAS / "bagging_n6param_a4_total.csv").iloc[0]
    return {"candidato": BASE, "auxiliares": "laboral + total (v2)", "umbral": umbral,
            "f1_cv": lider["f1_cv"], "f1_holdout": lider["f1_holdout"],
            "precision_holdout": lider["precision_holdout"], "recall_holdout": lider["recall_holdout"],
            **f1_banda(verdad, holdout, umbral), "tiempo_min": lider["tiempo_min"]}


def main():
    reloj = Presupuesto(MINUTOS)
    registrar_log("etapa6", "inicio")
    insumos = cargar_insumos(auxiliar_total=True)
    particion = insumos["particion"]
    verdad = particion.set_index("id")["pobre"]
    matriz = armar_matriz(insumos)
    bloques = insumos["bloques"]

    # 1. Dos partes por fuente con el contexto del hogar
    contexto_hogar = pd.concat([bloques["nivel_3"], bloques["laborales"]], axis=1)
    auxiliares = dict(insumos["auxiliares"])
    auxiliares["fuentes"] = cargar_o_calcular_anidado(
        "ingreso_fuentes_v3",
        crear_auxiliar_fuentes(cargar_personas(), cargar_ingreso_fuentes(), insumos["hogar"], contexto_hogar),
        insumos["grupos"])
    registrar_log("etapa6", f"auxiliar de fuentes listo ({reloj.transcurrido():.1f} min)")

    candidatos = [
        ("n7_v3_fuentes_bag5", "laboral + total (v2) + fuentes (v3)", ["laboral", "total", "fuentes"]),
        ("n7_v3_solo_fuentes_bag5", "fuentes (v3)", ["fuentes"]),
    ]
    filas = [fila_base(verdad)]
    resultados = {}

    def correr(nombre, descripcion, usados):
        if reloj.transcurrido() + MINUTOS_POR_CANDIDATO > MINUTOS:
            registrar_log("etapa6", f"sin tiempo para {nombre}")
            return
        resultado = evaluar_candidato(nombre, matriz, particion, PARAMETROS_N6,
                                      [auxiliares[a] for a in usados], con_test=True,
                                      semillas=SEMILLAS_BAGGING, reanudable=True)
        resultados[nombre] = (resultado, descripcion)
        filas.append({"candidato": nombre, "auxiliares": descripcion, "umbral": resultado["umbral"],
                      "f1_cv": resultado["f1_cv"], "f1_holdout": resultado["f1_holdout"],
                      "precision_holdout": resultado["precision_holdout"],
                      "recall_holdout": resultado["recall_holdout"],
                      **f1_banda(verdad, resultado["prediccion_holdout"], resultado["umbral"]),
                      **f1_subgrupos(resultado, insumos["hogar"], particion),
                      "tiempo_min": resultado["tiempo_min"]})
        pd.DataFrame(filas).to_csv(RUTA_TABLAS / "etapa6_ablacion.csv", index=False)

    for candidato in candidatos:
        correr(*candidato)

    # 2. Cuantiles, sobre la mejor combinación de la parte 1
    if reloj.transcurrido() + 5 + MINUTOS_POR_CANDIDATO <= MINUTOS and resultados:
        auxiliares["cuantiles"] = cargar_o_calcular_anidado(
            "ingreso_cuantiles_v3", crear_auxiliar_cuantiles(matriz, cargar_ingreso_hogar()), insumos["grupos"])
        registrar_log("etapa6", f"auxiliar de cuantiles listo ({reloj.transcurrido():.1f} min)")
        mejor = max(resultados, key=lambda n: resultados[n][0]["f1_holdout"])
        usados = dict((c[0], c[2]) for c in candidatos)[mejor]
        correr("n7_v3_cuantiles_bag5", resultados[mejor][1] + " + cuantiles (v3)", usados + ["cuantiles"])
    else:
        registrar_log("etapa6", "sin tiempo para los cuantiles")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_TABLAS / "etapa6_ablacion.csv", index=False)
    print(tabla.round(4).to_string(index=False))

    # 3. CSV solo si supera el corte
    if not resultados:
        registrar_log("etapa6", "ningún candidato alcanzó a correr: sin CSV")
        return
    mejor = max(resultados, key=lambda n: resultados[n][0]["f1_holdout"])
    resultado, descripcion = resultados[mejor]
    if resultado["f1_holdout"] > F1_MINIMO_CSV:
        ids_test = particion.loc[particion["base"] == "test", "id"]
        ruta, tasa = guardar_submission(
            resultado, f"{mejor}.csv", ids_test,
            f"LightGBM (hiperparámetros de n6, bagging 5 semillas) | a4 + auxiliares {descripcion} "
            "con OOF anidada | umbral OOF")
        registrar_log("etapa6", f"submission {ruta.name} tasa {tasa:.4f}")
    else:
        registrar_log("etapa6", f"mejor holdout {resultado['f1_holdout']:.4f} <= {F1_MINIMO_CSV}: sin CSV")
    for nombre in resultados:
        borrar_parciales(nombre)
    registrar_log("etapa6", f"fin ({reloj.transcurrido():.1f} min)")


if __name__ == "__main__":
    main()
