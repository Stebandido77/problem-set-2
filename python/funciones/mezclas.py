"""Promedios de candidatos ya evaluados: bagging de semillas y mezclas
ponderadas con pesos elegidos solo en OOF.

Holdout y test reciben la misma combinación que la OOF; el umbral lo vuelve a
elegir resultado_desde_predicciones() con buscar_umbral_f1() sobre la OOF.
"""

import numpy as np
import pandas as pd

from funciones.metricas import buscar_umbral_f1
from funciones.modelado import guardar_predicciones, resultado_desde_predicciones


def combinar_predicciones(nombre, resultados, pesos):
    """Promedio ponderado de las predicciones (OOF, holdout y test) de varios candidatos.

    Argumentos: nombre = candidato nuevo; resultados = lista de resultados
        (resultado_desde_predicciones()); pesos = arreglo que suma 1.
    Devuelve: diccionario con oof, prediccion_holdout y (si todos lo tienen)
        prediccion_test, alineados por id con el primer resultado.
    """
    primero = resultados[0]

    def combinar(serie_de):
        indice = serie_de(primero).index
        return sum(w * serie_de(r).reindex(indice) for w, r in zip(pesos, resultados)).rename("probabilidad")

    salida = {"nombre": nombre,
              "oof": pd.DataFrame({"probabilidad": combinar(lambda r: r["oof"]["probabilidad"]),
                                   "fold": primero["oof"]["fold"]}),
              "prediccion_holdout": combinar(lambda r: r["prediccion_holdout"])}
    if all("prediccion_test" in r for r in resultados):
        salida["prediccion_test"] = combinar(lambda r: r["prediccion_test"])
    return salida


def promediar_semillas(nombre, nombres, verdad):
    """Bagging: promedio simple de las probabilidades de varias semillas.

    Argumentos: nombre = candidato nuevo; nombres = candidatos de cada semilla;
        verdad = pobre por id.
    Devuelve: resultado del promedio (se guarda como un candidato más).
    """
    resultados = [resultado_desde_predicciones(n, verdad) for n in nombres]
    guardar_predicciones(combinar_predicciones(nombre, resultados, np.full(len(nombres), 1 / len(nombres))))
    return resultado_desde_predicciones(nombre, verdad)


def mezclar_por_oof(nombre, nombres, grilla, verdad):
    """Mezcla ponderada con los pesos de la grilla que maximizan el F1 OOF.

    Solo se mira la OOF (F1 con el umbral óptimo de buscar_umbral_f1()); ante
    empates gana el primer vector de la grilla.
    Argumentos: nombre = candidato nuevo; nombres = candidatos a mezclar;
        grilla = lista de vectores de pesos (cada uno suma 1); verdad = pobre por id.
    Devuelve: (resultado, pesos elegidos, tabla con el F1 OOF de cada vector).
    """
    resultados = [resultado_desde_predicciones(n, verdad) for n in nombres]
    indice = resultados[0]["oof"].index
    y = resultados[0]["oof"]["verdad"].to_numpy()
    oof = np.column_stack([r["oof"]["probabilidad"].reindex(indice).to_numpy() for r in resultados])
    f1 = [buscar_umbral_f1(y, oof @ np.asarray(w))[1] for w in grilla]
    pesos = np.asarray(grilla[int(np.argmax(f1))])
    guardar_predicciones(combinar_predicciones(nombre, resultados, pesos))
    tabla = pd.DataFrame(np.asarray(grilla), columns=[f"peso_{n}" for n in nombres]).assign(f1_oof=f1)
    return resultado_desde_predicciones(nombre, verdad), pesos, tabla
