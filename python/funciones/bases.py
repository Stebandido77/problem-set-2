"""Insumos comunes del nivel 7: partición, hogar, bloques de variables v2 y
auxiliares de ingreso (desde la caché de stores/procesados/nivel7/)."""

import pandas as pd

from funciones.anidado import asignar_grupos, cargar_o_calcular_anidado
from funciones.auxiliares import crear_auxiliar_personas
from funciones.datos import (cargar_hogares, cargar_ingreso_personas, cargar_particion,
                             cargar_personas, preparar_matriz)
from funciones.variables import construir_grupos

# Base ganadora de la etapa 1 (a4): nivel 3 + v2 sin Fex ni auxiliar de hogar
BLOQUES_A4 = ["nivel_3", "laborales", "perceptor", "pequenos"]


def cargar_insumos(auxiliar_total=False):
    """Carga todo lo que necesita un candidato del nivel 7.

    Argumentos: auxiliar_total = si True también arma el auxiliar de ingreso
        total (sufijo _total), con la misma OOF anidada.
    Devuelve: diccionario con particion, grupos, hogar, bloques y auxiliares
        ({nombre: {contexto: DataFrame}}).
    """
    particion = cargar_particion()
    grupos = asignar_grupos(particion)
    hogar = cargar_hogares()
    personas = cargar_personas()
    bloques = construir_grupos(personas, hogar)
    ingreso = cargar_ingreso_personas()
    auxiliares = {"laboral": cargar_o_calcular_anidado(
        "ingreso_personas", crear_auxiliar_personas(personas, ingreso, hogar), grupos)}
    if auxiliar_total:
        auxiliares["total"] = cargar_o_calcular_anidado(
            "ingreso_total_personas",
            crear_auxiliar_personas(personas, ingreso, hogar, "ingreso_total", "_total"), grupos)
    return {"particion": particion, "grupos": grupos, "hogar": hogar,
            "bloques": bloques, "auxiliares": auxiliares}


def armar_matriz(insumos, bloques=BLOQUES_A4, nativas=()):
    """Matriz fija (sin auxiliares) de un conjunto de bloques.

    Argumentos: insumos = cargar_insumos(); bloques = nombres de bloques;
        nativas = categóricas que quedan nativas para LightGBM.
    Devuelve: DataFrame indexado por id.
    """
    return preparar_matriz(pd.concat([insumos["bloques"][b] for b in bloques], axis=1), nativas)
