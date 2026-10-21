"""Insumos comunes del nivel 7: partición, hogar, bloques de variables v2 y
auxiliares de ingreso y de estrato (desde la caché de stores/procesados/nivel7/)."""

import pandas as pd

from funciones.anidado import asignar_grupos, cargar_o_calcular_anidado, crear_perezoso
from funciones.auxiliares import (crear_auxiliar_estrato, crear_auxiliar_fuentes, crear_auxiliar_personas,
                                  derivar_per_capita)
from funciones.datos import (cargar_estrato_hogar, cargar_hogares, cargar_ingreso_fuentes,
                             cargar_ingreso_personas, cargar_particion, cargar_personas, preparar_matriz)
from funciones.variables import construir_grupos

# Base ganadora de la etapa 1 (a4): nivel 3 + v2 sin Fex ni auxiliar de hogar
BLOQUES_A4 = ["nivel_3", "laborales", "perceptor", "pequenos"]
# Sufijo de las variables de agregar_ingreso_hogar() en cada auxiliar de ingreso
SUFIJO_AUXILIAR = {"laboral": "", "total": "_total", "fuentes": "_v3"}


def reconciliar_auxiliares(auxiliares, hogar, arriendo_imputado):
    """Recalcula las variables per cápita de los auxiliares de ingreso desde el
    total guardado en caché, con o sin el arriendo imputado del hogar.

    Las cachés de las etapas 1 a 10 se construyeron sin arriendo imputado; como
    guardan ingreso_pred_total, la construcción queda determinada por el
    argumento y no por cómo se calculó la caché.
    Argumentos: auxiliares = {nombre: {contexto: DataFrame}}; hogar =
        cargar_hogares(); arriendo_imputado = ver derivar_per_capita().
    Devuelve: el mismo diccionario con las tablas recalculadas.
    """
    return {nombre: {contexto: derivar_per_capita(tabla, hogar, SUFIJO_AUXILIAR[nombre], arriendo_imputado)
                     if nombre in SUFIJO_AUXILIAR else tabla
                     for contexto, tabla in por_contexto.items()}
            for nombre, por_contexto in auxiliares.items()}


def cargar_insumos(auxiliar_total=False, auxiliar_fuentes=False, auxiliar_estrato=False,
                   arriendo_imputado=False):
    """Carga todo lo que necesita un candidato del nivel 7.

    Argumentos: auxiliar_total = si True también arma el auxiliar de ingreso
        total (sufijo _total), con la misma OOF anidada; auxiliar_fuentes = si
        True también carga el auxiliar v3 de dos partes por fuente (etapa 6);
        auxiliar_estrato = si True también arma el auxiliar de estrato (etapa 11);
        arriendo_imputado = si True, el per cápita de los auxiliares de ingreso
        incluye el arriendo imputado del hogar (como Ingpcug); False es la
        construcción con la que se corrieron las etapas 1 a 10.
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
    if auxiliar_fuentes:
        contexto_hogar = pd.concat([bloques["nivel_3"], bloques["laborales"]], axis=1)
        auxiliares["fuentes"] = cargar_o_calcular_anidado(
            "ingreso_fuentes_v3",
            crear_perezoso(lambda: crear_auxiliar_fuentes(personas, cargar_ingreso_fuentes(), hogar, contexto_hogar)),
            grupos)
    if auxiliar_estrato:
        # Mismos 280 predictores de hogar de a4 (presentes en test), sin auxiliares previos
        auxiliares["estrato"] = cargar_o_calcular_anidado(
            "estrato_hogar",
            crear_perezoso(lambda: crear_auxiliar_estrato(
                preparar_matriz(pd.concat([bloques[b] for b in BLOQUES_A4], axis=1)), cargar_estrato_hogar())),
            grupos)
    auxiliares = reconciliar_auxiliares(auxiliares, hogar, arriendo_imputado)
    return {"particion": particion, "grupos": grupos, "hogar": hogar,
            "bloques": bloques, "auxiliares": auxiliares}


def armar_matriz(insumos, bloques=BLOQUES_A4, nativas=()):
    """Matriz fija (sin auxiliares) de un conjunto de bloques.

    Argumentos: insumos = cargar_insumos(); bloques = nombres de bloques;
        nativas = categóricas que quedan nativas para LightGBM.
    Devuelve: DataFrame indexado por id.
    """
    return preparar_matriz(pd.concat([insumos["bloques"][b] for b in bloques], axis=1), nativas)
