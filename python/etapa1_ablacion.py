"""Etapa 1 del nivel 7: variables v2 y ablación con los hiperparámetros de
n6_lightgbm FIJOS, misma CV y holdout que R.

Filas acumuladas: nivel 3 -> + laborales/composición -> + perceptor principal ->
+ hogares pequeños -> + auxiliar de ingreso (personas) -> + auxiliar directo de
hogar; y aparte, la mejor fila + factores de expansión (Fex).

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa1_ablacion.py [minutos_maximos]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import F1_HOLDOUT_N6, PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS
from funciones.anidado import asignar_grupos, cargar_o_calcular_anidado
from funciones.auxiliares import crear_auxiliar_hogar, crear_auxiliar_personas
from funciones.bitacora import Presupuesto, registrar_log
from funciones.datos import (cargar_hogares, cargar_ingreso_hogar, cargar_ingreso_personas,
                             cargar_particion, cargar_personas, preparar_matriz)
from funciones.modelado import evaluar_candidato, f1_subgrupos
from funciones.variables import construir_grupos

ETAPA = "etapa1"


def main(minutos_maximos=90):
    presupuesto = Presupuesto(minutos_maximos)
    registrar_log(ETAPA, "inicio")
    particion = cargar_particion()
    grupos = asignar_grupos(particion)
    hogar = cargar_hogares()
    personas = cargar_personas()

    bloques = construir_grupos(personas, hogar)
    registrar_log(ETAPA, "variables v2 construidas")

    # Auxiliar de ingreso de personas (la apuesta principal): primero, para que
    # entre aunque el tiempo no alcance para todas las filas
    aux_personas = cargar_o_calcular_anidado(
        "ingreso_personas", crear_auxiliar_personas(personas, cargar_ingreso_personas(), hogar), grupos)
    registrar_log(ETAPA, "auxiliar de personas listo")
    del personas

    filas_ablacion = [
        ("n7_a0_nivel3", ["nivel_3"], []),
        ("n7_a1_laborales", ["nivel_3", "laborales"], []),
        ("n7_a2_perceptor", ["nivel_3", "laborales", "perceptor"], []),
        ("n7_a3_pequenos", ["nivel_3", "laborales", "perceptor", "pequenos"], []),
        ("n7_a4_aux_personas", ["nivel_3", "laborales", "perceptor", "pequenos"], ["personas"]),
        ("n7_a5_aux_hogar", ["nivel_3", "laborales", "perceptor", "pequenos"], ["personas", "hogar"]),
    ]

    resultados = []
    ruta_tabla = RUTA_TABLAS / "ablacion_etapa1.csv"
    for nombre, nombres_bloques, auxiliares in filas_ablacion:
        if presupuesto.agotado():
            registrar_log(ETAPA, f"tiempo agotado antes de {nombre}")
            break
        matriz = preparar_matriz(pd.concat([bloques[b] for b in nombres_bloques], axis=1))
        dinamicas = [aux_personas] if "personas" in auxiliares else []
        if "hogar" in auxiliares:
            aux_hogar = cargar_o_calcular_anidado(
                "ingreso_hogar", crear_auxiliar_hogar(matriz, cargar_ingreso_hogar()), grupos)
            dinamicas.append(aux_hogar)
        resultado = evaluar_candidato(nombre, matriz, particion, PARAMETROS_N6, dinamicas)
        resultados.append(resumir(resultado, nombres_bloques, auxiliares, matriz, dinamicas, hogar, particion))
        pd.DataFrame(resultados).to_csv(ruta_tabla, index=False)

    # Aparte: la mejor fila (por F1 CV) + factores de expansión
    if resultados and not presupuesto.agotado():
        mejor = max(resultados, key=lambda r: r["f1_cv"])
        fila = next(f for f in filas_ablacion if f[0] == mejor["candidato"])
        nombres_bloques = fila[1] + ["fex"]
        matriz = preparar_matriz(pd.concat([bloques[b] for b in nombres_bloques], axis=1))
        dinamicas = [aux_personas] if "personas" in fila[2] else []
        if "hogar" in fila[2]:
            dinamicas.append(cargar_o_calcular_anidado("ingreso_hogar", None, grupos))
        resultado = evaluar_candidato(f"{mejor['candidato']}_fex", matriz, particion, PARAMETROS_N6, dinamicas)
        resultados.append(resumir(resultado, nombres_bloques, fila[2], matriz, dinamicas, hogar, particion))
        pd.DataFrame(resultados).to_csv(ruta_tabla, index=False)

    tabla = pd.DataFrame(resultados)
    tabla["delta_holdout_vs_n6"] = tabla["f1_holdout"] - F1_HOLDOUT_N6
    tabla.to_csv(ruta_tabla, index=False)
    registrar_log(ETAPA, f"fin ({presupuesto.transcurrido():.1f} min)")
    print(tabla.drop(columns=["bloques"]).round(4).to_string(index=False))


def resumir(resultado, nombres_bloques, auxiliares, matriz, dinamicas, hogar, particion):
    """Fila de la tabla de ablación de un candidato."""
    n_variables = matriz.shape[1] + sum(next(iter(d.values())).shape[1] for d in dinamicas)
    fila = {
        "candidato": resultado["nombre"],
        "bloques": " + ".join(nombres_bloques + [f"aux_{a}" for a in auxiliares]),
        "n_columnas": n_variables,
        "umbral": resultado["umbral"],
        "f1_cv": resultado["f1_cv"], "f1_cv_sd": resultado["f1_cv_sd"],
        "precision_cv": resultado["precision_cv"], "recall_cv": resultado["recall_cv"],
        "f1_holdout": resultado["f1_holdout"],
        "precision_holdout": resultado["precision_holdout"],
        "recall_holdout": resultado["recall_holdout"],
        **f1_subgrupos(resultado, hogar, particion),
        "tiempo_min": resultado["tiempo_min"],
    }
    importancia = resultado["importancia"]
    (importancia / importancia.sum()).sort_values(ascending=False).rename("gain").to_csv(
        RUTA_TABLAS / f"importancia_{resultado['nombre']}.csv")
    return fila


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 90)
