"""Etapa 5: ¿cuánto techo queda? Solo usa train y holdout (nunca el test).

  a) Hogares del holdout cerca de la línea: share con ingreso per cápita / Lp en
     [0,8; 1,2] y [0,9; 1,1], y F1 del modelo final dentro y fuera de cada banda.
  b) Oráculo parcial: en el holdout se reemplaza el auxiliar de ingreso (laboral,
     total o ambos) por el ingreso OBSERVADO, agregado igual que el auxiliar, y se
     predice con el modelo ya entrenado. Es solo diagnóstico: dice cuánto F1 queda
     si el auxiliar fuera perfecto en predicción.
  c) Curva de aprendizaje: F1 holdout entrenando con 25/50/75/100% del 80%
     (umbral con OOF sobre la misma submuestra; los auxiliares se dejan fijos).

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa5_techo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from configuracion import PARAMETROS_N6, RUTA_NIVEL7, RUTA_TABLAS, SEMILLA
from etapa2_optuna import SEMILLAS_BAGGING
from funciones.anidado import definir_contextos
from funciones.auxiliares import FUERA_UNIDAD_GASTO, agregar_ingreso_hogar
from funciones.bases import armar_matriz, cargar_insumos
from funciones.bitacora import Presupuesto, registrar_log
from funciones.datos import cargar_ingreso_hogar, cargar_ingreso_personas, cargar_personas, imputar_mediana
from funciones.metricas import buscar_umbral_f1, calcular_metricas
from funciones.modelado import ajustar_lightgbm, armar_contexto, en_grupos

CANDIDATO = "n7_n6param_a4_total_bag5"
BANDAS = ((0.8, 1.2), (0.9, 1.1))
FRACCIONES = (0.25, 0.5, 0.75, 1.0)


def f1_en(verdad, clase, mascara):
    """F1, n de hogares y n de pobres en un subconjunto."""
    m = calcular_metricas(verdad[mascara], clase[mascara])
    return {"hogares": int(mascara.sum()), "pobres": int(verdad[mascara].sum()), "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"]}


def bandas_linea(verdad_holdout, prob_holdout, umbral):
    """a) F1 dentro y fuera de las bandas alrededor de la línea de pobreza."""
    razon = np.exp(cargar_ingreso_hogar()).reindex(prob_holdout.index).to_numpy()
    verdad = verdad_holdout.to_numpy()
    clase = (prob_holdout.to_numpy() >= umbral).astype(int)
    filas = [{"banda": "todo el holdout", "share_hogares": 1.0, **f1_en(verdad, clase, np.ones_like(verdad, bool))}]
    for bajo, alto in BANDAS:
        dentro = (razon >= bajo) & (razon <= alto)
        filas.append({"banda": f"dentro [{bajo}; {alto}]", "share_hogares": dentro.mean(), **f1_en(verdad, clase, dentro)})
        filas.append({"banda": f"fuera [{bajo}; {alto}]", "share_hogares": (~dentro).mean(), **f1_en(verdad, clase, ~dentro)})
    return pd.DataFrame(filas)


def ingreso_observado_hogar(ids_hogar, hogar, columna, sufijo):
    """Ingreso observado agregado al hogar igual que el auxiliar (solo train/holdout)."""
    personas = cargar_personas()[["id", "Orden", "P6050"]]
    personas = personas[personas["id"].isin(ids_hogar)]
    ingreso = personas.merge(cargar_ingreso_personas(), on=["id", "Orden"], how="left")
    assert ingreso[columna].notna().all(), "faltan ingresos observados en el holdout"
    en_ug = ~ingreso["P6050"].isin(FUERA_UNIDAD_GASTO).to_numpy()
    return agregar_ingreso_hogar(ingreso["id"].to_numpy(), ingreso[columna].to_numpy(), en_ug, hogar, sufijo)


def oraculo(insumos, matriz, verdad, umbral):
    """b) Modelo final entrenado en el 80% y holdout con auxiliares observados."""
    tabla = insumos["particion"].set_index("id")
    entrenamiento, evaluacion = definir_contextos()["holdout"]
    ids_ent = tabla.index[en_grupos(tabla, entrenamiento)]
    ids_hold = tabla.index[en_grupos(tabla, evaluacion)]
    dinamicas = [insumos["auxiliares"]["laboral"], insumos["auxiliares"]["total"]]
    x_ent = armar_contexto(matriz, dinamicas, "holdout", ids_ent)
    x_hold = armar_contexto(matriz, dinamicas, "holdout", ids_hold)
    x_ent, x_hold = imputar_mediana(x_ent, x_hold)
    modelos = [ajustar_lightgbm(x_ent, verdad.loc[ids_ent].astype(int).to_numpy(), PARAMETROS_N6, s) for s in SEMILLAS_BAGGING]

    observado = {
        "laboral": ingreso_observado_hogar(ids_hold, insumos["hogar"], "ingreso_laboral", ""),
        "total": ingreso_observado_hogar(ids_hold, insumos["hogar"], "ingreso_total", "_total"),
    }
    escenarios = {"auxiliares predichos (modelo final)": [], "laboral observado": ["laboral"],
                  "total observado": ["total"], "laboral y total observados": ["laboral", "total"]}
    y_hold = verdad.loc[ids_hold].astype(int).to_numpy()
    filas = []
    for nombre, reemplazos in escenarios.items():
        x = x_hold.copy()
        for fuente in reemplazos:
            valores = observado[fuente].reindex(ids_hold)
            x[valores.columns] = valores.to_numpy(dtype=np.float32)
        probabilidad = np.mean([m.predict(x) for m in modelos], axis=0)
        metricas = calcular_metricas(y_hold, (probabilidad >= umbral).astype(int))
        filas.append({"escenario": nombre, "f1_holdout": metricas["f1"], "precision": metricas["precision"],
                      "recall": metricas["recall"], "fn": metricas["n_fn"], "fp": metricas["n_fp"]})
    return pd.DataFrame(filas)


def curva_aprendizaje(insumos, matriz, verdad):
    """c) F1 holdout con submuestras estratificadas del 80% (1 semilla)."""
    tabla = insumos["particion"].set_index("id")
    dinamicas = [insumos["auxiliares"]["laboral"], insumos["auxiliares"]["total"]]
    contextos = definir_contextos()
    ids_80 = tabla.index[en_grupos(tabla, (1, 2, 3, 4, 5))]
    generador = np.random.default_rng(SEMILLA)
    filas = []
    for fraccion in FRACCIONES:
        # Muestra estratificada por pobre, la misma en CV y en holdout
        elegidos = set()
        for clase in (0, 1):
            ids_clase = ids_80[verdad.loc[ids_80].astype(int).to_numpy() == clase]
            elegidos |= set(generador.choice(ids_clase, int(round(fraccion * len(ids_clase))), replace=False))
        oof_verdad, oof_prob = [], []
        for k in range(1, 6):
            entrenamiento, evaluacion = contextos[f"cv_{k}"]
            ids_ent = [i for i in tabla.index[en_grupos(tabla, entrenamiento)] if i in elegidos]
            ids_val = [i for i in tabla.index[en_grupos(tabla, evaluacion)] if i in elegidos]
            x_ent, x_val = imputar_mediana(armar_contexto(matriz, dinamicas, f"cv_{k}", ids_ent),
                                           armar_contexto(matriz, dinamicas, f"cv_{k}", ids_val))
            modelo = ajustar_lightgbm(x_ent, verdad.loc[ids_ent].astype(int).to_numpy(), PARAMETROS_N6)
            oof_verdad.append(verdad.loc[ids_val].astype(int).to_numpy())
            oof_prob.append(modelo.predict(x_val))
        umbral, f1_oof = buscar_umbral_f1(np.concatenate(oof_verdad), np.concatenate(oof_prob))
        entrenamiento, evaluacion = contextos["holdout"]
        ids_ent = [i for i in tabla.index[en_grupos(tabla, entrenamiento)] if i in elegidos]
        ids_hold = tabla.index[en_grupos(tabla, evaluacion)]
        x_ent, x_hold = imputar_mediana(armar_contexto(matriz, dinamicas, "holdout", ids_ent),
                                        armar_contexto(matriz, dinamicas, "holdout", ids_hold))
        modelo = ajustar_lightgbm(x_ent, verdad.loc[ids_ent].astype(int).to_numpy(), PARAMETROS_N6)
        metricas = calcular_metricas(verdad.loc[ids_hold].astype(int).to_numpy(),
                                     (modelo.predict(x_hold) >= umbral).astype(int))
        filas.append({"fraccion_del_80": fraccion, "hogares_entrenamiento": len(ids_ent),
                      "umbral": umbral, "f1_oof": f1_oof, "f1_holdout": metricas["f1"]})
        registrar_log("etapa5", f"curva {fraccion:.2f}: holdout {metricas['f1']:.4f}")
    return pd.DataFrame(filas)


def main():
    reloj = Presupuesto(30)
    registrar_log("etapa5", "inicio")
    insumos = cargar_insumos(auxiliar_total=True)
    verdad = insumos["particion"].set_index("id")["pobre"]
    matriz = armar_matriz(insumos)

    predicciones = pd.read_parquet(RUTA_NIVEL7 / f"predicciones_{CANDIDATO}.parquet").set_index("id")
    oof = predicciones[predicciones["tipo"] == "oof"]
    holdout = predicciones[predicciones["tipo"] == "holdout"]["probabilidad"]
    umbral, _ = buscar_umbral_f1(verdad.loc[oof.index], oof["probabilidad"])

    tabla_bandas = bandas_linea(verdad.loc[holdout.index].astype(int), holdout, umbral)
    tabla_bandas.to_csv(RUTA_TABLAS / "techo_bandas.csv", index=False)
    print(tabla_bandas.round(4).to_string(index=False))
    registrar_log("etapa5", "a) bandas listas")

    tabla_oraculo = oraculo(insumos, matriz, verdad, umbral)
    tabla_oraculo.to_csv(RUTA_TABLAS / "techo_oraculo.csv", index=False)
    print(tabla_oraculo.round(4).to_string(index=False))
    registrar_log("etapa5", "b) oráculo listo")

    if reloj.agotado():
        registrar_log("etapa5", "tiempo agotado antes de la curva de aprendizaje")
        return
    tabla_curva = curva_aprendizaje(insumos, matriz, verdad)
    tabla_curva.to_csv(RUTA_TABLAS / "techo_curva_aprendizaje.csv", index=False)
    print(tabla_curva.round(4).to_string(index=False))
    registrar_log("etapa5", f"fin ({reloj.transcurrido():.1f} min)")


if __name__ == "__main__":
    main()
