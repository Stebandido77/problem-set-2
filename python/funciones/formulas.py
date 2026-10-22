"""Programación genética de variables (etapa 14) con gplearn: fórmulas sobre las
variables de más gain, evolucionadas con cruce y mutación (SymbolicTransformer),
con el F1 de la clase pobre con umbral óptimo (la métrica del pipeline) como
aptitud. Las fórmulas se ajustan solo con las filas de entrenamiento del
contexto y después se aplican, como variables nuevas, a su entrenamiento y a su
evaluación (formular_contexto()).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from funciones.metricas import buscar_umbral_f1

FUNCIONES = ("add", "sub", "mul", "div", "log", "max", "min")
TOPE = 1e12  # las fórmulas se recortan a [-TOPE, TOPE] (mul y div pueden desbordar)


def aptitud_f1_umbral(y, y_pred, w):
    """F1 de la clase pobre con el umbral óptimo, en la mejor de las dos
    orientaciones, solo con las filas de peso > 0.

    gplearn evalúa cada programa en una submuestra y pone peso 0 a las filas que
    quedaron fuera; la salida de un programa es un puntaje con cualquier escala,
    así que se pasa a rangos en (0, 1] antes de buscar el umbral (F1 invariante a
    transformaciones monótonas).
    Argumentos: y = 0/1; y_pred = salida del programa; w = pesos (0 = fuera).
    Devuelve: F1 en [0, 1] (0 si el programa es constante o no finito).
    """
    mascara = np.asarray(w) > 0
    verdad = np.asarray(y)[mascara]
    valor = np.asarray(y_pred, dtype=float)[mascara]
    if len(verdad) == 0 or not np.all(np.isfinite(valor)) or valor.min() == valor.max():
        return 0.0
    rango = rankdata(valor) / len(valor)
    return max(buscar_umbral_f1(verdad, rango)[1], buscar_umbral_f1(verdad, 1 + 1 / len(valor) - rango)[1])


def crear_transformador(columnas, n_formulas=15, poblacion=1000, generaciones=25, fraccion_filas=0.25,
                        semilla=2026, n_jobs=1):
    """SymbolicTransformer de gplearn con la aptitud F1 del pipeline.

    Argumentos: columnas = nombres de las variables (para leer las fórmulas);
        n_formulas = fórmulas que devuelve (las menos correlacionadas del salón de
        la fama); poblacion, generaciones = tamaño de la búsqueda; fraccion_filas =
        submuestra por programa para la aptitud; semilla, n_jobs = de gplearn.
    Devuelve: transformador sin ajustar.
    """
    from gplearn.fitness import make_fitness  # noqa: PLC0415 (carga solo si se usa)
    from gplearn.genetic import SymbolicTransformer  # noqa: PLC0415
    return SymbolicTransformer(
        population_size=poblacion, generations=generaciones,
        hall_of_fame=min(poblacion, max(50, 3 * n_formulas)), n_components=n_formulas,
        function_set=FUNCIONES, metric=make_fitness(function=aptitud_f1_umbral, greater_is_better=True, wrap=False),
        max_samples=fraccion_filas, parsimony_coefficient=0.0005, init_depth=(2, 5), tournament_size=20,
        p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1,
        feature_names=list(columnas), random_state=semilla, n_jobs=n_jobs, verbose=0)


def formulas_de(transformador):
    """Fórmulas elegidas por un transformador ajustado, con su aptitud (F1) y longitud."""
    return [{"formula": str(programa), "aptitud": float(programa.raw_fitness_), "longitud": int(programa.length_)}
            for programa in transformador._best_programs]


def transformar(transformador, x):
    """Aplica las fórmulas a x: DataFrame formula_1..formula_k (float32, finito)."""
    valores = transformador.transform(x.to_numpy(np.float64))
    valores = np.clip(np.nan_to_num(valores, nan=0.0, posinf=TOPE, neginf=-TOPE), -TOPE, TOPE)
    return pd.DataFrame(valores.astype(np.float32), index=x.index,
                        columns=[f"formula_{i + 1}" for i in range(valores.shape[1])])


def formular_contexto(x_ent, y_ent, x_eval, transformador):
    """Ajusta las fórmulas con el entrenamiento de un contexto y las aplica a su
    entrenamiento y a su evaluación (que nunca entra al ajuste).

    Argumentos: x_ent, x_eval = DataFrames indexados por id con las mismas columnas;
        y_ent = 0/1 del entrenamiento; transformador = crear_transformador().
    Devuelve: (tabla indexada por los ids de x_ent y x_eval, fórmulas de formulas_de()).
    """
    transformador.fit(x_ent.to_numpy(np.float64), np.asarray(y_ent))
    tabla = pd.concat([transformar(transformador, x_ent), transformar(transformador, x_eval)])
    return tabla, formulas_de(transformador)


def main():
    """CLI: fórmulas de un contexto en un proceso aparte y liviano.

    gplearn paraleliza con procesos (loky) que en Windows reimportan el módulo
    principal; si ese módulo importa torch (como la etapa 14) los workers mueren,
    así que la etapa lanza `python -m funciones.formulas` por contexto. Lee las
    matrices del contexto (parquet con las columnas pedidas y, en el
    entrenamiento, pobre) y escribe la tabla de fórmulas (parquet) y las
    fórmulas ganadoras (json).
    """
    import argparse  # noqa: PLC0415 (solo para la línea de comandos)
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    analizador = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("--entrenamiento", required=True, help="parquet del entrenamiento (con pobre)")
    analizador.add_argument("--evaluacion", required=True, help="parquet de la evaluación")
    analizador.add_argument("--salida", required=True, help="parquet de salida con las fórmulas aplicadas")
    analizador.add_argument("--formulas", required=True, help="json de salida con las fórmulas ganadoras")
    analizador.add_argument("--columnas", required=True, nargs="+", help="variables de entrada de las fórmulas")
    analizador.add_argument("--n-formulas", type=int, default=15)
    analizador.add_argument("--poblacion", type=int, default=1000)
    analizador.add_argument("--generaciones", type=int, default=25)
    analizador.add_argument("--fraccion-filas", type=float, default=0.25)
    analizador.add_argument("--semilla", type=int, default=2026)
    analizador.add_argument("--n-jobs", type=int, default=1)
    args = analizador.parse_args()

    x_ent = pd.read_parquet(args.entrenamiento, columns=list(args.columnas) + ["pobre"])
    y_ent = x_ent.pop("pobre").to_numpy().astype(int)
    x_eval = pd.read_parquet(args.evaluacion, columns=list(args.columnas))
    transformador = crear_transformador(args.columnas, n_formulas=args.n_formulas, poblacion=args.poblacion,
                                        generaciones=args.generaciones, fraccion_filas=args.fraccion_filas,
                                        semilla=args.semilla, n_jobs=args.n_jobs)
    tabla, formulas = formular_contexto(x_ent, y_ent, x_eval, transformador)
    temporal = Path(args.salida).with_suffix(".tmp.parquet")
    tabla.to_parquet(temporal)
    temporal.replace(args.salida)
    Path(args.formulas).write_text(json.dumps(formulas, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
