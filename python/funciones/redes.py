"""Redes neuronales tabulares del nivel 7 (etapa 10) con las implementaciones
oficiales de sus autores:

  - RealMLP (Holzmüller, Grinsztajn y Steinwart, 2024) del paquete pytabkit,
    con su preprocesamiento propio (centrado en la mediana, robust scaling y
    smooth clipping), ajustado solo con las filas que recibe fit().
  - TabM (Gorishniy, Kotelnikov y Babenko, 2025) del paquete tabm; el
    entrenamiento sigue el ejemplo oficial (AdamW, pérdida promedio de los k
    submodelos, predicción = promedio de sus probabilidades). Las numéricas
    continuas pasan por una transformación de cuantiles a normal ajustada solo
    con el entrenamiento.

Las dos usan early stopping sobre un 10% interno estratificado del
entrenamiento del contexto (particion_parada()); el fold de evaluación nunca
entra a fit(). Las categóricas ya llegan en dummies (one-hot) desde
preparar_matriz().
"""

import copy
import gc
import time

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.preprocessing import QuantileTransformer

from funciones.evolutivo import muestra_estratificada

# Núcleos físicos: torch con hilos lógicos (hyperthreading) corre más lento
N_NUCLEOS_FISICOS = psutil.cpu_count(logical=False) or 1
FRACCION_PARADA = 0.10
FILAS_PREDICCION = 8192


def particion_parada(y, semilla, fraccion=FRACCION_PARADA):
    """Divide el entrenamiento de un contexto en ajuste y parada temprana.

    La parada es una submuestra estratificada de `fraccion` de las filas de y
    (que son solo las de entrenamiento del contexto).
    Argumentos: y = arreglo 0/1 del entrenamiento; semilla = entero;
        fraccion = parte que va a la parada temprana.
    Devuelve: (posiciones de ajuste, posiciones de parada), disjuntas y ordenadas.
    """
    rng = np.random.default_rng(semilla)
    parada = muestra_estratificada(y, int(round(fraccion * len(y))), rng)
    ajuste = np.setdiff1d(np.arange(len(y)), parada)
    return ajuste, parada


def columnas_variables(x):
    """Columnas con más de un valor distinto (las constantes no aportan a una red).

    Argumentos: x = DataFrame de entrenamiento del contexto.
    Devuelve: lista de nombres de columnas.
    """
    return list(x.columns[x.nunique() > 1])


def escalar_cuantiles(x_ajuste, *otras, semilla=0):
    """Transformación de cuantiles a normal de las columnas continuas.

    Se ajusta solo con x_ajuste; las demás matrices solo se transforman. Las
    columnas binarias (dummies) quedan en 0/1.
    Argumentos: x_ajuste = DataFrame de ajuste; otras = DataFrames con las mismas
        columnas; semilla = para la submuestra del transformador.
    Devuelve: lista de arreglos float32 (x_ajuste primero, luego otras).
    """
    continuas = np.flatnonzero(x_ajuste.nunique().to_numpy() > 2)
    transformador = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                        subsample=200_000, random_state=semilla)
    transformador.fit(x_ajuste.iloc[:, continuas].to_numpy(np.float32))
    salida = []
    for x in (x_ajuste, *otras):
        matriz = x.to_numpy(np.float32, copy=True)
        matriz[:, continuas] = transformador.transform(matriz[:, continuas])
        salida.append(matriz)
    return salida


# ---- RealMLP ------------------------------------------------------------------

def crear_realmlp(parametros, semilla):
    """RealMLP-TD de pytabkit en CPU con selección del mejor epoch por log loss.

    Argumentos: parametros = hiperparámetros de RealMLP_TD_Classifier;
        semilla = random_state (inicialización y orden de los lotes).
    Devuelve: clasificador sin ajustar.
    """
    from pytabkit import RealMLP_TD_Classifier  # noqa: PLC0415 (carga pesada, solo si se usa)
    return RealMLP_TD_Classifier(device="cpu", random_state=semilla, n_threads=N_NUCLEOS_FISICOS, verbosity=0,
                                 val_metric_name="cross_entropy", use_ls=False, use_early_stopping=True,
                                 **parametros)


def crear_ajuste_realmlp(parametros, semilla):
    """Función ajustar() de RealMLP para evaluar_modelo().

    Argumentos: parametros = hiperparámetros de RealMLP; semilla = entero.
    Devuelve: función(x_ent, y_ent, x_eval) -> (probabilidad, None).
    """
    def ajustar(x_ent, y_ent, x_eval):
        # Se pasa x_ent completo y la parada como val_idxs: pytabkit ya copia los
        # datos varias veces en fit() y separar aquí ajuste y parada duplicaba la
        # matriz (salida por memoria en el contexto holdout). Las columnas
        # constantes quedan en 0 tras su centrado y escalado robusto.
        _, parada = particion_parada(y_ent, semilla)
        modelo = crear_realmlp(parametros, semilla)
        modelo.fit(x_ent, y_ent, val_idxs=parada)
        gc.collect()
        probabilidad = modelo.predict_proba(x_eval)[:, 1]
        return probabilidad, None

    return ajustar


# ---- TabM ---------------------------------------------------------------------

def predecir_tabm(modelo, x):
    """P(pobre) de TabM: promedio de las probabilidades de los k submodelos."""
    modelo.eval()
    salida = []
    with torch.no_grad():
        for inicio in range(0, len(x), FILAS_PREDICCION):
            logit = modelo(torch.from_numpy(x[inicio:inicio + FILAS_PREDICCION])).squeeze(-1)
            salida.append(torch.sigmoid(logit).mean(dim=1).numpy())
    return np.concatenate(salida).astype(np.float64)


def log_loss(y, probabilidad):
    """Entropía cruzada binaria promedio (con recorte numérico)."""
    p = np.clip(probabilidad, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def entrenar_tabm(x_ajuste, y_ajuste, x_parada, y_parada, parametros, semilla):
    """Entrena TabM con early stopping por log loss en la parada.

    Argumentos: x_ajuste, x_parada = arreglos float32 ya escalados; y_ajuste,
        y_parada = 0/1; parametros = k, d_block, n_blocks, dropout, arch_type,
        lr, weight_decay, batch_size, max_epocas, paciencia; semilla = entero.
    Devuelve: (modelo con los pesos del mejor epoch, historia de log loss).
    """
    import tabm  # noqa: PLC0415 (carga pesada, solo si se usa)
    torch.manual_seed(semilla)
    torch.set_num_threads(N_NUCLEOS_FISICOS)
    modelo = tabm.TabM.make(n_num_features=x_ajuste.shape[1], d_out=1, k=parametros["k"],
                            d_block=parametros["d_block"], n_blocks=parametros["n_blocks"],
                            dropout=parametros["dropout"], arch_type=parametros["arch_type"])
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=parametros["lr"],
                                    weight_decay=parametros["weight_decay"])
    x = torch.from_numpy(x_ajuste)
    y = torch.from_numpy(np.asarray(y_ajuste, dtype=np.float32))
    generador = torch.Generator().manual_seed(semilla)
    historia, mejor, mejor_estado, sin_mejora = [], np.inf, None, 0
    for _ in range(parametros["max_epocas"]):
        modelo.train()
        orden = torch.randperm(len(y), generator=generador)
        for inicio in range(0, len(orden), parametros["batch_size"]):
            lote = orden[inicio:inicio + parametros["batch_size"]]
            logit = modelo(x[lote]).squeeze(-1)  # (lote, k): cada submodelo con su pérdida
            perdida = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, y[lote][:, None].expand_as(logit))
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
        perdida_parada = log_loss(y_parada, predecir_tabm(modelo, x_parada))
        historia.append(perdida_parada)
        if perdida_parada < mejor:
            mejor, mejor_estado, sin_mejora = perdida_parada, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= parametros["paciencia"]:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, historia


def crear_ajuste_tabm(parametros, semilla, historias=None):
    """Función ajustar() de TabM para evaluar_modelo().

    Argumentos: parametros = ver entrenar_tabm(); semilla = entero;
        historias = lista opcional donde se anota (épocas, mejor log loss, segundos).
    Devuelve: función(x_ent, y_ent, x_eval) -> (probabilidad, None).
    """
    def ajustar(x_ent, y_ent, x_eval):
        inicio = time.time()
        columnas = columnas_variables(x_ent)
        ajuste, parada = particion_parada(y_ent, semilla)
        x_ent = x_ent[columnas]
        x_ajuste, x_parada, x_evaluacion = escalar_cuantiles(x_ent.iloc[ajuste], x_ent.iloc[parada],
                                                             x_eval[columnas], semilla=semilla)
        modelo, historia = entrenar_tabm(x_ajuste, y_ent[ajuste], x_parada, y_ent[parada], parametros, semilla)
        del x_ajuste, x_parada
        probabilidad = predecir_tabm(modelo, x_evaluacion)
        if historias is not None:
            historias.append({"epocas": len(historia), "mejor_epoca": int(np.argmin(historia)) + 1,
                              "log_loss_parada": float(np.min(historia)), "segundos": time.time() - inicio})
        return probabilidad, None

    return ajustar


def crear_ajuste_red(red, parametros, semilla):
    """ajustar() de la red indicada ('realmlp' o 'tabm')."""
    if red == "realmlp":
        return crear_ajuste_realmlp(parametros, semilla)
    if red == "tabm":
        return crear_ajuste_tabm(parametros, semilla)
    raise ValueError(f"red desconocida: {red}")


def describir_columnas(x):
    """Conteo de columnas por tipo (constantes, binarias, continuas) para el reporte."""
    distintos = x.nunique()
    return {"constantes": int((distintos <= 1).sum()), "binarias": int((distintos == 2).sum()),
            "continuas": int((distintos > 2).sum())}


def correlaciones_con(base, otras):
    """Pearson y Spearman de cada serie de `otras` con `base` (alineadas por id).

    Argumentos: base = Serie; otras = {nombre: Serie}.
    Devuelve: DataFrame con candidato, pearson_con_lightgbm, spearman_con_lightgbm.
    """
    filas = []
    for nombre, serie in otras.items():
        serie = serie.reindex(base.index)
        filas.append({"candidato": nombre, "pearson_con_lightgbm": base.corr(serie),
                      "spearman_con_lightgbm": base.corr(serie, method="spearman")})
    return pd.DataFrame(filas)
