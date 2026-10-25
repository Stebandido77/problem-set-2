"""Red de conjuntos (DeepSets con atención) para la etapa 12: cada hogar es el
conjunto de sus personas más las variables del hogar.

  - Codificador por persona (MLP compartido) sobre las variables de persona
    presentes en test (limpias de R y códigos originales, con indicadoras de NA).
  - Agregación invariante a permutaciones: suma, media, máximo y una atención
    (un peso por persona, softmax dentro del hogar).
  - Las variables de hogar (nivel 3, en dummies) pasan por su propio codificador
    y se concatenan con la agregación; una cabeza MLP da el logit de pobre.
  - Parada temprana por log loss en un 10% interno estratificado del
    entrenamiento del contexto; la estandarización se estima con el ajuste.
Solo CPU: los lotes se arman por hogares de tamaño parecido para no rellenar de
más (el hogar más grande tiene 28 personas y el percentil 99 son 9).
"""

import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from configuracion import SEMILLA
from funciones.redes import N_NUCLEOS_FISICOS, log_loss, particion_parada

EXCLUIDAS_PERSONA = ["id", "Orden", "Clase", "Dominio", "Depto", "Fex_c", "Fex_dpto", "Oficio"]
PARAMETROS_CONJUNTOS = {"d_oculto": 128, "d_salida": 256, "p_drop": 0.15, "lr": 1e-3, "weight_decay": 1e-5,
                        "lote": 512, "max_epocas": 40, "paciencia": 5}


def preparar_personas_conjuntos(personas):
    """Matriz de variables por persona: numéricas con NA en 0 más indicadoras de
    NA, categóricas en dummies.

    Argumentos: personas = cargar_personas().
    Devuelve: (matriz float32 personas x variables, Serie id de hogar por fila).
    """
    x = personas.drop(columns=[c for c in EXCLUIDAS_PERSONA if c in personas.columns and c != "id"])
    ids = x.pop("id")
    categoricas = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
    numericas = [c for c in x.columns if c not in categoricas]
    partes = [x[numericas].astype(np.float32)]
    con_na = [c for c in numericas if x[c].isna().any()]
    partes.append(x[con_na].isna().astype(np.float32).add_suffix("_na"))
    partes.append(pd.get_dummies(x[categoricas].astype("string").fillna("NA"), dtype=np.float32))
    matriz = pd.concat(partes, axis=1)
    matriz[numericas] = matriz[numericas].fillna(0.0)
    return matriz.to_numpy(np.float32), ids.reset_index(drop=True)


class Conjuntos:
    """Estructura de conjuntos: personas apiladas y, por hogar, las posiciones de
    sus personas; arma lotes rellenos con máscara.

    Argumentos: x_personas = matriz personas x variables; ids_persona = id de
        hogar de cada fila; x_hogar = DataFrame de variables de hogar indexado por id.
    """

    def __init__(self, x_personas, ids_persona, x_hogar):
        orden = np.argsort(ids_persona.to_numpy(), kind="stable")
        self.x_personas = np.vstack([x_personas[orden], np.zeros((1, x_personas.shape[1]), np.float32)])
        self.fila_relleno = len(self.x_personas) - 1
        ids_ordenados = ids_persona.to_numpy()[orden]
        hogares, inicio, conteo = np.unique(ids_ordenados, return_index=True, return_counts=True)
        self.inicio = pd.Series(inicio, index=hogares)
        self.conteo = pd.Series(conteo, index=hogares)
        self.x_hogar = x_hogar

    def indices(self, ids):
        """Matriz hogares x M de posiciones de personas (relleno al final) y su máscara."""
        inicio, conteo = self.inicio.loc[ids].to_numpy(), self.conteo.loc[ids].to_numpy()
        m = int(conteo.max())
        posiciones = inicio[:, None] + np.arange(m)[None, :]
        mascara = np.arange(m)[None, :] < conteo[:, None]
        posiciones = np.where(mascara, posiciones, self.fila_relleno)
        return posiciones, mascara.astype(np.float32)

    def lote(self, ids, media_p, desvio_p, media_h, desvio_h):
        """Tensores (personas [B, M, d], máscara [B, M], hogar [B, d_h]) estandarizados."""
        posiciones, mascara = self.indices(ids)
        personas = (self.x_personas[posiciones] - media_p) / desvio_p
        personas = personas * mascara[..., None]
        hogar = (self.x_hogar.loc[ids].to_numpy(np.float32) - media_h) / desvio_h
        return (torch.from_numpy(personas.astype(np.float32)), torch.from_numpy(mascara),
                torch.from_numpy(hogar.astype(np.float32)))

    def estadisticas(self, ids):
        """Media y desviación de las variables de persona (sobre las personas de
        esos hogares) y de hogar, para estandarizar con el ajuste."""
        posiciones, mascara = self.indices(ids)
        filas = posiciones[mascara.astype(bool)]
        personas = self.x_personas[filas]
        hogar = self.x_hogar.loc[ids].to_numpy(np.float32)
        return (personas.mean(0), np.where(personas.std(0) > 0, personas.std(0), 1.0),
                hogar.mean(0), np.where(hogar.std(0) > 0, hogar.std(0), 1.0))


class RedConjuntos(nn.Module):
    """Codificador por persona, agregación (suma, media, máximo, atención) y cabeza."""

    def __init__(self, d_persona, d_hogar, d_oculto, d_salida, p_drop):
        super().__init__()
        self.codificador = nn.Sequential(nn.Linear(d_persona, d_oculto), nn.SiLU(), nn.Linear(d_oculto, d_oculto), nn.SiLU())
        self.atencion = nn.Linear(d_oculto, 1)
        self.hogar = nn.Sequential(nn.Linear(d_hogar, d_oculto), nn.SiLU())
        self.cabeza = nn.Sequential(nn.Linear(5 * d_oculto, d_salida), nn.SiLU(), nn.Dropout(p_drop),
                                    nn.Linear(d_salida, d_oculto), nn.SiLU(), nn.Linear(d_oculto, 1))

    def forward(self, personas, mascara, hogar):
        h = self.codificador(personas) * mascara[..., None]
        n = mascara.sum(1, keepdim=True).clamp(min=1.0)
        suma = h.sum(1)
        media = suma / n
        maximo = (h + (mascara[..., None] - 1.0) * 1e4).max(1).values
        pesos = torch.softmax(self.atencion(h).squeeze(-1).masked_fill(mascara == 0, -1e4), dim=1)
        atendido = (pesos[..., None] * h).sum(1)
        z = torch.cat([suma, media, maximo, atendido, self.hogar(hogar)], dim=1)
        return self.cabeza(z).squeeze(-1)


def lotes_por_tamano(conjuntos, ids, tamano, rng=None):
    """Lotes de hogares de tamaño parecido (ordenados por número de personas, con
    ruido aleatorio en el ajuste para que los lotes cambien entre épocas)."""
    conteo = conjuntos.conteo.loc[ids].to_numpy().astype(float)
    if rng is not None:
        conteo = conteo + rng.uniform(0, 0.99, len(conteo))
    orden = np.argsort(conteo, kind="stable")
    bloques = [orden[i:i + tamano] for i in range(0, len(orden), tamano)]
    if rng is not None:
        rng.shuffle(bloques)
    return bloques


def predecir_conjuntos(modelo, conjuntos, ids, estadisticas, tamano=2048):
    """P(pobre) de los hogares `ids` con el modelo (sin gradientes)."""
    modelo.eval()
    salida = np.zeros(len(ids))
    ids = np.asarray(ids)
    with torch.no_grad():
        for bloque in lotes_por_tamano(conjuntos, ids, tamano):
            salida[bloque] = torch.sigmoid(modelo(*conjuntos.lote(ids[bloque], *estadisticas))).numpy()
    return salida


def entrenar_conjuntos(conjuntos, ids_ajuste, y_ajuste, ids_parada, y_parada, parametros, semilla):
    """Entrena la red con Adam y parada temprana por log loss en la parada.

    Devuelve: (modelo con los mejores pesos, estadísticas de estandarización,
        historia por época).
    """
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    estadisticas = conjuntos.estadisticas(ids_ajuste)
    modelo = RedConjuntos(conjuntos.x_personas.shape[1], conjuntos.x_hogar.shape[1], parametros["d_oculto"],
                          parametros["d_salida"], parametros["p_drop"])
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=parametros["lr"], weight_decay=parametros["weight_decay"])
    perdida = nn.BCEWithLogitsLoss()
    ids_ajuste, y_ajuste = np.asarray(ids_ajuste), np.asarray(y_ajuste, dtype=np.float32)
    mejor, mejor_estado, sin_mejora, historia = np.inf, None, 0, []
    for epoca in range(parametros["max_epocas"]):
        inicio = time.time()
        modelo.train()
        for bloque in lotes_por_tamano(conjuntos, ids_ajuste, parametros["lote"], rng):
            personas, mascara, hogar = conjuntos.lote(ids_ajuste[bloque], *estadisticas)
            optimizador.zero_grad()
            perdida(modelo(personas, mascara, hogar), torch.from_numpy(y_ajuste[bloque])).backward()
            optimizador.step()
        logloss = log_loss(y_parada, predecir_conjuntos(modelo, conjuntos, ids_parada, estadisticas))
        historia.append({"epoca": epoca + 1, "log_loss_parada": logloss, "segundos": time.time() - inicio})
        if logloss < mejor - 1e-5:
            mejor, mejor_estado, sin_mejora = logloss, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= parametros["paciencia"]:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, estadisticas, historia


def crear_ajuste_conjuntos(conjuntos, parametros=PARAMETROS_CONJUNTOS, semilla=SEMILLA, historias=None):
    """Función ajustar() para evaluar_modelo(): x_ent y x_eval solo aportan los ids
    (índice); la estructura de conjuntos vive en la clausura.

    Argumentos: conjuntos = Conjuntos; parametros = PARAMETROS_CONJUNTOS; semilla;
        historias = lista donde se anotan las historias de entrenamiento.
    Devuelve: función(x_ent, y_ent, x_eval) -> (probabilidad, None).
    """
    torch.set_num_threads(max(N_NUCLEOS_FISICOS - 1, 1))

    def ajustar(x_ent, y_ent, x_eval):
        y_ent = np.asarray(y_ent)
        ajuste, parada = particion_parada(y_ent, semilla)
        ids = np.asarray(x_ent.index)
        modelo, estadisticas, historia = entrenar_conjuntos(conjuntos, ids[ajuste], y_ent[ajuste], ids[parada],
                                                           y_ent[parada], parametros, semilla)
        if historias is not None:
            historias.append(historia)
        return predecir_conjuntos(modelo, conjuntos, np.asarray(x_eval.index), estadisticas), None

    return ajustar
