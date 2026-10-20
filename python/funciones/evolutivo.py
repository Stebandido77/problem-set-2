"""Bagging de árboles evolutivos: en cada bolsa, un algoritmo genético sobre
árboles de clasificación completos, con cruce de subárboles y mutación.

La aptitud de un árbol es exactamente la métrica del submission: el F1 de la
clase pobre con el umbral óptimo de buscar_umbral_f1() (funciones/metricas.py),
calculado sobre las observaciones out-of-bag de la bolsa, menos una penalización
leve por profundidad. Las probabilidades de hoja salen solo de las filas de la
bolsa; las out-of-bag nunca entran al ajuste del árbol.

Representación: árbol binario completo en arreglo (montículo), nodo i con hijos
2i + 1 y 2i + 2. variable[i] >= 0 es un nodo interno (va a la izquierda si el
valor discretizado es <= umbral[i]), -1 es una hoja y -2 un nodo que no existe.
Los predictores se discretizan en cuantiles del entrenamiento del contexto.
"""

import numpy as np

from funciones.metricas import buscar_umbral_f1

HOJA = -1
VACIO = -2


# ---- Métrica ------------------------------------------------------------------

def aptitud_f1(verdad, probabilidad):
    """F1 de la clase pobre con umbral óptimo, idéntico al del pipeline.

    Envuelve buscar_umbral_f1() sin reimplementarlo, para que la selección del
    genético use la misma métrica que el submission (nunca accuracy, AUC ni F1
    a umbral 0,5).
    Argumentos: verdad = arreglo 0/1 (1 = pobre); probabilidad = P(pobre).
    Devuelve: (umbral óptimo, F1 en ese umbral).
    """
    return buscar_umbral_f1(verdad, probabilidad)


# ---- Discretización -----------------------------------------------------------

def calcular_cortes(x_ent, n_cortes):
    """Puntos de corte por variable: cuantiles únicos del entrenamiento.

    Argumentos: x_ent = DataFrame numérico sin NA; n_cortes = cuantiles por variable.
    Devuelve: lista de arreglos de cortes, uno por columna (puede ser vacío).
    """
    cuantiles = np.arange(1, n_cortes) / n_cortes
    cortes = []
    for columna in x_ent.columns:
        valores = x_ent[columna].to_numpy(np.float64)
        candidatos = np.unique(np.quantile(valores, cuantiles))
        # El último corte igual al máximo no separa a nadie
        cortes.append(candidatos[candidatos < valores.max()])
    return cortes


def discretizar(x, cortes):
    """Valor discretizado de cada celda: cuántos cortes son < al valor.

    Ir a la izquierda con discretizado <= t equivale a x <= cortes[t].
    Argumentos: x = DataFrame con las columnas de cortes; cortes = calcular_cortes().
    Devuelve: arreglo uint8 (filas x columnas).
    """
    salida = np.empty(x.shape, dtype=np.uint8)
    for j, columna in enumerate(x.columns):
        salida[:, j] = np.searchsorted(cortes[j], x[columna].to_numpy(np.float64), side="left")
    return salida


# ---- Árboles ------------------------------------------------------------------

def profundidad_nodo(i):
    """Profundidad del nodo i del montículo (raíz = 0)."""
    return (i + 1).bit_length() - 1


def crear_vacio(profundidad_maxima):
    """Árbol de solo raíz-hoja: (variable, umbral) de tamaño 2^(D+1) - 1."""
    n_nodos = 2 ** (profundidad_maxima + 1) - 1
    variable = np.full(n_nodos, VACIO, dtype=np.int16)
    variable[0] = HOJA
    return variable, np.zeros(n_nodos, dtype=np.int16)


def profundidad_arbol(variable):
    """Número de niveles de nodos internos (0 si la raíz es hoja)."""
    internos = np.flatnonzero(variable >= 0)
    return 0 if len(internos) == 0 else profundidad_nodo(int(internos.max())) + 1


def borrar_subarbol(variable, umbral, i):
    """Marca como inexistentes el nodo i y todos sus descendientes."""
    pila = [i]
    while pila:
        k = pila.pop()
        if k >= len(variable) or variable[k] == VACIO:
            continue
        if variable[k] >= 0:
            pila += [2 * k + 1, 2 * k + 2]
        variable[k], umbral[k] = VACIO, 0


def partir(variable, umbral, i, v, t):
    """Convierte la hoja i en nodo interno (v, t) con dos hojas hijas."""
    variable[i], umbral[i] = v, t
    variable[2 * i + 1] = variable[2 * i + 2] = HOJA


def trasplantar(destino, i, fuente, j, profundidad_maxima):
    """Copia el subárbol j de fuente en el nodo i de destino (truncado a la
    profundidad máxima: lo que sobra se vuelve hoja).

    Argumentos: destino, fuente = tuplas (variable, umbral); destino se modifica.
    """
    var_d, umb_d = destino
    var_f, umb_f = fuente
    borrar_subarbol(var_d, umb_d, i)
    pila = [(i, j)]
    while pila:
        a, b = pila.pop()
        if var_f[b] >= 0 and profundidad_nodo(a) < profundidad_maxima:
            var_d[a], umb_d[a] = var_f[b], umb_f[b]
            pila += [(2 * a + 1, 2 * b + 1), (2 * a + 2, 2 * b + 2)]
        else:
            var_d[a], umb_d[a] = HOJA, 0


def rutear(variable, umbral, x):
    """Hoja a la que llega cada fila.

    Argumentos: variable, umbral = árbol; x = arreglo discretizado (filas x columnas).
    Devuelve: arreglo con el índice de nodo (hoja) de cada fila.
    """
    filas = np.arange(x.shape[0])
    nodo = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(profundidad_arbol(variable)):
        v = variable[nodo]
        interno = v >= 0
        if not interno.any():
            break
        izquierda = x[filas, np.where(interno, v, 0)] <= umbral[nodo]
        nodo = np.where(interno, 2 * nodo + 1 + (~izquierda), nodo)
    return nodo


def probabilidades_hoja(hojas, y, n_nodos, previo, fuerza):
    """P(pobre) por hoja con suavizado hacia la tasa de la bolsa.

    Argumentos: hojas = rutear() de las filas de la bolsa; y = 0/1 de esas filas;
        n_nodos = tamaño del árbol; previo = tasa de pobres de la bolsa;
        fuerza = pseudo-observaciones del suavizado.
    Devuelve: arreglo de probabilidades por nodo.
    """
    n = np.bincount(hojas, minlength=n_nodos)
    pobres = np.bincount(hojas, weights=y, minlength=n_nodos)
    return (pobres + fuerza * previo) / (n + fuerza)


# ---- Genético -----------------------------------------------------------------

class GeneticoArboles:
    """Algoritmo genético sobre árboles de una bolsa.

    Argumentos:
      x_bolsa, y_bolsa = filas de la bolsa (discretizadas) con que se calculan
                         las probabilidades de hoja
      x_oob, y_oob     = filas out-of-bag con que se calcula la aptitud
      n_umbrales       = cortes disponibles por variable
      pesos_variables  = probabilidad de elegir cada variable al crear o mutar
      config           = diccionario con poblacion, generaciones, profundidad_maxima,
                         prob_cruce, prob_mutacion, torneo, elite, penalizacion,
                         fuerza_suavizado y mejores (árboles que se guardan)
      semilla          = semilla del generador
    """

    def __init__(self, x_bolsa, y_bolsa, x_oob, y_oob, n_umbrales, pesos_variables, config, semilla):
        self.x_bolsa, self.y_bolsa = x_bolsa, y_bolsa.astype(np.float64)
        self.x_oob, self.y_oob = x_oob, y_oob.astype(int)
        self.n_umbrales = np.asarray(n_umbrales)
        self.utiles = np.flatnonzero(self.n_umbrales > 0)
        pesos = np.asarray(pesos_variables, dtype=float)[self.utiles]
        self.pesos = pesos / pesos.sum()
        self.config = config
        self.rng = np.random.default_rng(semilla)
        self.previo = float(self.y_bolsa.mean())
        self.memoria = {}
        self.n_evaluaciones = 0

    # -- operadores --

    def variable_y_corte(self):
        v = int(self.rng.choice(self.utiles, p=self.pesos))
        return v, int(self.rng.integers(self.n_umbrales[v]))

    def arbol_aleatorio(self):
        """Árbol aleatorio de profundidad tope sorteada en [2, D] (ramped half-and-half)."""
        d_max = self.config["profundidad_maxima"]
        tope = int(self.rng.integers(2, d_max + 1))
        completo = self.rng.random() < 0.5
        variable, umbral = crear_vacio(d_max)
        pila = [0]
        while pila:
            i = pila.pop()
            d = profundidad_nodo(i)
            if d < tope and (d == 0 or completo or self.rng.random() < 0.6):
                partir(variable, umbral, i, *self.variable_y_corte())
                pila += [2 * i + 1, 2 * i + 2]
        return variable, umbral

    def mutar(self, arbol):
        """Una mutación al azar: cambiar variable, mover corte, podar o crecer."""
        variable, umbral = arbol
        internos = np.flatnonzero(variable >= 0)
        hojas = [i for i in np.flatnonzero(variable == HOJA)
                 if profundidad_nodo(int(i)) < self.config["profundidad_maxima"]]
        operaciones = []
        if len(internos):
            operaciones += ["variable", "corte"]
        if len(internos) > 1:
            operaciones.append("podar")
        if hojas:
            operaciones.append("crecer")
        operacion = operaciones[int(self.rng.integers(len(operaciones)))]
        if operacion == "variable":
            i = int(self.rng.choice(internos))
            variable[i], umbral[i] = self.variable_y_corte()
        elif operacion == "corte":
            i = int(self.rng.choice(internos))
            paso = int(self.rng.integers(-3, 4))
            umbral[i] = np.clip(umbral[i] + paso, 0, self.n_umbrales[variable[i]] - 1)
        elif operacion == "podar":
            i = int(self.rng.choice(internos[internos > 0]))
            borrar_subarbol(variable, umbral, i)
            variable[i] = HOJA
        else:
            i = int(self.rng.choice(hojas))
            partir(variable, umbral, i, *self.variable_y_corte())
        return variable, umbral

    def cruzar(self, padre, madre):
        """Intercambio de subárboles: un nodo al azar de cada padre (no la raíz del
        receptor) y cada hijo recibe el subárbol del otro."""
        d_max = self.config["profundidad_maxima"]
        hijo = (padre[0].copy(), padre[1].copy())
        hija = (madre[0].copy(), madre[1].copy())
        nodos_padre = np.flatnonzero(padre[0] != VACIO)
        nodos_madre = np.flatnonzero(madre[0] != VACIO)
        i = int(self.rng.choice(nodos_padre[nodos_padre > 0])) if len(nodos_padre) > 1 else 0
        j = int(self.rng.choice(nodos_madre[nodos_madre > 0])) if len(nodos_madre) > 1 else 0
        trasplantar(hijo, i, madre, j, d_max)
        trasplantar(hija, j, padre, i, d_max)
        return hijo, hija

    # -- aptitud --

    def ajustar_hojas(self, arbol):
        """Probabilidades de hoja del árbol con las filas de la bolsa."""
        variable, umbral = arbol
        hojas = rutear(variable, umbral, self.x_bolsa)
        return probabilidades_hoja(hojas, self.y_bolsa, len(variable), self.previo,
                                   self.config["fuerza_suavizado"])

    def evaluar(self, arbol):
        """Aptitud = F1 OOB con umbral óptimo - penalización por profundidad.

        Devuelve: (aptitud, F1 OOB).
        """
        variable, umbral = arbol
        clave = variable.tobytes() + umbral.tobytes()
        if clave not in self.memoria:
            p_hoja = self.ajustar_hojas(arbol)
            prob_oob = p_hoja[rutear(variable, umbral, self.x_oob)]
            f1 = aptitud_f1(self.y_oob, prob_oob)[1]
            self.memoria[clave] = (f1 - self.config["penalizacion"] * profundidad_arbol(variable), f1)
            self.n_evaluaciones += 1
        return self.memoria[clave]

    def torneo(self, poblacion, aptitudes):
        rivales = self.rng.integers(len(poblacion), size=self.config["torneo"])
        return poblacion[int(rivales[np.argmax(aptitudes[rivales])])]

    def correr(self):
        """Evoluciona la población y devuelve los mejores árboles distintos.

        Devuelve: diccionario con arboles (lista de (variable, umbral, p_hoja)),
            aptitud y f1 de cada uno, e historia (mejor aptitud por generación).
        """
        c = self.config
        poblacion = [self.arbol_aleatorio() for _ in range(c["poblacion"])]
        aptitudes = np.array([self.evaluar(a)[0] for a in poblacion])
        historia = [float(aptitudes.max())]
        for _ in range(c["generaciones"]):
            orden = np.argsort(-aptitudes)
            nueva = [(poblacion[k][0].copy(), poblacion[k][1].copy()) for k in orden[:c["elite"]]]
            while len(nueva) < c["poblacion"]:
                padre, madre = self.torneo(poblacion, aptitudes), self.torneo(poblacion, aptitudes)
                if self.rng.random() < c["prob_cruce"]:
                    hijos = self.cruzar(padre, madre)
                else:
                    hijos = ((padre[0].copy(), padre[1].copy()), (madre[0].copy(), madre[1].copy()))
                for hijo in hijos:
                    if self.rng.random() < c["prob_mutacion"]:
                        hijo = self.mutar(hijo)
                    nueva.append(hijo)
            poblacion = nueva[:c["poblacion"]]
            aptitudes = np.array([self.evaluar(a)[0] for a in poblacion])
            historia.append(float(aptitudes.max()))

        elegidos, vistos = [], set()
        for k in np.argsort(-aptitudes):
            clave = poblacion[k][0].tobytes() + poblacion[k][1].tobytes()
            if clave in vistos:
                continue
            vistos.add(clave)
            elegidos.append(k)
            if len(elegidos) == c["mejores"]:
                break
        return {"arboles": [(poblacion[k][0], poblacion[k][1], self.ajustar_hojas(poblacion[k]))
                            for k in elegidos],
                "aptitud": [float(aptitudes[k]) for k in elegidos],
                "f1": [self.evaluar(poblacion[k])[1] for k in elegidos],
                "historia": historia}


def predecir_arboles(arboles, x):
    """Promedio de las probabilidades de hoja de una lista de árboles.

    Argumentos: arboles = lista de (variable, umbral, p_hoja); x = discretizado.
    Devuelve: arreglo de P(pobre).
    """
    return np.mean([p_hoja[rutear(variable, umbral, x)] for variable, umbral, p_hoja in arboles], axis=0)


def muestra_estratificada(y, tamano, rng):
    """Posiciones de una submuestra sin reemplazo que conserva la tasa de pobres.

    Argumentos: y = arreglo 0/1; tamano = filas a tomar; rng = generador.
    Devuelve: arreglo ordenado de posiciones.
    """
    y = np.asarray(y)
    tamano = min(tamano, len(y))
    elegidas = []
    for clase in (0, 1):
        posiciones = np.flatnonzero(y == clase)
        k = int(round(tamano * len(posiciones) / len(y)))
        elegidas.append(rng.choice(posiciones, size=min(k, len(posiciones)), replace=False))
    return np.sort(np.concatenate(elegidas))
