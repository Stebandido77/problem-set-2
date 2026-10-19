"""Colonia artificial de abejas (ABC, Karaboga 2005) para selección de variables
e hiperparámetros a la vez.

Cada fuente de alimento es un vector z de largo n_binarias + n_continuas: las
primeras n_binarias posiciones son una máscara de variables (0/1) y las demás
son hiperparámetros codificados en [0, 1]. Cada ciclo tiene tres fases:
  - abejas empleadas: cada fuente prueba una vecina (se mueve hacia o desde otra
    fuente al azar) y se queda con la mejor (selección codiciosa);
  - abejas observadoras: eligen fuentes con probabilidad proporcional a su
    aptitud y repiten el movimiento sobre ellas;
  - exploradora: la fuente con más intentos fallidos por encima del límite de
    abandono se reemplaza por una al azar.
Movimiento: en una posición continua j, v_j = z_j + phi (z_j - z_kj) con phi en
[-1, 1] (fórmula original); en las binarias (ABC binario con sigmoide), un
subconjunto al azar de bits toma 1 con probabilidad sigmoide(10 (z_j + phi
(z_j - z_kj) - 0,5)), así que un bit que coincide con la vecina casi nunca
cambia y uno que difiere cambia con probabilidad creciente en |phi|.

Reanudable: cada evaluación se guarda en un JSONL (caché por huella del vector)
y el estado de la colonia (fuentes, aptitudes, intentos, ciclo y estado del
generador aleatorio) se guarda al cerrar cada ciclo.
"""

import hashlib
import json

import numpy as np


def huella(z):
    """Identificador estable de un vector solución."""
    return hashlib.sha1(np.round(np.asarray(z, float), 6).tobytes()).hexdigest()


class ColoniaAbejas:
    """ABC con caché de evaluaciones y estado en disco.

    Argumentos:
      aptitud      = función(z) -> (aptitud, detalle); se maximiza
      n_binarias   = largo de la máscara de variables
      n_continuas  = número de hiperparámetros codificados en [0, 1]
      n_fuentes    = fuentes de alimento (= abejas empleadas = observadoras)
      limite       = intentos sin mejora antes de abandonar una fuente
      ruta         = carpeta para el estado y la caché
      semilla      = semilla del generador
      iniciales    = vectores que entran a la población inicial (p. ej. todas
                     las variables con hiperparámetros por defecto)
      bits_por_movimiento = cuántos bits de la máscara se tocan en cada movimiento
      registrar    = función(texto) para el log
    """

    def __init__(self, aptitud, n_binarias, n_continuas, n_fuentes, limite, ruta, semilla,
                 iniciales=(), bits_por_movimiento=3, registrar=print):
        self.aptitud_externa = aptitud
        self.n_binarias, self.dimension = n_binarias, n_binarias + n_continuas
        self.n_fuentes, self.limite = n_fuentes, limite
        self.bits = bits_por_movimiento
        self.registrar = registrar
        self.ruta = ruta
        self.ruta.mkdir(parents=True, exist_ok=True)
        self.ruta_cache = ruta / "evaluaciones.jsonl"
        self.ruta_estado = ruta / "estado.json"
        self.cache = {}
        if self.ruta_cache.exists():
            for linea in self.ruta_cache.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    fila = json.loads(linea)
                    self.cache[fila["huella"]] = fila
        if self.ruta_estado.exists():
            estado = json.loads(self.ruta_estado.read_text(encoding="utf-8"))
            self.fuentes = np.array(estado["fuentes"])
            self.aptitudes = np.array(estado["aptitudes"])
            self.intentos = np.array(estado["intentos"])
            self.ciclo = estado["ciclo"]
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = estado["rng"]
            self.registrar(f"ABC retomada en el ciclo {self.ciclo} ({len(self.cache)} evaluaciones en caché)")
        else:
            self.rng = np.random.default_rng(semilla)
            self.ciclo = 0
            self.fuentes = np.array([self.fuente_al_azar() for _ in range(n_fuentes)])
            for i, z in enumerate(iniciales):
                self.fuentes[i] = z
            self.aptitudes = np.array([self.evaluar(z, "inicial") for z in self.fuentes])
            self.intentos = np.zeros(n_fuentes, int)
            self.guardar_estado()

    def fuente_al_azar(self):
        """Máscara con densidad al azar entre 0,4 y 0,9 e hiperparámetros uniformes."""
        densidad = self.rng.uniform(0.4, 0.9)
        mascara = (self.rng.random(self.n_binarias) < densidad).astype(float)
        if mascara.sum() == 0:
            mascara[self.rng.integers(self.n_binarias)] = 1
        return np.concatenate([mascara, self.rng.random(self.dimension - self.n_binarias)])

    def evaluar(self, z, fase):
        """Aptitud de z, desde la caché si ya se evaluó (y se anota si no)."""
        clave = huella(z)
        if clave not in self.cache:
            valor, detalle = self.aptitud_externa(z)
            fila = {"huella": clave, "aptitud": valor, "ciclo": self.ciclo, "fase": fase,
                    "n_variables": int(np.asarray(z[:self.n_binarias]).sum()), **detalle,
                    "z": np.round(np.asarray(z, float), 6).tolist()}
            with open(self.ruta_cache, "a", encoding="utf-8") as archivo:
                archivo.write(json.dumps(fila) + "\n")
            self.cache[clave] = fila
            mejor = max(f["aptitud"] for f in self.cache.values())
            self.registrar(f"ciclo {self.ciclo} {fase}: aptitud {valor:.4f} | mejor {mejor:.4f} | "
                           f"{fila['n_variables']} variables | {len(self.cache)} evaluaciones")
        return self.cache[clave]["aptitud"]

    def vecina(self, i):
        """Movimiento de una abeja desde la fuente i hacia/desde otra fuente k."""
        k = self.rng.choice([j for j in range(self.n_fuentes) if j != i])
        z, otra = self.fuentes[i].copy(), self.fuentes[k]
        # Una posición continua, con la fórmula original
        j = self.rng.integers(self.n_binarias, self.dimension)
        z[j] = np.clip(z[j] + self.rng.uniform(-1, 1) * (z[j] - otra[j]), 0, 1)
        # Algunos bits de la máscara, con la versión binaria (sigmoide)
        for b in self.rng.choice(self.n_binarias, size=min(self.bits, self.n_binarias), replace=False):
            paso = z[b] + self.rng.uniform(-1, 1) * (z[b] - otra[b])
            z[b] = float(self.rng.random() < 1 / (1 + np.exp(-10 * (paso - 0.5))))
        if z[:self.n_binarias].sum() == 0:
            z[self.rng.integers(self.n_binarias)] = 1
        return z

    def intentar(self, i, fase):
        """Evalúa una vecina de la fuente i y se queda con la mejor."""
        z = self.vecina(i)
        valor = self.evaluar(z, fase)
        if valor > self.aptitudes[i]:
            self.fuentes[i], self.aptitudes[i], self.intentos[i] = z, valor, 0
        else:
            self.intentos[i] += 1

    def correr_ciclo(self):
        """Un ciclo completo: empleadas, observadoras y exploradora."""
        for i in range(self.n_fuentes):
            self.intentar(i, "empleada")
        probabilidad = np.clip(self.aptitudes, 1e-9, None)
        probabilidad = probabilidad / probabilidad.sum()
        for i in self.rng.choice(self.n_fuentes, size=self.n_fuentes, p=probabilidad):
            self.intentar(i, "observadora")
        peor = int(np.argmax(self.intentos))
        if self.intentos[peor] > self.limite:
            self.fuentes[peor] = self.fuente_al_azar()
            self.aptitudes[peor] = self.evaluar(self.fuentes[peor], "exploradora")
            self.intentos[peor] = 0
        self.ciclo += 1
        self.guardar_estado()

    def guardar_estado(self):
        """Escribe el estado de la colonia (atómico: temporal + renombrar)."""
        estado = {"fuentes": self.fuentes.tolist(), "aptitudes": self.aptitudes.tolist(),
                  "intentos": self.intentos.tolist(), "ciclo": self.ciclo,
                  "rng": self.rng.bit_generator.state}
        temporal = self.ruta_estado.with_suffix(".tmp")
        temporal.write_text(json.dumps(estado), encoding="utf-8")
        temporal.replace(self.ruta_estado)
        # La mejor solución de toda la corrida, aparte, para leerla sin la caché
        mejor = max(self.cache.values(), key=lambda f: f["aptitud"])
        temporal = (self.ruta / "mejor.json").with_suffix(".tmp")
        temporal.write_text(json.dumps({**mejor, "ciclo_guardado": self.ciclo}), encoding="utf-8")
        temporal.replace(self.ruta / "mejor.json")

    def mejor(self):
        """(vector, aptitud) de la mejor solución evaluada en toda la corrida (la
        memoria de la colonia: una exploradora pudo haber abandonado su fuente)."""
        fila = max(self.cache.values(), key=lambda f: f["aptitud"])
        return np.array(fila["z"]), float(fila["aptitud"])
