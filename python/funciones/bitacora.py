"""Log por etapa del nivel 7 con hora y RAM (stores/logs/nivel7.log)."""

from datetime import datetime
import time

import psutil

from configuracion import RUTA_LOGS

ARCHIVO_LOG = RUTA_LOGS / "nivel7.log"


def registrar_log(etapa, evento, inicio=None):
    """Agrega una línea al log con la hora, la RAM del proceso y la del sistema.

    Argumentos: etapa = nombre corto de la etapa o candidato; evento = texto;
        inicio = time.time() del comienzo, para anotar los minutos transcurridos.
    Devuelve: la línea escrita.
    """
    proceso = psutil.Process().memory_info().rss / 1024**3
    sistema = psutil.virtual_memory()
    usado = (sistema.total - sistema.available) / 1024**3
    linea = (
        f"{datetime.now():%Y-%m-%d %H:%M:%S} | {etapa:<26} | {evento:<34} | "
        f"proceso {proceso:5.2f} GB | sistema {usado:5.1f} / {sistema.total / 1024**3:4.1f} GB"
    )
    if inicio is not None:
        linea += f" | {(time.time() - inicio) / 60:.1f} min"
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as archivo:
        archivo.write(linea + "\n")
    print(linea, flush=True)
    return linea


class Presupuesto:
    """Reloj de una etapa: avisa cuándo se pasó del tiempo máximo.

    Argumentos: minutos = tiempo máximo de la etapa.
    """

    def __init__(self, minutos):
        self.inicio = time.time()
        self.minutos = minutos

    def transcurrido(self):
        return (time.time() - self.inicio) / 60

    def agotado(self):
        return self.transcurrido() >= self.minutos
