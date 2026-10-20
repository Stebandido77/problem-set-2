"""Log por etapa del nivel 7 con hora y RAM (stores/logs/nivel7.log)."""

from datetime import datetime
import os
import threading
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

    Argumentos: minutos = tiempo máximo de la etapa; ruta_inicio = archivo
        opcional donde se guarda la hora de arranque, para que un relanzamiento
        (por memoria o por error) siga contando desde el primer arranque.
    """

    def __init__(self, minutos, ruta_inicio=None):
        self.minutos = minutos
        self.inicio = time.time()
        if ruta_inicio is not None:
            if ruta_inicio.exists():
                self.inicio = float(ruta_inicio.read_text(encoding="utf-8"))
            else:
                ruta_inicio.parent.mkdir(parents=True, exist_ok=True)
                ruta_inicio.write_text(repr(self.inicio), encoding="utf-8")

    def restante(self):
        return self.minutos - self.transcurrido()

    def transcurrido(self):
        return (time.time() - self.inicio) / 60

    def agotado(self):
        return self.transcurrido() >= self.minutos


# Límites de memoria: por encima (proceso) o por debajo (sistema) se sale limpio
LIMITE_PROCESO_GB = 6.0
MINIMO_LIBRE_SISTEMA_GB = 2.0
CODIGO_SALIDA_MEMORIA = 3
CODIGO_SALIDA_TIEMPO = 4


def motivo_memoria():
    """Texto con el motivo si la memoria pasó algún límite; None si todo va bien."""
    proceso = psutil.Process().memory_info().rss / 1024**3
    libre = psutil.virtual_memory().available / 1024**3
    if proceso > LIMITE_PROCESO_GB:
        return f"el proceso usa {proceso:.2f} GB (límite {LIMITE_PROCESO_GB} GB)"
    if libre < MINIMO_LIBRE_SISTEMA_GB:
        return f"el sistema tiene {libre:.2f} GB libres (mínimo {MINIMO_LIBRE_SISTEMA_GB} GB)"
    return None


def iniciar_vigilante_memoria(etapa, intervalo=5):
    """Hilo que revisa la memoria cada `intervalo` segundos y, si pasa un límite,
    lo anota en el log y termina el proceso con CODIGO_SALIDA_MEMORIA.

    Salir así es limpio porque todo el avance ya está en disco (cada fold, ajuste
    auxiliar y evaluación se guarda al terminar, con escritura atómica): lo que
    se pierde es solo el ajuste en curso, y al relanzar se retoma desde ahí.
    Argumentos: etapa = nombre para el log; intervalo = segundos entre revisiones.
    """
    def vigilar():
        while True:
            motivo = motivo_memoria()
            if motivo:
                registrar_log(etapa, f"SALIDA POR MEMORIA: {motivo}; se retoma al relanzar")
                os._exit(CODIGO_SALIDA_MEMORIA)
            time.sleep(intervalo)

    threading.Thread(target=vigilar, daemon=True, name="vigilante_memoria").start()


def iniciar_corte_duro(etapa, reloj, desactivar, intervalo=10):
    """Hilo que termina el proceso (CODIGO_SALIDA_TIEMPO) cuando el reloj se agota.

    Es la red de seguridad del tope de tiempo: el plan de la etapa ya deja
    reserva para el cierre, y todo lo terminado está en disco. Al empezar el
    cierre (que no entrena nada) quien llama hace desactivar.set().
    Argumentos: etapa = nombre para el log; reloj = Presupuesto;
        desactivar = threading.Event; intervalo = segundos entre revisiones.
    """
    def vigilar():
        while not desactivar.is_set():
            if reloj.agotado():
                registrar_log(etapa, f"CORTE DURO: {reloj.minutos} min agotados")
                os._exit(CODIGO_SALIDA_TIEMPO)
            time.sleep(intervalo)

    threading.Thread(target=vigilar, daemon=True, name="corte_duro").start()
