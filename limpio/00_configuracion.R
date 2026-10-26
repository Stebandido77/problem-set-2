# =============================================================================
# 00_configuracion.R — Etapa 17 (diseño limpio)
# Paquetes, rutas, semilla y constantes compartidas por todos los scripts de
# limpio/. Se corre con:
#   "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" limpio/main.R
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(lightgbm)
  library(here)
})

SEMILLA <- 20261025
NUM_HILOS <- 6

# Modo prueba: corre todo en miniatura (5 % de los hogares, pocas rondas) en una
# carpeta aparte para verificar el flujo antes de la corrida real.
MODO_PRUEBA <- "--prueba" %in% commandArgs(trailingOnly = TRUE)

RUTA_RAW <- here::here("stores", "raw")
RUTA_IDS <- here::here("limpio", "ids")
RUTA_SALIDAS <- if (MODO_PRUEBA) {
  here::here("limpio", "salidas_prueba")
} else {
  here::here("limpio", "salidas")
}
dir.create(RUTA_SALIDAS, showWarnings = FALSE, recursive = TRUE)

# Búsqueda de hiperparámetros y rondas
N_CONFIGURACIONES <- if (MODO_PRUEBA) 2 else 12
MAX_RONDAS <- if (MODO_PRUEBA) 60 else 3000
PARADA_TEMPRANA <- if (MODO_PRUEBA) 10 else 100

# Dominio ausente en el test
DOMINIO_AUSENTE_TEST <- "BOGOTA"

# Variables categóricas de la base (se pasan como enteros a LightGBM)
VARIABLES_CATEGORICAS <- c("dominio", "tenencia", "posicion_jefe", "oficio_jefe")

source(here::here("limpio", "funciones", "limpieza.R"))
source(here::here("limpio", "funciones", "construir_base.R"))
source(here::here("limpio", "funciones", "particiones.R"))
source(here::here("limpio", "funciones", "metricas.R"))
source(here::here("limpio", "funciones", "modelos.R"))
source(here::here("limpio", "funciones", "auxiliar.R"))

# Escribe un mensaje con hora en consola y en el log de la corrida.
registrar <- function(...) {
  linea <- paste0(format(Sys.time(), "%H:%M:%S"), " | ", paste0(...))
  cat(linea, "\n")
  cat(linea, "\n", file = file.path(RUTA_SALIDAS, "log.txt"), append = TRUE)
}
