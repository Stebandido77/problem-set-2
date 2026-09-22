# ==============================================================================
# 00_configuracion.R
# Carga paquetes, define rutas y parámetros globales, y trae las funciones del
# proyecto. Todos los demás scripts arrancan con source() de este archivo.
# ==============================================================================

if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  tidyverse,   # manipulación y gráficos
  here,        # rutas relativas al proyecto
  janitor,     # limpieza de nombres
  skimr,       # resúmenes descriptivos
  tidymodels,  # recetas, validación cruzada, tuning
  themis,      # balanceo de clases (SMOTE, downsampling)
  glmnet,      # elastic net
  rpart,       # CART
  rpart.plot,
  ranger,      # random forest
  xgboost,     # boosting
  discrim,     # naive bayes en tidymodels
  naivebayes,
  vip,         # importancia de variables
  knitr,       # exportar tablas a LaTeX
  doParallel   # paralelización del tuning
)

# ---- Parámetros globales ----------------------------------------------------
SEMILLA <- 2026
N_FOLDS <- 5
EQUIPO  <- "06"

# ---- Rutas -------------------------------------------------------------------
RUTA_CRUDOS      <- here("stores", "raw")
RUTA_PROCESADOS  <- here("stores", "procesados")
RUTA_RESULTADOS  <- here("stores", "resultados")
RUTA_SUBMISSIONS <- here("stores", "submissions")
RUTA_FIGURAS     <- here("views", "figuras")
RUTA_TABLAS      <- here("views", "tablas")

# ---- Funciones del proyecto --------------------------------------------------
archivos_funciones <- list.files(
  here("scripts", "funciones"), pattern = "\\.R$", full.names = TRUE
)
walk(archivos_funciones, source)

# ---- Paralelización ----------------------------------------------------------
registrar_nucleos <- function(nucleos = parallel::detectCores() - 1) {
  registerDoParallel(cores = max(nucleos, 1))
  invisible(nucleos)
}

set.seed(SEMILLA)
theme_set(tema_proyecto())
