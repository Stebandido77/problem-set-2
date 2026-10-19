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
  lightgbm,    # boosting (nivel complejo)
  bonsai,      # conecta lightgbm con parsnip
  discrim,     # naive bayes en tidymodels
  naivebayes,
  knitr,       # exportar tablas a LaTeX
  future,      # paralelización del tuning (tune >= 2.0 ya no usa foreach)
  butcher,     # adelgaza los modelos ajustados antes de guardarlos
  ps           # RAM de la sesión y de los workers para el log de la corrida
)
# Nota: vip salió de CRAN; la importancia de variables se calcula directamente
# desde cada motor en calcular_importancia() (funciones/modelado.R).

# ---- Parámetros globales ----------------------------------------------------
SEMILLA <- 2026
N_FOLDS <- 5
EQUIPO  <- "06"

# Proporción del train que se separa como holdout intocable
PROPORCION_HOLDOUT <- 0.2

# Brecha F1_cv - F1_holdout a partir de la cual se marca sobreajuste
UMBRAL_SOBREAJUSTE <- 0.02

# Top 3 para Kaggle elegido a mano, en orden de posición. n3_logit no entra por
# F1 (hay modelos con más F1 en holdout): entra por interpretabilidad, como
# benchmark lineal para la discusión de política. Con NULL, elegir_top3() aplica
# la regla automática (un representante por algoritmo según el F1 en holdout).
TOP3_MANUAL <- c("n6_lightgbm", "n4_random_forest", "n3_logit")

# Paralelismo. Hay dos fuentes y se usan según el algoritmo (ver
# configurar_paralelismo() en funciones/escalera.R):
#   - Hilos del motor (ranger, xgboost, lightgbm): los folds corren en secuencia y
#     cada ajuste usa N_NUCLEOS hilos. Los hilos comparten la misma copia de los
#     datos, así que no multiplican la RAM.
#   - Workers de future (Logit, Elastic Net, CART, Naive Bayes): cada worker es una
#     sesión de R con su propia copia de los folds, por eso se limitan a
#     N_WORKERS_MAX. Con 10 workers sobre la base completa una máquina de 23 GB se
#     quedó sin memoria.
# Ambos se pueden cambiar sin tocar el código con PS2_NUCLEOS y PS2_WORKERS.
N_NUCLEOS     <- as.integer(Sys.getenv("PS2_NUCLEOS", max(parallel::detectCores() - 1, 1)))
N_WORKERS_MAX <- as.integer(Sys.getenv("PS2_WORKERS", 4))

# Modo rápido: tunea con una submuestra estratificada y grillas pequeñas para
# validar que todo corre de punta a punta. Se puede activar sin tocar el código
# con la variable de entorno PS2_MODO_RAPIDO=TRUE.
MODO_RAPIDO     <- Sys.getenv("PS2_MODO_RAPIDO", "FALSE") == "TRUE"
FRACCION_RAPIDA <- 0.15

# ---- Rutas -------------------------------------------------------------------
RUTA_CRUDOS      <- here("stores", "raw")
RUTA_PROCESADOS  <- here("stores", "procesados")
RUTA_RESULTADOS  <- here("stores", "resultados")
RUTA_MODELOS     <- here("stores", "modelos")
RUTA_SUBMISSIONS <- here("stores", "submissions")
RUTA_FIGURAS     <- here("views", "figuras")
RUTA_TABLAS      <- here("views", "tablas")
RUTA_LOGS        <- here("stores", "logs")

# En modo rápido todo lo que produce la escalera va a stores/rapido/ para no
# mezclarse con (ni reutilizarse en) la corrida completa.
if (MODO_RAPIDO) {
  RUTA_RESULTADOS  <- here("stores", "rapido", "resultados")
  RUTA_MODELOS     <- here("stores", "rapido", "modelos")
  RUTA_SUBMISSIONS <- here("stores", "rapido", "submissions")
  RUTA_FIGURAS     <- here("stores", "rapido", "figuras")
  RUTA_TABLAS      <- here("stores", "rapido", "tablas")
  RUTA_LOGS        <- here("stores", "rapido", "logs")
}

walk(
  c(RUTA_PROCESADOS, RUTA_RESULTADOS, RUTA_MODELOS, RUTA_FIGURAS, RUTA_TABLAS, RUTA_LOGS,
    file.path(RUTA_SUBMISSIONS, c("candidatos", "top3"))),
  dir.create, recursive = TRUE, showWarnings = FALSE
)

# ---- Funciones del proyecto --------------------------------------------------
archivos_funciones <- list.files(
  here("scripts", "funciones"), pattern = "\\.R$", full.names = TRUE
)
walk(archivos_funciones, source)

# ---- Paralelización ----------------------------------------------------------
# Registra workers de future; tune los usa automáticamente en tune_grid() y
# fit_resamples(). Conserva la firma de la versión anterior (con doParallel),
# pero nunca pasa de N_WORKERS_MAX.
registrar_nucleos <- function(nucleos = N_NUCLEOS) {
  # Las bases de entrenamiento superan el límite por defecto (500 MB) de future
  options(future.globals.maxSize = 4 * 1024^3)
  plan(multisession, workers = max(min(nucleos, N_WORKERS_MAX), 1))
  invisible(nucleos)
}

set.seed(SEMILLA)
theme_set(tema_proyecto())

if (MODO_RAPIDO) message(">> MODO_RAPIDO activo: submuestra del ", FRACCION_RAPIDA * 100, "% y grillas pequeñas")
