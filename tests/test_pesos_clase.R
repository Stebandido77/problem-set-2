# ==============================================================================
# tests/test_pesos_clase.R
# Verifica con datos sintéticos que el balanceo con pesos de clase hace algo en
# cada motor, en el modo que usa el pipeline: parsnip en clasificación,
# prediciendo probabilidades, con crear_receta(), crear_flujo() y la
# especificación de especificar_algoritmo(). No basta con que el motor acepte el
# argumento: un peso que se acepta pero no se aplica (como class.weights en los
# bosques de probabilidad de ranger) dejaría el nivel 5 empatado por construcción.
# Por eso cada motor tiene que subir P(pobre) de forma material.
#
# Correr desde la raíz del repo:
#   Rscript -e "testthat::test_file('tests/test_pesos_clase.R', stop_on_failure = TRUE)"
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
library(testthat)

# Cuánto tiene que subir P(pobre) promedio para contar como efecto material.
# Con razón ~2,7 los motores que sí aplican el peso la suben entre 0,11 y 0,18.
SUBIDA_MINIMA <- 0.05

# Base sintética con los mismos niveles que la real, una relación no lineal (para
# que los árboles tengan qué partir) y una variable categórica (para las dummies).
crear_datos_sinteticos <- function(n = 8000) {
  set.seed(SEMILLA)
  tibble(
    id = as.character(seq_len(n)),
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = runif(n),
    grupo = factor(sample(c("a", "b", "c"), n, replace = TRUE))
  ) |>
    mutate(
      eta = -2 + 1.1 * x1 - 0.7 * x2 + 1.2 * (x3 > 0.6) + 0.5 * (grupo == "c") + 0.6 * x1 * x2,
      pobre = factor(if_else(runif(n) < plogis(eta), "pobre", "no_pobre"),
                     levels = c("pobre", "no_pobre"))
    ) |>
    select(-eta)
}

datos <- crear_datos_sinteticos()
entrenamiento <- datos[1:6000, ]
evaluacion <- datos[6001:8000, ]
razon <- calcular_razon_pesos(entrenamiento)

# P(pobre) promedio en la evaluación, ajustando como lo hace el pipeline.
# Argumentos: especificacion = salida de especificar_algoritmo();
#   hiperparametros = valores fijos para los tune() (tibble vacío si no hay).
# Devuelve: número entre 0 y 1.
probabilidad_media_pipeline <- function(especificacion, hiperparametros) {
  base <- entrenamiento
  columna_pesos <- NULL
  if (especificacion$mecanismo_pesos == "case_weights") {
    base <- agregar_pesos_caso(base, especificacion$peso_caso)
    columna_pesos <- "peso_caso"
  }
  modelo <- especificacion$modelo
  if (ncol(hiperparametros) > 0) modelo <- finalize_model(modelo, hiperparametros)

  ajuste <- crear_flujo(crear_receta(base), modelo, columna_pesos) |>
    fit(base) |>
    suppressWarnings()  # glm avisa por pesos no enteros: es lo esperado
  mean(predict(ajuste, evaluacion, type = "prob")$.pred_pobre)
}

hiperparametros <- list(
  logit = tibble(),
  elastic_net = tibble(penalty = 1e-3, mixture = 0.5),
  cart = tibble(cost_complexity = 1e-3, tree_depth = 8),
  random_forest = tibble(mtry = 3, min_n = 20),
  xgboost = tibble(trees = 200, tree_depth = 4, learn_rate = 0.1),
  naive_bayes = tibble(smoothness = 1, Laplace = 0),
  lightgbm = tibble(trees = 200, tree_depth = 4, learn_rate = 0.1, min_n = 20, mtry = 1,
                    loss_reduction = 0, sample_size = 1)
)

test_that("los datos sintéticos replican el orden de niveles y el desbalance del proyecto", {
  expect_identical(levels(datos$pobre), c("pobre", "no_pobre"))
  expect_gt(razon, 2)
})

test_that("todos los algoritmos del pipeline tienen un mecanismo de pesos registrado", {
  expect_setequal(names(MOTOR_POR_ALGORITMO), names(hiperparametros))
  for (motor in MOTOR_POR_ALGORITMO) {
    expect_false(peso_pobre_motor(motor, razon)$mecanismo == "ninguno", label = motor)
  }
})

# El test central: el peso tiene que subir P(pobre) en cada motor. Si un motor
# acepta el argumento pero no lo aplica, la subida es ~0 y este test falla.
for (id_algoritmo in names(hiperparametros)) {
  test_that(paste0(id_algoritmo, ": los pesos de clase suben P(pobre) en el modo del pipeline"), {
    sin_pesos <- especificar_algoritmo(id_algoritmo)
    con_pesos <- especificar_algoritmo(id_algoritmo, razon_pesos = razon)
    expect_identical(sin_pesos$mecanismo_pesos, "sin_pesos")
    expect_true(con_pesos$pesos_aplicados)

    p_sin <- probabilidad_media_pipeline(sin_pesos, hiperparametros[[id_algoritmo]])
    p_con <- probabilidad_media_pipeline(con_pesos, hiperparametros[[id_algoritmo]])
    expect_gt(p_con, p_sin + SUBIDA_MINIMA,
              label = sprintf("P(pobre) con pesos en %s (sin pesos: %.3f)", id_algoritmo, p_sin))
  })
}

# Orden de scale_pos_weight: el valor contrario tiene que BAJAR P(pobre). Si una
# versión nueva de xgboost o lightgbm cambia qué clase toma como positiva, falla.
test_that("scale_pos_weight: la positiva es pobre en xgboost y no_pobre en lightgbm", {
  probabilidad_con_peso <- function(id_algoritmo, peso) {
    especificacion <- especificar_algoritmo(id_algoritmo)
    especificacion$modelo <- especificacion$modelo |>
      set_args(scale_pos_weight = !!peso)
    probabilidad_media_pipeline(especificacion, hiperparametros[[id_algoritmo]])
  }
  expect_equal(peso_pobre_motor("xgboost", razon)$valor, razon)
  expect_equal(peso_pobre_motor("lightgbm", razon)$valor, 1 / razon)

  for (id_algoritmo in c("xgboost", "lightgbm")) {
    sin_pesos <- probabilidad_con_peso(id_algoritmo, 1)
    contrario <- probabilidad_con_peso(id_algoritmo, 1 / peso_pobre_motor(
      MOTOR_POR_ALGORITMO[[id_algoritmo]], razon)$valor)
    expect_lt(contrario, sin_pesos - SUBIDA_MINIMA, label = paste(id_algoritmo, "con el peso al revés"))
  }
})

# Documenta por qué ranger usa case_weights: class.weights se acepta pero no
# mueve P(pobre) en un bosque de probabilidad.
test_that("ranger: class.weights no mueve P(pobre) en bosques de probabilidad", {
  especificacion <- especificar_algoritmo("random_forest")
  p_sin <- probabilidad_media_pipeline(especificacion, hiperparametros$random_forest)
  especificacion$modelo <- especificacion$modelo |>
    set_engine("ranger", seed = SEMILLA, num.threads = N_NUCLEOS, class.weights = !!c(razon, 1))
  p_class_weights <- probabilidad_media_pipeline(especificacion, hiperparametros$random_forest)

  expect_lt(abs(p_class_weights - p_sin), 0.01)
  expect_identical(peso_pobre_motor("ranger", razon)$mecanismo, "case_weights")
})

test_that("un motor sin mecanismo verificado avisa y queda marcado", {
  expect_identical(peso_pobre_motor("lm", razon)$mecanismo, "ninguno")

  # Simula un motor sin mecanismo cambiando temporalmente el motor de CART
  original <- MOTOR_POR_ALGORITMO
  on.exit(assign("MOTOR_POR_ALGORITMO", original, envir = globalenv()), add = TRUE)
  motores <- original
  motores[["cart"]] <- "motor_sin_pesos"
  assign("MOTOR_POR_ALGORITMO", motores, envir = globalenv())

  expect_warning(
    especificacion <- especificar_algoritmo("cart", razon_pesos = razon),
    "no tiene un mecanismo de pesos"
  )
  expect_false(especificacion$pesos_aplicados)
  # Sin pesos pedidos no avisa
  expect_no_warning(sin_pesos <- especificar_algoritmo("cart"))
  expect_true(is.na(sin_pesos$pesos_aplicados))
})

test_that("los pesos por fila no cambian las filas de los folds", {
  entrada <- list(
    particion = list(entrenamiento = entrenamiento, holdout = evaluacion),
    folds = vfold_cv(entrenamiento, v = 3, strata = pobre)
  )
  con_pesos <- agregar_pesos_caso_particion(entrada$particion, entrada$folds, razon)

  expect_identical(map(con_pesos$folds$splits, complement), map(entrada$folds$splits, complement))
  expect_true(hardhat::is_case_weights(analysis(con_pesos$folds$splits[[1]])$peso_caso))
  pesos <- as.double(con_pesos$particion$entrenamiento$peso_caso)
  expect_equal(unique(pesos[entrenamiento$pobre == "pobre"]), razon)
  expect_equal(unique(pesos[entrenamiento$pobre == "no_pobre"]), 1)
})
