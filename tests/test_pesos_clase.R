# ==============================================================================
# tests/test_pesos_clase.R
# Verifica con datos sintéticos la convención de pesos de clase de cada motor:
# qué valor hay que pasarle para que la clase pobre pese más. No se asume: se
# mide. Si una actualización de xgboost, lightgbm, parsnip o bonsai cambia qué
# clase toma como "positiva", estos tests fallan antes de una corrida larga.
#
# Correr desde la raíz del repo:
#   Rscript -e 'testthat::test_file("tests/test_pesos_clase.R")'
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
library(testthat)

# Base sintética con los mismos niveles y el mismo desbalance que la real
# (alrededor de 20% de pobres) y una columna id, para poder usar crear_receta().
crear_datos_sinteticos <- function(n = 4000) {
  set.seed(SEMILLA)
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  probabilidad <- plogis(-1.9 + 1.2 * x1 - 0.8 * x2)
  tibble(
    id = as.character(seq_len(n)),
    x1 = x1,
    x2 = x2,
    pobre = factor(if_else(runif(n) < probabilidad, "pobre", "no_pobre"),
                   levels = c("pobre", "no_pobre"))
  )
}

datos <- crear_datos_sinteticos()
razon <- calcular_razon_pesos(datos)

# P(pobre) promedio que predice un modelo ajustado con la receta del proyecto.
# Argumentos: modelo = especificación parsnip sin tune().
# Devuelve: número entre 0 y 1.
probabilidad_media <- function(modelo) {
  ajuste <- workflow() |>
    add_recipe(crear_receta(datos)) |>
    add_model(modelo) |>
    fit(datos)
  mean(predict(ajuste, datos, type = "prob")$.pred_pobre)
}

# Hiperparámetros fijos y razonables para que el efecto de los pesos se vea claro
hiperparametros_xgboost <- tibble(trees = 100, tree_depth = 3, learn_rate = 0.1)
hiperparametros_lightgbm <- tibble(
  trees = 100, tree_depth = 3, learn_rate = 0.1, min_n = 20, mtry = 1,
  loss_reduction = 0, sample_size = 1
)

test_that("los datos sintéticos replican el orden de niveles y el desbalance del proyecto", {
  expect_identical(levels(datos$pobre), c("pobre", "no_pobre"))
  expect_gt(razon, 3)
  expect_lt(razon, 5.5)
})

test_that("xgboost: la clase positiva es pobre (scale_pos_weight = razón)", {
  xgboost_con <- function(peso) {
    boost_tree(trees = 100, tree_depth = 3, learn_rate = 0.1) |>
      set_engine("xgboost", nthread = 2, scale_pos_weight = !!peso) |>
      set_mode("classification")
  }
  sin_pesos <- probabilidad_media(xgboost_con(1))
  con_razon <- probabilidad_media(xgboost_con(razon))
  con_inversa <- probabilidad_media(xgboost_con(1 / razon))

  # Pesar más la clase positiva sube P(pobre) solo si la positiva es pobre
  expect_gt(con_razon, sin_pesos + 0.1)
  expect_lt(con_inversa, sin_pesos - 0.05)
  expect_equal(peso_pobre_motor("xgboost", razon), razon)
})

test_that("lightgbm: la clase positiva es no_pobre (scale_pos_weight = 1 / razón)", {
  lightgbm_con <- function(peso) {
    boost_tree(trees = 100, tree_depth = 3, learn_rate = 0.1, min_n = 20) |>
      set_engine("lightgbm", num_threads = 2, verbose = -1, seed = SEMILLA,
                 scale_pos_weight = !!peso) |>
      set_mode("classification")
  }
  sin_pesos <- probabilidad_media(lightgbm_con(1))
  con_razon <- probabilidad_media(lightgbm_con(razon))
  con_inversa <- probabilidad_media(lightgbm_con(1 / razon))

  # Aquí es al revés que en xgboost: el que sube P(pobre) es 1 / razón
  expect_gt(con_inversa, sin_pesos + 0.1)
  expect_lt(con_razon, sin_pesos - 0.05)
  expect_equal(peso_pobre_motor("lightgbm", razon), 1 / razon)
})

test_that("ranger: class.weights sigue el orden de los niveles (pobre, no_pobre)", {
  # parsnip ajusta bosques de probabilidad, donde class.weights solo entra en la
  # regla de partición y casi no mueve P(pobre). El orden de los pesos se
  # verifica en modo clase (voto por mayoría), donde sí cambia las predicciones.
  share_predicho_pobre <- function(pesos) {
    bosque <- ranger::ranger(
      pobre ~ x1 + x2, data = datos, num.trees = 200, min.node.size = 50,
      class.weights = pesos, probability = FALSE, seed = SEMILLA, num.threads = 2
    )
    mean(bosque$predictions == "pobre")
  }
  con_razon <- share_predicho_pobre(c(razon, 1))
  con_inversa <- share_predicho_pobre(c(1, razon))

  expect_gt(con_razon, con_inversa + 0.1)
  expect_equal(peso_pobre_motor("ranger", razon), c(razon, 1))
  expect_null(peso_pobre_motor("ranger", NULL))
})

test_that("especificar_algoritmo() aplica los pesos en la dirección correcta", {
  for (id_algoritmo in c("xgboost", "lightgbm")) {
    hiperparametros <- if (id_algoritmo == "xgboost") hiperparametros_xgboost else hiperparametros_lightgbm
    sin_pesos <- especificar_algoritmo(id_algoritmo)$modelo |>
      finalize_model(hiperparametros) |>
      probabilidad_media()
    con_pesos <- especificar_algoritmo(id_algoritmo, razon_pesos = razon)$modelo |>
      finalize_model(hiperparametros) |>
      probabilidad_media()

    expect_gt(con_pesos, sin_pesos + 0.1, label = paste(id_algoritmo, "con pesos"))
  }
})

test_that("sin balanceo, los motores no reciben pesos", {
  expect_equal(peso_pobre_motor("xgboost"), 1)
  expect_equal(peso_pobre_motor("lightgbm"), 1)
  expect_error(peso_pobre_motor("glmnet", razon), "sin regla de pesos")
})
