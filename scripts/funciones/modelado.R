# ==============================================================================
# funciones/modelado.R
# Receta común, validación cruzada, tuning y predicción. Todos los algoritmos
# pasan por aquí para garantizar una cancha nivelada (mismas variables y mismo
# preprocesamiento).
#
# Esquema de evaluación de la escalera (evaluar_candidato()):
#   1. Holdout estratificado del 20% que no se toca hasta la evaluación final.
#   2. CV estratificada sobre el 80% restante para elegir hiperparámetros.
#   3. Umbral elegido con predicciones fuera de fold (OOF), nunca con las mismas
#      predicciones con que se mide el F1 reportado.
#   4. Reajuste sobre el 80% -> F1 en holdout con el umbral OOF fijo.
#   5. Reajuste sobre el 100% del train -> predicción del test y CSV para Kaggle.
# ==============================================================================

# ---- Receta y particiones ---------------------------------------------------

# Receta de preprocesamiento compartida por todos los algoritmos.
# Argumentos:
#   datos     = base de entrenamiento
#   balanceo  = "ninguno", "smote", "submuestreo" o "pesos" (manejo del desbalance).
#               Con "pesos" la receta no cambia: los pesos de clase van en el
#               motor (ver especificar_algoritmo()).
#   variables = vector con los predictores del nivel; NULL usa todas las columnas.
#               Las demás columnas quedan con rol "no_usada" y no se exigen al hornear.
# Devuelve: objeto recipe sin estimar.
crear_receta <- function(datos, balanceo = c("ninguno", "smote", "submuestreo", "pesos"),
                         variables = NULL) {
  balanceo <- match.arg(balanceo)

  receta <- recipe(pobre ~ ., data = datos) |>
    update_role(id, new_role = "id")

  if (!is.null(variables)) {
    # La columna de pesos por fila (si la hay) conserva su rol case_weights
    pesos_caso <- names(datos)[map_lgl(datos, hardhat::is_case_weights)]
    no_usadas <- setdiff(names(datos), c("id", "pobre", variables, pesos_caso))
    if (length(no_usadas) > 0) {
      receta <- receta |>
        update_role(all_of(no_usadas), new_role = "no_usada") |>
        update_role_requirements("no_usada", bake = FALSE)
    }
  }

  receta <- receta |>
    step_impute_median(all_numeric_predictors()) |>
    step_impute_mode(all_nominal_predictors()) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())

  # El balanceo solo se aplica en entrenamiento (skip = TRUE por defecto en themis):
  # nunca toca la partición de validación de cada fold ni el holdout.
  receta <- switch(
    balanceo,
    ninguno     = receta,
    pesos       = receta,
    smote       = receta |> step_smote(pobre, seed = SEMILLA),
    submuestreo = receta |> step_downsample(pobre, seed = SEMILLA)
  )

  receta
}

# Particiones de validación cruzada estratificadas por la variable objetivo.
crear_folds <- function(datos, v = N_FOLDS) {
  set.seed(SEMILLA)
  vfold_cv(datos, v = v, strata = pobre)
}

# Separa el holdout estratificado a nivel hogar (una fila = un hogar).
# Argumentos:
#   datos      = base de train completa (con pobre)
#   test       = base de test (se guarda en la misma lista por comodidad)
#   proporcion = share que va al holdout
#   fraccion   = share del 80% que se usa para entrenar (< 1 solo en MODO_RAPIDO)
# Devuelve: lista con entrenamiento (80% o su submuestra), holdout (20%),
#   completo (lo que se usa para el reajuste final) y test.
crear_particion_holdout <- function(datos, test, proporcion = PROPORCION_HOLDOUT,
                                    fraccion = if (MODO_RAPIDO) FRACCION_RAPIDA else 1) {
  set.seed(SEMILLA)
  division <- initial_split(datos, prop = 1 - proporcion, strata = pobre)
  entrenamiento <- training(division)
  holdout <- testing(division)

  if (fraccion < 1) {
    set.seed(SEMILLA)
    entrenamiento <- entrenamiento |>
      group_by(pobre) |>
      slice_sample(prop = fraccion) |>
      ungroup()
  }

  list(
    entrenamiento = entrenamiento,
    holdout = holdout,
    completo = if (fraccion < 1) bind_rows(entrenamiento, holdout) else datos,
    test = test
  )
}

# ---- Especificación de algoritmos -------------------------------------------

# Algoritmos cuyo motor paraleliza con hilos sobre una sola copia de los datos.
# Para ellos los folds corren en secuencia, sin workers de future.
ALGORITMOS_CON_HILOS <- c("random_forest", "xgboost", "lightgbm")

# Motor de cada algoritmo de especificar_algoritmo().
MOTOR_POR_ALGORITMO <- c(
  logit = "glm", elastic_net = "glmnet", cart = "rpart", random_forest = "ranger",
  xgboost = "xgboost", naive_bayes = "naivebayes", lightgbm = "lightgbm"
)

# Cómo se le da más peso a la clase pobre en cada motor, en el modo que usa el
# pipeline (parsnip en clasificación, prediciendo probabilidades). Es la única
# fuente de esta convención; tests/test_pesos_clase.R verifica que cada
# mecanismo efectivamente mueve P(pobre).
#   - scale_pos_weight (xgboost, lightgbm): multiplica el peso de la clase
#     "positiva" del motor, que con los niveles (pobre, no_pobre) no es la misma
#     en los dos: en xgboost es pobre (valor = razón); en lightgbm es no_pobre
#     (valor = 1 / razón).
#   - case_weights (ranger, rpart, glm, glmnet): pesos por fila con
#     hardhat::importance_weights(); las filas pobres pesan razón. En ranger no
#     se usa class.weights: en bosques de probabilidad solo entra en la regla de
#     partición y deja P(pobre) igual.
#   - prior (naivebayes): parsnip no le pasa case weights. Con pesos de clase
#     n_no_pobre / n_pobre las dos clases pesan lo mismo y las verosimilitudes
#     condicionales no cambian, así que el equivalente exacto es un prior uniforme.
#   - ninguno: el motor no tiene un mecanismo verificado.
# Argumentos: motor = nombre del motor de parsnip;
#   razon_pesos = n_no_pobre / n_pobre (calcular_razon_pesos()), o NULL sin pesos.
# Devuelve: lista con mecanismo y valor (NULL si no aplica). Sin razon_pesos,
#   mecanismo = "sin_pesos".
peso_pobre_motor <- function(motor, razon_pesos = NULL) {
  mecanismo <- switch(
    motor,
    xgboost = , lightgbm = "scale_pos_weight",
    ranger = , rpart = , glm = , glmnet = "case_weights",
    naivebayes = "prior",
    "ninguno"
  )
  if (is.null(razon_pesos)) return(list(mecanismo = "sin_pesos", valor = NULL))

  valor <- switch(
    mecanismo,
    scale_pos_weight = if (motor == "xgboost") razon_pesos else 1 / razon_pesos,
    case_weights = razon_pesos,
    prior = c(0.5, 0.5),  # orden de los niveles: (pobre, no_pobre)
    NULL
  )
  list(mecanismo = mecanismo, valor = valor)
}

# Agrega la columna de pesos por fila (importance_weights): razón para los
# pobres y 1 para los demás. La receta le asigna el rol case_weights sola.
# Argumentos: datos = base con pobre; razon_pesos; columna = nombre de la columna.
# Devuelve: la base con la columna agregada.
agregar_pesos_caso <- function(datos, razon_pesos, columna = "peso_caso") {
  datos[[columna]] <- hardhat::importance_weights(if_else(datos$pobre == "pobre", razon_pesos, 1))
  datos
}

# Workflow con la receta y el modelo, y los pesos por fila si los hay.
# Argumentos: receta; modelo = especificación parsnip;
#   columna_pesos = nombre de la columna de pesos, o NULL.
# Devuelve: workflow sin ajustar.
crear_flujo <- function(receta, modelo, columna_pesos = NULL) {
  flujo <- workflow() |> add_recipe(receta) |> add_model(modelo)
  if (!is.null(columna_pesos)) flujo <- flujo |> add_case_weights(all_of(columna_pesos))
  flujo
}

# Tamaño de grilla según el modo: en MODO_RAPIDO todo se reduce para validar.
tamano_grilla <- function(completo, rapido) {
  if (MODO_RAPIDO) rapido else completo
}

# Especificación parsnip + grilla de cada algoritmo. Centraliza los rangos de
# búsqueda para que estén documentados en un solo lugar.
# Argumentos:
#   nombre      = "logit", "elastic_net", "cart", "random_forest", "xgboost",
#                 "naive_bayes" o "lightgbm"
#   razon_pesos = si no es NULL, peso relativo de la clase pobre (n_no_pobre / n_pobre)
# Devuelve: lista con algoritmo (nombre del curso), modelo, grilla, descripcion
#   de la búsqueda y parallel_over.
especificar_algoritmo <- function(nombre, razon_pesos = NULL) {
  set.seed(SEMILLA)

  # parsnip guarda los argumentos como expresiones que se evalúan dentro de los
  # workers en paralelo, donde no existen las funciones ni las constantes del
  # proyecto: se calculan aquí y se inyectan con !!
  semilla <- SEMILLA
  n_arboles_rf <- tamano_grilla(500, 100)
  # Los motores con hilos corren los folds en secuencia y usan todos los núcleos
  hilos <- N_NUCLEOS

  motor <- unname(MOTOR_POR_ALGORITMO[nombre])
  if (is.na(motor)) stop("Algoritmo desconocido: ", nombre)
  pesos <- peso_pobre_motor(motor, razon_pesos)
  if (pesos$mecanismo == "ninguno") {
    warning(
      "El motor ", motor, " no tiene un mecanismo de pesos de clase que mueva P(pobre): ",
      "el candidato queda con pesos_aplicados = FALSE y no compite en el nivel 5",
      call. = FALSE
    )
  }
  # Los mecanismos que viven en el motor se inyectan aquí; case_weights va en
  # los datos y en el workflow (ver correr_candidato_escalera())
  peso_motor <- if (pesos$mecanismo == "scale_pos_weight") pesos$valor else 1
  prior_nb <- if (pesos$mecanismo == "prior") pesos$valor else NULL

  especificacion <- switch(
    nombre,

    logit = list(
      algoritmo = "Logit",
      modelo = logistic_reg() |> set_engine("glm"),
      grilla = NULL,
      busqueda = "sin hiperparámetros"
    ),

    # glmnet estima todo el camino de penalty de una vez: la grilla regular sale barata
    elastic_net = list(
      algoritmo = "Elastic Net",
      modelo = logistic_reg(penalty = tune(), mixture = tune()) |> set_engine("glmnet"),
      grilla = grid_regular(
        penalty(range = c(-5, 0)),
        mixture(range = c(0, 1)),
        levels = c(penalty = tamano_grilla(20, 4), mixture = tamano_grilla(6, 2))
      ),
      busqueda = "grilla regular: penalty 10^[-5, 0] x mixture [0, 1]"
    ),

    cart = list(
      algoritmo = "CART",
      modelo = decision_tree(cost_complexity = tune(), tree_depth = tune()) |>
        set_engine("rpart") |>
        set_mode("classification"),
      grilla = grid_regular(
        cost_complexity(range = c(-5, -1)),
        tree_depth(range = c(3, 15)),
        levels = tamano_grilla(6, 2)
      ),
      busqueda = "grilla regular: cost_complexity 10^[-5, -1] x tree_depth [3, 15]"
    ),

    random_forest = list(
      algoritmo = "Random Forest",
      modelo = rand_forest(mtry = tune(), min_n = tune(), trees = !!n_arboles_rf) |>
        set_engine(
          "ranger", importance = "impurity", seed = !!semilla, num.threads = !!hilos
        ) |>
        set_mode("classification"),
      grilla = grid_space_filling(
        mtry(range = c(3, 25)),
        min_n(range = c(5, 60)),
        size = tamano_grilla(15, 2),
        type = "latin_hypercube"
      ),
      busqueda = "latin hypercube: mtry [3, 25] x min_n [5, 60], 500 árboles"
    ),

    xgboost = list(
      algoritmo = "Boosting",
      modelo = boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune()) |>
        set_engine("xgboost", nthread = !!hilos, scale_pos_weight = !!peso_motor) |>
        set_mode("classification"),
      grilla = grid_space_filling(
        trees(range = c(200, tamano_grilla(1000, 300))),
        tree_depth(range = c(3, 8)),
        learn_rate(range = c(-3, -1)),
        size = tamano_grilla(20, 3),
        type = "latin_hypercube"
      ),
      busqueda = "latin hypercube: trees [200, 1000] x tree_depth [3, 8] x learn_rate 10^[-3, -1]"
    ),

    naive_bayes = list(
      algoritmo = "Naive Bayes",
      modelo = naive_Bayes(smoothness = tune(), Laplace = tune()) |>
        set_engine("naivebayes", prior = !!prior_nb),
      grilla = grid_regular(
        smoothness(range = c(0.5, 1.5)),
        Laplace(range = c(0, 3)),
        levels = tamano_grilla(4, 2)
      ),
      busqueda = "grilla regular: smoothness [0.5, 1.5] x Laplace [0, 3]"
    ),

    lightgbm = list(
      algoritmo = "Boosting",
      modelo = boost_tree(
        trees = tune(), tree_depth = tune(), learn_rate = tune(), min_n = tune(),
        mtry = tune(), loss_reduction = tune(), sample_size = tune()
      ) |>
        set_engine(
          "lightgbm", num_threads = !!hilos, verbose = -1, seed = !!semilla,
          counts = FALSE,  # mtry como proporción de columnas
          scale_pos_weight = !!peso_motor
        ) |>
        set_mode("classification"),
      grilla = grid_space_filling(
        trees(range = c(200, tamano_grilla(2000, 400))),
        tree_depth(range = c(3, 12)),
        learn_rate(range = c(-3, -0.7)),
        min_n(range = c(10, 200)),
        mtry_prop(range = c(0.2, 1)),
        loss_reduction(range = c(-10, 1)),
        sample_prop(range = c(0.5, 1)),
        size = tamano_grilla(50, 3),
        type = "latin_hypercube"
      ) |>
        # mtry_prop se llama así en la grilla; sample_prop ya sale como sample_size
        rename(any_of(c(mtry = "mtry_prop", sample_size = "sample_prop"))),
      busqueda = paste(
        "latin hypercube 50: trees [200, 2000], tree_depth [3, 12], learn_rate 10^[-3, -0.7],",
        "min_n [10, 200], mtry [0.2, 1], loss_reduction 10^[-10, 1], sample_size [0.5, 1]"
      )
    ),

    stop("Algoritmo desconocido: ", nombre)
  )

  especificacion$nombre <- nombre
  especificacion$mecanismo_pesos <- pesos$mecanismo
  especificacion$peso_caso <- if (pesos$mecanismo == "case_weights") pesos$valor
  # NA: no se pidieron pesos; FALSE: se pidieron pero el motor no los aplica
  especificacion$pesos_aplicados <- if (is.null(razon_pesos)) NA else pesos$mecanismo != "ninguno"
  especificacion
}

# Fija el número de hilos del motor en los ajustes finales (holdout y 100%).
# Los motores con hilos ya traen N_NUCLEOS desde especificar_algoritmo(); se
# conserva por si una especificación llega de otra parte con 1 hilo.
# Argumentos: modelo = especificación parsnip; hilos = número de hilos.
# Devuelve: la especificación con los hilos actualizados.
ajustar_hilos <- function(modelo, hilos = N_NUCLEOS) {
  argumento <- switch(
    modelo$engine,
    ranger   = "num.threads",
    xgboost  = "nthread",
    lightgbm = "num_threads",
    NULL
  )
  if (is.null(argumento)) return(modelo)

  argumentos <- modelo$eng_args
  argumentos[[argumento]] <- rlang::quo(!!hilos)
  set_engine(modelo, modelo$engine, !!!argumentos)
}

# ---- Predicciones fuera de fold ---------------------------------------------

# Tuning (o solo remuestreo si no hay grilla) guardando las predicciones OOF.
# Argumentos: flujo = workflow; folds; grilla (NULL si no hay hiperparámetros);
#   parallel_over = estrategia de paralelización de tune.
# Devuelve: lista con predicciones (.config, fold, .row, verdad, probabilidad)
#   y configuraciones (una fila por combinación de hiperparámetros).
obtener_oof_tidymodels <- function(flujo, folds, grilla, parallel_over = "everything") {
  metricas <- metric_set(roc_auc)

  set.seed(SEMILLA)
  resultados <- if (is.null(grilla)) {
    fit_resamples(flujo, folds, metrics = metricas,
                  control = control_resamples(save_pred = TRUE, parallel_over = parallel_over))
  } else {
    tune_grid(flujo, folds, grid = grilla, metrics = metricas,
              control = control_grid(save_pred = TRUE, parallel_over = parallel_over))
  }

  nombres_parametros <- if (is.null(grilla)) character() else names(grilla)

  predicciones <- collect_predictions(resultados, summarize = FALSE) |>
    transmute(
      .config, fold = id, .row,
      verdad = as.integer(pobre == "pobre"),
      probabilidad = .pred_pobre
    )

  configuraciones <- collect_metrics(resultados) |>
    distinct(.config, across(any_of(nombres_parametros)))

  list(predicciones = predicciones, configuraciones = configuraciones)
}

# Versión manual de las predicciones OOF para modelos que no pasan por parsnip
# (LPM y reglas naive). Corre secuencialmente: son modelos baratos.
# Argumentos:
#   ajustar_predecir = función(entrenamiento, evaluacion, config) que devuelve
#                      lista con probabilidad (y opcionalmente modelo)
#   folds            = salida de crear_folds()
#   configuraciones  = tibble con una fila por configuración (NULL = una sola)
# Devuelve: misma estructura que obtener_oof_tidymodels().
obtener_oof_manual <- function(ajustar_predecir, folds, configuraciones = NULL) {
  if (is.null(configuraciones)) configuraciones <- tibble(.row_config = 1)
  configuraciones <- configuraciones |>
    mutate(.config = paste0("config_", row_number()))

  predicciones <- map_dfr(seq_along(folds$splits), function(k) {
    particion <- folds$splits[[k]]
    analisis <- analysis(particion)
    evaluacion <- assessment(particion)

    map_dfr(seq_len(nrow(configuraciones)), function(j) {
      tibble(
        .config = configuraciones$.config[j],
        fold = folds$id[k],
        .row = complement(particion),
        verdad = as.integer(evaluacion$pobre == "pobre"),
        probabilidad = ajustar_predecir(analisis, evaluacion, configuraciones[j, ])$probabilidad
      )
    })
  })

  list(
    predicciones = predicciones,
    configuraciones = configuraciones |> select(-any_of(".row_config"))
  )
}

# Elige la configuración de hiperparámetros con el mayor F1 OOF, evaluando cada
# configuración en su propio umbral óptimo (no en el 0,5 de select_best()).
# Argumentos: predicciones = salida $predicciones de obtener_oof_*().
# Devuelve: lista con la tabla por configuración y el .config ganador.
seleccionar_por_f1_oof <- function(predicciones) {
  tabla <- predicciones |>
    group_by(.config) |>
    group_modify(function(datos, llave) {
      busqueda <- buscar_umbral_f1(datos$verdad, datos$probabilidad)
      tibble(umbral = busqueda$umbral, f1_oof = max(busqueda$curva$f1))
    }) |>
    ungroup() |>
    arrange(desc(f1_oof))

  list(tabla = tabla, mejor = tabla$.config[1])
}

# Métricas de CV con un umbral fijo: media y desviación entre folds.
# Argumentos: predicciones = OOF de una sola configuración; umbral = umbral fijo.
# Devuelve: tibble de una fila con f1_cv, f1_cv_sd, precision_cv, recall_cv, accuracy_cv.
metricas_por_fold <- function(predicciones, umbral) {
  predicciones |>
    group_by(fold) |>
    group_modify(~ calcular_metricas(.x$verdad, as.integer(.x$probabilidad >= umbral))) |>
    ungroup() |>
    summarise(
      f1_cv        = mean(f1),
      f1_cv_sd     = sd(f1),
      precision_cv = mean(precision),
      recall_cv    = mean(recall),
      accuracy_cv  = mean(accuracy)
    )
}

# ---- Importancia de variables -----------------------------------------------

# Importancia de variables directamente desde el motor de cada algoritmo
# (reemplaza a vip, que salió de CRAN).
# Argumentos: modelo = workflow ajustado o lm (LPM).
# Devuelve: tibble (variable, importancia) normalizada a máximo 1, o NULL si el
#   algoritmo no tiene una medida natural (Naive Bayes).
calcular_importancia <- function(modelo) {
  motor <- if (inherits(modelo, "workflow")) extract_fit_engine(modelo) else modelo

  importancia <- tryCatch(
    {
      if (inherits(motor, "ranger")) {
        imp <- ranger::importance(motor)
        tibble(variable = names(imp), importancia = unname(imp))
      } else if (inherits(motor, "xgb.Booster")) {
        xgboost::xgb.importance(model = motor) |>
          as_tibble() |>
          transmute(variable = Feature, importancia = Gain)
      } else if (inherits(motor, "lgb.Booster")) {
        lightgbm::lgb.importance(motor) |>
          as_tibble() |>
          transmute(variable = Feature, importancia = Gain)
      } else if (inherits(motor, "rpart")) {
        imp <- motor$variable.importance
        tibble(variable = names(imp), importancia = unname(imp))
      } else if (inherits(motor, "glmnet")) {
        # Predictores normalizados en la receta: |coeficiente| es comparable
        tidy(extract_fit_parsnip(modelo)) |>
          filter(term != "(Intercept)") |>
          transmute(variable = term, importancia = abs(estimate))
      } else if (inherits(motor, c("glm", "lm"))) {
        # |estadístico t o z|: efecto sobre su error estándar
        broom::tidy(motor) |>
          filter(term != "(Intercept)", !is.na(statistic)) |>
          transmute(variable = term, importancia = abs(statistic))
      } else {
        NULL
      }
    },
    error = function(e) NULL
  )

  if (is.null(importancia) || nrow(importancia) == 0) return(NULL)

  importancia |>
    mutate(importancia = importancia / max(importancia)) |>
    arrange(desc(importancia))
}

# Convierte una fila de hiperparámetros en texto legible para el leaderboard.
describir_hiperparametros <- function(hiperparametros) {
  if (is.null(hiperparametros) || ncol(hiperparametros) == 0) return("ninguno")
  valores <- map_chr(hiperparametros, function(x) {
    if (is.numeric(x)) format(signif(x, 3)) else as.character(x)
  })
  paste(names(valores), valores, sep = " = ", collapse = "; ")
}

# ---- Núcleo de la escalera --------------------------------------------------

# Evalúa un candidato de la escalera de punta a punta (ver encabezado del archivo).
# Si stores/modelos/<nombre>.rds existe, lo carga en vez de recalcular.
# Argumentos:
#   nombre, nivel, algoritmo, especificacion = identificación del candidato
#   particion        = salida de crear_particion_holdout()
#   folds            = folds sobre particion$entrenamiento (los mismos para todos)
#   receta           = salida de crear_receta()
#   modelo, grilla   = especificación parsnip y grilla (tipo "tidymodels")
#   ajustar_predecir = función(entrenamiento, evaluacion, config) para tipo "manual"
#   configuraciones  = tibble de configuraciones para tipo "manual"
#   parallel_over    = estrategia de paralelización de tune
#   balanceo, variables, busqueda = metadatos para el leaderboard
#   columna_pesos    = columna de pesos por fila (case_weights) o NULL
#   mecanismo_pesos, pesos_aplicados = salida de especificar_algoritmo(); quedan
#                      guardados para que un balanceo con pesos que el motor no
#                      aplica no se reporte como si hubiera competido
# Devuelve: lista con hiperparámetros, umbral, métricas CV y holdout, predicciones
#   OOF/holdout/test e importancia de variables.
evaluar_candidato <- function(nombre, nivel, algoritmo, especificacion, particion, folds,
                              receta = NULL, modelo = NULL, grilla = NULL,
                              ajustar_predecir = NULL, configuraciones = NULL,
                              parallel_over = "everything", balanceo = "ninguno",
                              variables = character(), busqueda = "sin hiperparámetros",
                              columna_pesos = NULL, mecanismo_pesos = "sin_pesos",
                              pesos_aplicados = NA) {
  ruta_resultado <- file.path(RUTA_MODELOS, paste0(nombre, ".rds"))
  if (file.exists(ruta_resultado)) {
    message("[", nombre, "] cargado desde ", basename(ruta_resultado))
    registrar_log(nombre, "cargado de caché")
    return(readRDS(ruta_resultado))
  }

  message("\n[", nombre, "] ", especificacion)
  inicio <- Sys.time()
  registrar_log(nombre, "inicio", reiniciar_pico = TRUE)
  es_tidymodels <- !is.null(modelo)

  # 1. Predicciones fuera de fold sobre el 80%
  oof <- if (es_tidymodels) {
    flujo <- crear_flujo(receta, modelo, columna_pesos)
    obtener_oof_tidymodels(flujo, folds, grilla, parallel_over)
  } else {
    obtener_oof_manual(ajustar_predecir, folds, configuraciones)
  }
  registrar_log(nombre, "fin de la CV")

  # 2. Hiperparámetros y umbral elegidos solo con predicciones OOF
  seleccion <- seleccionar_por_f1_oof(oof$predicciones)
  predicciones_mejor <- oof$predicciones |> filter(.config == seleccion$mejor)
  busqueda_umbral <- buscar_umbral_f1(predicciones_mejor$verdad, predicciones_mejor$probabilidad)
  umbral <- busqueda_umbral$umbral
  metricas_cv <- metricas_por_fold(predicciones_mejor, umbral)
  hiperparametros <- oof$configuraciones |>
    filter(.config == seleccion$mejor) |>
    select(-.config)

  ajustador <- if (es_tidymodels) {
    function(entrenamiento, evaluacion) {
      flujo_final <- finalize_workflow(flujo, hiperparametros)
      flujo_final <- flujo_final |>
        update_model(ajustar_hilos(extract_spec_parsnip(flujo_final)))
      ajuste <- fit(flujo_final, entrenamiento)
      list(
        probabilidad = predict(ajuste, evaluacion, type = "prob")$.pred_pobre,
        modelo = ajuste
      )
    }
  } else {
    function(entrenamiento, evaluacion) {
      ajustar_predecir(entrenamiento, evaluacion, hiperparametros)
    }
  }

  # 3. Holdout: reajuste sobre el 80% y umbral OOF fijo
  probabilidad_holdout <- ajustador(particion$entrenamiento, particion$holdout)$probabilidad
  verdad_holdout <- as.integer(particion$holdout$pobre == "pobre")
  clases_holdout <- as.integer(probabilidad_holdout >= umbral)
  metricas_holdout <- calcular_metricas(verdad_holdout, clases_holdout)
  registrar_log(nombre, "fin del holdout")

  # 4. Reajuste sobre el 100% del train y predicción del test
  ajuste_final <- ajustador(particion$completo, particion$test)
  clases_test <- as.integer(ajuste_final$probabilidad >= umbral)
  registrar_log(nombre, "fin del ajuste final")

  resultado <- list(
    nombre = nombre,
    nivel = nivel,
    algoritmo = algoritmo,
    especificacion = especificacion,
    balanceo = balanceo,
    mecanismo_pesos = mecanismo_pesos,
    pesos_aplicados = pesos_aplicados,
    variables = variables,
    busqueda = busqueda,
    n_configuraciones = nrow(seleccion$tabla),
    hiperparametros = hiperparametros,
    umbral = umbral,
    curva_umbral = busqueda_umbral$curva,
    seleccion = seleccion$tabla,
    metricas_cv = metricas_cv,
    metricas_holdout = metricas_holdout,
    oof = predicciones_mejor |> select(.row, fold, verdad, probabilidad),
    prediccion_holdout = tibble(
      id = particion$holdout$id, probabilidad = probabilidad_holdout, clase = clases_holdout
    ),
    prediccion_test = tibble(
      id = particion$test$id, probabilidad = ajuste_final$probabilidad, clase = clases_test
    ),
    importancia = calcular_importancia(ajuste_final$modelo),
    archivo_submission = paste0(nombre, ".csv"),
    tiempo_min = as.numeric(difftime(Sys.time(), inicio, units = "mins"))
  )

  guardar_submission(
    ids = particion$test$id,
    clases = clases_test,
    algoritmo = algoritmo,
    especificacion = especificacion,
    f1_cv = metricas_cv$f1_cv,
    carpeta = file.path(RUTA_SUBMISSIONS, "candidatos"),
    nombre_archivo = resultado$archivo_submission,
    nivel = nivel,
    f1_holdout = metricas_holdout$f1,
    registrar = FALSE
  )

  # El modelo ajustado sobre el 100% se guarda aparte y adelgazado. El resultado
  # solo lleva predicciones (OOF, y las de holdout y test que usa el stacking),
  # métricas, hiperparámetros y umbral: nunca el objeto de tuning.
  modelo_final <- adelgazar_modelo(ajuste_final$modelo, particion$test, ajuste_final$probabilidad)
  saveRDS(modelo_final, file.path(RUTA_MODELOS, paste0(nombre, "_ajuste.rds")))
  saveRDS(resultado, ruta_resultado)

  message(sprintf(
    "[%s] F1 CV = %.4f (sd %.4f) | F1 holdout = %.4f | umbral = %.2f | %.1f min",
    nombre, metricas_cv$f1_cv, metricas_cv$f1_cv_sd, metricas_holdout$f1, umbral,
    resultado$tiempo_min
  ))
  registrar_log(nombre, "fin", inicio = inicio)

  # Libera lo pesado antes del siguiente candidato
  rm(oof, ajuste_final, modelo_final, ajustador)
  gc(verbose = FALSE)
  resultado
}

# Adelgaza un modelo ajustado con butcher (quita datos de entrenamiento, llamadas
# y entornos que no hacen falta para predecir). Si es un workflow, verifica que
# siga prediciendo igual sobre unas filas del test; si butcher falla o cambia
# las predicciones, devuelve el modelo original.
# Argumentos: modelo = workflow ajustado, lm (LPM) o NULL (reglas);
#   test = base de test; probabilidad = P(pobre) del modelo original en test;
#   n_filas = filas del test con que se verifica.
# Devuelve: el modelo adelgazado (o el original).
adelgazar_modelo <- function(modelo, test, probabilidad, n_filas = 200) {
  if (!inherits(modelo, c("workflow", "lm"))) return(modelo)

  adelgazado <- tryCatch(butcher::butcher(modelo), error = function(e) NULL)
  if (is.null(adelgazado)) return(modelo)
  # El lm del LPM predice sobre la receta horneada; butcher conserva los
  # coeficientes, que es lo único que usa predict.lm
  if (inherits(adelgazado, "lm")) return(adelgazado)

  filas <- seq_len(min(n_filas, nrow(test)))
  prediccion <- tryCatch(
    predict(adelgazado, test[filas, ], type = "prob")$.pred_pobre,
    error = function(e) NA_real_
  )
  if (anyNA(prediccion) || max(abs(prediccion - probabilidad[filas])) > 1e-8) {
    message("  butcher cambió las predicciones: se guarda el modelo sin adelgazar")
    return(modelo)
  }
  adelgazado
}

# ---- Ajustadores manuales ----------------------------------------------------

# Construye el ajustador del modelo lineal de probabilidad. Va aparte porque
# parsnip no lo trata como clasificador: se estima con lm() sobre la receta horneada.
# Argumentos: receta = salida de crear_receta().
# Devuelve: función(entrenamiento, evaluacion, config) -> lista(probabilidad, modelo).
crear_ajustador_lpm <- function(receta) {
  function(entrenamiento, evaluacion, config = NULL) {
    receta_estimada <- prep(receta, training = entrenamiento)
    base_train <- bake(receta_estimada, new_data = NULL, all_predictors(), all_outcomes()) |>
      mutate(pobre = as.integer(pobre == "pobre"))
    base_eval <- bake(receta_estimada, new_data = evaluacion, all_predictors())

    modelo <- lm(pobre ~ ., data = base_train)
    # Con dominio y departamento hay dummies colineales: lm las descarta (NA) y avisa
    probabilidad <- suppressWarnings(predict(modelo, newdata = base_eval))
    list(probabilidad = pmin(pmax(probabilidad, 0), 1), modelo = modelo)
  }
}

# Ajustador de las reglas naive del nivel 1.
# config$variable puede ser "todos_pobres", "todos_no_pobres" o el nombre de una
# variable: en ese caso la "probabilidad" es el percentil de la variable en el
# entrenamiento (en el sentido en que se asocia con pobreza), así el umbral
# óptimo equivale a cortar la variable en un percentil.
# Devuelve: lista(probabilidad, modelo).
ajustar_predecir_regla <- function(entrenamiento, evaluacion, config) {
  variable <- config$variable
  n <- nrow(evaluacion)
  if (variable == "todos_pobres")    return(list(probabilidad = rep(1, n), modelo = NULL))
  if (variable == "todos_no_pobres") return(list(probabilidad = rep(0, n), modelo = NULL))

  mediana <- median(entrenamiento[[variable]], na.rm = TRUE)
  x <- coalesce(entrenamiento[[variable]], mediana)
  sentido <- sign(cor(x, as.integer(entrenamiento$pobre == "pobre")))
  acumulada <- ecdf(sentido * x)

  x_evaluacion <- coalesce(evaluacion[[variable]], mediana)
  list(
    probabilidad = acumulada(sentido * x_evaluacion),
    modelo = list(variable = variable, sentido = sentido, mediana = mediana)
  )
}

# ---- Flujos por algoritmo (compatibles con las llamadas originales) ---------

# Conjunto de métricas que se calcula durante el tuning (flujo sin holdout).
metricas_tuning <- metric_set(f_meas, precision, recall, roc_auc)

# Corre el flujo completo de un algoritmo: tuning por CV, umbral óptimo de F1,
# ajuste final, predicción en test y submission a Kaggle.
# Argumentos:
#   algoritmo      = nombre del algoritmo (ver guardar_submission())
#   modelo         = especificación parsnip (con tune() donde aplique)
#   receta         = salida de crear_receta()
#   folds          = salida de crear_folds()
#   grilla         = grilla de hiperparámetros (NULL si no hay nada que tunear)
#   datos_train    = base de entrenamiento
#   datos_test     = base de test
#   especificacion = descripción corta para la bitácora
#   Argumentos nuevos (opcionales; sin ellos la función se comporta como antes):
#   datos_holdout  = si no es NULL, usa el esquema con holdout de evaluar_candidato():
#                    datos_train es el 80% y el reajuste final usa train + holdout
#   nivel, nombre, balanceo, variables, busqueda, parallel_over, columna_pesos,
#   mecanismo_pesos, pesos_aplicados = ver evaluar_candidato()
# Devuelve: lista con mejores hiperparámetros, umbral, métricas CV y modelo final
#   (o la salida de evaluar_candidato() si hay holdout).
correr_algoritmo <- function(algoritmo, modelo, receta, folds, grilla,
                             datos_train, datos_test, especificacion,
                             datos_holdout = NULL, nivel = NA_character_, nombre = NULL,
                             balanceo = "ninguno", variables = character(),
                             busqueda = "sin hiperparámetros",
                             parallel_over = "everything",
                             columna_pesos = NULL, mecanismo_pesos = "sin_pesos",
                             pesos_aplicados = NA) {
  if (!is.null(datos_holdout)) {
    particion <- list(
      entrenamiento = datos_train,
      holdout = datos_holdout,
      completo = bind_rows(datos_train, datos_holdout),
      test = datos_test
    )
    return(evaluar_candidato(
      nombre = nombre %||% str_replace_all(str_to_lower(algoritmo), " ", "_"),
      nivel = nivel, algoritmo = algoritmo, especificacion = especificacion,
      particion = particion, folds = folds, receta = receta, modelo = modelo,
      grilla = grilla, parallel_over = parallel_over, balanceo = balanceo,
      variables = variables, busqueda = busqueda, columna_pesos = columna_pesos,
      mecanismo_pesos = mecanismo_pesos, pesos_aplicados = pesos_aplicados
    ))
  }

  flujo <- workflow() |> add_recipe(receta) |> add_model(modelo)
  control <- control_grid(save_pred = TRUE, parallel_over = "everything")

  set.seed(SEMILLA)
  resultados_cv <- if (is.null(grilla)) {
    fit_resamples(flujo, folds, metrics = metricas_tuning,
                  control = control_resamples(save_pred = TRUE))
  } else {
    tune_grid(flujo, folds, grid = grilla, metrics = metricas_tuning, control = control)
  }

  mejores <- if (is.null(grilla)) tibble() else select_best(resultados_cv, metric = "f_meas")

  # Predicciones fuera de muestra del mejor modelo -> umbral que maximiza F1
  predicciones_oof <- collect_predictions(resultados_cv, parameters = if (nrow(mejores)) mejores)
  busqueda_umbral <- buscar_umbral_f1(
    verdad = as.integer(predicciones_oof$pobre == "pobre"),
    probabilidad = predicciones_oof$.pred_pobre
  )
  metricas_cv <- busqueda_umbral$curva |> filter(umbral == busqueda_umbral$umbral)

  # Ajuste final con toda la base de entrenamiento
  flujo_final <- if (nrow(mejores)) finalize_workflow(flujo, mejores) else flujo
  modelo_final <- fit(flujo_final, datos_train)

  probabilidad_test <- predict(modelo_final, datos_test, type = "prob")$.pred_pobre
  guardar_submission(
    ids = datos_test$id,
    clases = as.integer(probabilidad_test >= busqueda_umbral$umbral),
    algoritmo = algoritmo,
    especificacion = especificacion,
    f1_cv = metricas_cv$f1
  )

  resultado <- list(
    algoritmo = algoritmo,
    especificacion = especificacion,
    mejores_hiperparametros = mejores,
    umbral = busqueda_umbral$umbral,
    metricas_cv = metricas_cv,
    curva_umbral = busqueda_umbral$curva,
    modelo_final = modelo_final
  )

  guardar_resultado(resultado)
  resultado
}

# Modelo lineal de probabilidad. Mismos argumentos que correr_algoritmo(), sin
# modelo ni grilla. Con datos_holdout usa el esquema de evaluar_candidato().
correr_lpm <- function(receta, folds, datos_train, datos_test, especificacion,
                       datos_holdout = NULL, nivel = NA_character_, nombre = "lpm",
                       variables = character()) {
  ajustar_y_predecir <- crear_ajustador_lpm(receta)

  if (!is.null(datos_holdout)) {
    particion <- list(
      entrenamiento = datos_train,
      holdout = datos_holdout,
      completo = bind_rows(datos_train, datos_holdout),
      test = datos_test
    )
    return(evaluar_candidato(
      nombre = nombre, nivel = nivel, algoritmo = "LPM", especificacion = especificacion,
      particion = particion, folds = folds, ajustar_predecir = ajustar_y_predecir,
      variables = variables
    ))
  }

  predicciones_oof <- map_dfr(folds$splits, function(particion) {
    evaluacion <- assessment(particion)
    tibble(
      verdad = as.integer(evaluacion$pobre == "pobre"),
      probabilidad = ajustar_y_predecir(analysis(particion), evaluacion)$probabilidad
    )
  })

  busqueda_umbral <- buscar_umbral_f1(predicciones_oof$verdad, predicciones_oof$probabilidad)
  metricas_cv <- busqueda_umbral$curva |> filter(umbral == busqueda_umbral$umbral)

  final <- ajustar_y_predecir(datos_train, datos_test)
  guardar_submission(
    ids = datos_test$id,
    clases = as.integer(final$probabilidad >= busqueda_umbral$umbral),
    algoritmo = "LPM",
    especificacion = especificacion,
    f1_cv = metricas_cv$f1
  )

  resultado <- list(
    algoritmo = "LPM",
    especificacion = especificacion,
    mejores_hiperparametros = tibble(),
    umbral = busqueda_umbral$umbral,
    metricas_cv = metricas_cv,
    curva_umbral = busqueda_umbral$curva,
    modelo_final = final$modelo
  )

  guardar_resultado(resultado)
  resultado
}

# Guarda el objeto completo (.rds, no se sube) y una fila de métricas (.csv, sí se sube).
guardar_resultado <- function(resultado) {
  marca <- format(Sys.time(), "%Y%m%d_%H%M%S")
  base_nombre <- paste0(str_replace_all(str_to_lower(resultado$algoritmo), " ", "_"), "_", marca)

  saveRDS(resultado, file.path(RUTA_RESULTADOS, paste0(base_nombre, ".rds")))

  ruta_metricas <- file.path(RUTA_RESULTADOS, "metricas_cv.csv")
  fila <- resultado$metricas_cv |>
    mutate(
      algoritmo = resultado$algoritmo,
      especificacion = resultado$especificacion,
      archivo_rds = paste0(base_nombre, ".rds"),
      .before = 1
    )
  write_csv(fila, ruta_metricas, append = file.exists(ruta_metricas))
}

# Carga los datos procesados que produce 01_preparar_datos.R.
cargar_bases_modelado <- function() {
  list(
    train = readRDS(file.path(RUTA_PROCESADOS, "train_hogar.rds")),
    test  = readRDS(file.path(RUTA_PROCESADOS, "test_hogar.rds"))
  )
}
