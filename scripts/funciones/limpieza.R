# ==============================================================================
# funciones/limpieza.R
# Carga, limpieza y unión de las bases de hogares y personas (DANE - MESE).
#
# OJO: los nombres originales (P6020, P6040, ...) siguen el diccionario de la
# GEIH. Verifíquenlos contra la página de la competencia antes de correr; si
# alguna variable no existe en el test, sáquenla de la lista.
# ==============================================================================

# Carga los cuatro CSV de Kaggle.
# Argumentos: ruta = carpeta donde están los archivos.
# Devuelve: lista con train_hogares, train_personas, test_hogares, test_personas.
cargar_datos_crudos <- function(ruta = RUTA_CRUDOS) {
  archivos <- c(
    train_hogares  = "train_hogares.csv",
    train_personas = "train_personas.csv",
    test_hogares   = "test_hogares.csv",
    test_personas  = "test_personas.csv"
  )

  faltantes <- archivos[!file.exists(file.path(ruta, archivos))]
  if (length(faltantes) > 0) {
    stop(
      "Faltan archivos en ", ruta, ": ", paste(faltantes, collapse = ", "),
      "\nDescárguenlos de Kaggle y pónganlos en stores/raw/."
    )
  }

  map(archivos, ~ read_csv(file.path(ruta, .x), show_col_types = FALSE))
}

# Reporta qué variables están en el train pero no en el test.
# Argumentos: train, test = data frames del mismo nivel (hogar o persona).
# Devuelve: tibble con las variables ausentes en el test.
variables_solo_en_train <- function(train, test) {
  tibble(variable = setdiff(names(train), names(test)))
}

# Construye variables a nivel persona a partir de la base individual.
# Solo usa variables que también están en el test.
# Argumentos: personas = base de personas cruda.
# Devuelve: tibble de personas con variables limpias.
limpiar_personas <- function(personas) {
  personas |>
    transmute(
      id,
      mujer         = as.integer(P6020 == 2),
      edad          = P6040,
      jefe_hogar    = as.integer(P6050 == 1),
      menor_edad    = as.integer(P6040 < 18),
      adulto_mayor  = as.integer(P6040 >= 65),
      # P6210: nivel educativo (9 = no sabe); faltante -> 0
      nivel_educ    = if_else(is.na(P6210) | P6210 == 9, 0, P6210),
      ocupado       = as.integer(!is.na(Oc) & Oc == 1),
      afiliado_salud = as.integer(!is.na(P6090) & P6090 == 1),
      cotiza_pension = as.integer(!is.na(P6920) & P6920 %in% c(1, 3))
    )
}

# Agrega la información de personas al nivel hogar.
# Argumentos: personas = salida de limpiar_personas().
# Devuelve: tibble con una fila por id de hogar.
agregar_personas_a_hogar <- function(personas) {
  personas |>
    group_by(id) |>
    summarise(
      n_miembros         = n(),
      n_menores          = sum(menor_edad),
      n_adultos_mayores  = sum(adulto_mayor),
      n_ocupados         = sum(ocupado),
      prop_ocupados      = n_ocupados / n_miembros,
      prop_mujeres       = mean(mujer),
      educ_maxima        = max(nivel_educ),
      educ_promedio      = mean(nivel_educ),
      prop_afiliados_salud = mean(afiliado_salud),
      prop_cotiza_pension  = mean(cotiza_pension),
      jefe_mujer         = max(mujer * jefe_hogar),
      jefe_edad          = max(edad * jefe_hogar),
      jefe_educ          = max(nivel_educ * jefe_hogar),
      jefe_ocupado       = max(ocupado * jefe_hogar),
      .groups = "drop"
    ) |>
    mutate(
      n_edad_trabajar  = n_miembros - n_menores - n_adultos_mayores,
      tasa_dependencia = (n_menores + n_adultos_mayores) / pmax(n_edad_trabajar, 1)
    )
}

# Limpia la base de hogares y renombra variables al español.
# any_of() permite que la misma función sirva para train (con pobre) y test.
# Argumentos: hogares = base de hogares cruda.
# Devuelve: tibble de hogares limpio.
limpiar_hogares <- function(hogares) {
  nombres <- c(
    pobre          = "Pobre",
    dominio        = "Dominio",
    departamento   = "Depto",
    cuartos        = "P5000",
    dormitorios    = "P5010",
    tenencia       = "P5090",
    personas_ug    = "Npersug",
    linea_pobreza  = "Lp"
  )

  hogares |>
    rename(any_of(nombres)) |>
    select(id, any_of(names(nombres))) |>
    mutate(
      across(any_of(c("dominio", "departamento", "tenencia")), as.factor),
      personas_por_cuarto = personas_ug / pmax(dormitorios, 1)
    )
}

# Pipeline completo: limpia y une hogares con personas.
# Argumentos: hogares, personas = bases crudas del mismo split (train o test).
# Devuelve: base a nivel hogar lista para modelar.
construir_base_hogar <- function(hogares, personas) {
  variables_personas <- personas |>
    limpiar_personas() |>
    agregar_personas_a_hogar()

  hogares |>
    limpiar_hogares() |>
    left_join(variables_personas, by = "id")
}

# Convierte la variable objetivo a factor con "pobre" como primer nivel
# (yardstick toma el primer nivel como evento positivo para el F1).
# Argumentos: datos = base de train con pobre en 0/1.
# Devuelve: la misma base con pobre como factor.
preparar_objetivo <- function(datos) {
  datos |>
    mutate(pobre = factor(pobre, levels = c(1, 0), labels = c("pobre", "no_pobre")))
}
