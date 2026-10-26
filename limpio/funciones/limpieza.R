# =============================================================================
# limpieza.R — Lectura de los CSV crudos y tratamiento de los códigos
# especiales definidos en el diccionario del DANE (ver limpio/diseno.md §1.1).
# =============================================================================

# Columnas de personas comunes a train y test que usa la base.
COLUMNAS_PERSONAS <- c(
  "id", "Orden", "P6020", "P6040", "P6050", "P6090", "P6100", "P6210",
  "P6210s1", "P6240", "Oficio", "P6430", "P6545", "P6585s2", "P6585s3",
  "P6800", "P6870", "P6920", "P7040", "P7090", "P7495", "P7500s2", "P7500s3",
  "P7505", "P7510s1", "P7510s2", "P7510s3", "P7510s5", "P7510s6", "P7510s7",
  "Pet", "Oc", "Des", "Ina"
)

COLUMNAS_HOGARES <- c(
  "id", "Clase", "Dominio", "P5000", "P5010", "P5090", "P5100", "P5130",
  "P5140", "Nper", "Npersug", "Lp"
)

# Variables cuyo código 9 significa "no sabe, no informa".
VARIABLES_CODIGO_9 <- c(
  "P6090", "P6100", "P6210", "P6545", "P6585s2", "P6585s3", "P7500s2",
  "P7500s3", "P7510s1", "P7510s2", "P7510s3", "P7510s5", "P7510s6", "P7510s7"
)

# Montos y conteos de hogar cuyos códigos 98/99 son "no sabe"/"no informa".
VARIABLES_CODIGO_98_99 <- c("P5000", "P5010", "P5100", "P5130", "P5140")

TOPE_HORAS <- 120

# Lee hogares y personas de una muestra ("train" o "test").
# Argumentos: muestra, "train" o "test".
# Devuelve: lista con data.tables `hogares` y `personas`. En train, `hogares`
# incluye además Pobre e Ingpcug (objetivo y objetivo auxiliar).
leer_muestra <- function(muestra) {
  extra <- if (muestra == "train") c("Pobre", "Ingpcug") else character(0)
  hogares <- fread(
    file.path(RUTA_RAW, paste0(muestra, "_hogares.csv")),
    select = c(COLUMNAS_HOGARES, extra)
  )
  personas <- fread(
    file.path(RUTA_RAW, paste0(muestra, "_personas.csv")),
    select = COLUMNAS_PERSONAS
  )
  list(hogares = hogares, personas = personas)
}

# Pasa a NA los códigos especiales y topa las horas.
# Argumentos: hogares y personas (data.tables, se modifican por referencia).
# Devuelve: invisible, la lista con ambas tablas limpias.
limpiar_codigos <- function(hogares, personas) {
  for (v in VARIABLES_CODIGO_98_99) {
    set(hogares, which(hogares[[v]] %in% c(98, 99)), v, NA)
  }
  for (v in VARIABLES_CODIGO_9) {
    set(personas, which(personas[[v]] == 9), v, NA)
  }
  personas[P6800 > TOPE_HORAS, P6800 := TOPE_HORAS]
  for (v in c("Pet", "Oc", "Des", "Ina")) {
    set(personas, which(is.na(personas[[v]])), v, 0L)
  }
  invisible(list(hogares = hogares, personas = personas))
}
