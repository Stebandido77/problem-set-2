# ==============================================================================
# funciones/limpieza.R
# Carga, limpieza y unión de las bases de hogares y personas (DANE - MESE).
#
# Los códigos de las variables (P6020, P6100, P6430, ...) se verificaron contra
# el diccionario del DANE (ddi-documentation-spanish-608.pdf). Solo se usan
# variables que existen en el test.
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

  # guess_max alto: algunas columnas de ingreso vienen vacías en las primeras
  # miles de filas y read_csv las adivinaría como lógicas
  map(archivos, ~ read_csv(file.path(ruta, .x), show_col_types = FALSE, guess_max = 1e5))
}

# Reporta qué variables están en el train pero no en el test.
# Argumentos: train, test = data frames del mismo nivel (hogar o persona).
# Devuelve: tibble con las variables ausentes en el test.
variables_solo_en_train <- function(train, test) {
  tibble(variable = setdiff(names(train), names(test)))
}

# Indicador 0/1 para preguntas sí/no del DANE (1 = sí, 2 = no). NA cuenta como no,
# porque en la GEIH el NA casi siempre significa "no le aplicaba la pregunta".
# Argumentos: x = vector con códigos 1/2/NA.
# Devuelve: vector entero 0/1.
es_si <- function(x) {
  as.integer(!is.na(x) & x == 1)
}

# Años de educación aprobados a partir del nivel (P6210) y el grado (P6210s1).
# Primaria, secundaria y media reportan el grado absoluto (1o a 13o); superior
# reporta años aprobados en ese nivel, que se suman a 11 de bachillerato.
# Argumentos: nivel = P6210 (1 a 6, 9 = no sabe); grado = P6210s1.
# Devuelve: vector numérico de años (NA si no hay información).
calcular_anos_educacion <- function(nivel, grado) {
  grado <- if_else(grado %in% 98:99, NA_real_, as.numeric(grado))
  case_when(
    nivel %in% 1:2 ~ 0,
    nivel == 3     ~ pmin(coalesce(grado, 3), 5),
    nivel == 4     ~ pmax(coalesce(grado, 7), 5),
    nivel == 5     ~ pmax(coalesce(grado, 11), 9),
    nivel == 6     ~ 11 + pmin(coalesce(grado, 3), 10),
    .default = NA_real_
  )
}

# Construye variables a nivel persona a partir de la base individual.
# Solo usa variables que también están en el test.
# Argumentos: personas = base de personas cruda.
# Devuelve: tibble de personas con variables limpias.
limpiar_personas <- function(personas) {
  personas |>
    transmute(
      id,
      mujer          = as.integer(P6020 == 2),
      edad           = P6040,
      jefe_hogar     = as.integer(P6050 == 1),
      menor_5        = as.integer(P6040 < 5),
      menor_edad     = as.integer(P6040 < 18),
      adulto_mayor   = as.integer(P6040 >= 65),
    edad_trabajar  = as.integer(P6040 >= 15 & P6040 < 65),
      adulto         = as.integer(P6040 >= 18),
      # P6210: nivel educativo (9 = no sabe); faltante -> 0
      nivel_educ     = if_else(is.na(P6210) | P6210 == 9, 0, P6210),
      anos_educ      = calcular_anos_educacion(P6210, P6210s1),
      ocupado        = es_si(Oc),
      desocupado     = es_si(Des),
      afiliado_salud = es_si(P6090),
      # P6100 = 3: régimen subsidiado, la puerta de entrada es el SISBEN
      regimen_subsidiado = as.integer(P6100 %in% 3),
      # P6920 = 1 cotiza, 3 ya pensionado
      cotiza_pension = as.integer(P6920 %in% c(1, 3)),
      # Informalidad: ocupado que no cotiza a pensión
      informal       = as.integer(ocupado == 1 & P6920 %in% 2),
      cuenta_propia  = as.integer(ocupado == 1 & P6430 %in% 4),
      # P6870 de 1 a 3: empresa de 5 personas o menos
      empresa_pequena = as.integer(ocupado == 1 & P6870 %in% 1:3),
      posicion = case_when(
        ocupado == 0     ~ "no_ocupado",
        P6430 %in% 1:2   ~ "asalariado",
        P6430 == 4       ~ "cuenta_propia",
        P6430 == 5       ~ "patron",
        .default         = "otro"  # doméstico, sin remuneración, jornalero
      ),
      # Oficio (CIUO a dos dígitos): el primer dígito agrupa el gran grupo
      oficio_grupo = if_else(ocupado == 0 | is.na(Oficio), "no_ocupado",
                             paste0("g", Oficio %/% 10)),
      horas_trabajo  = if_else(ocupado == 1, coalesce(P6800, 0), 0),
      subempleo      = es_si(P7090),
      segundo_trabajo = es_si(P7040),
      # Indicadores sí/no de fuentes de ingreso (los montos no están en el test)
      recibe_subsidio_laboral = pmax(es_si(P6585s1), es_si(P6585s2),
                                     es_si(P6585s3), es_si(P6585s4)),
      recibe_prestaciones = pmax(es_si(P6545), es_si(P6580), es_si(P6630s1),
                                 es_si(P6630s2), es_si(P6630s3), es_si(P6630s4),
                                 es_si(P6630s6)),
      recibe_especie = pmax(es_si(P6590), es_si(P6600), es_si(P6610), es_si(P6620)),
      recibe_arriendos = es_si(P7495),
      recibe_pension   = es_si(P7500s2),
      recibe_remesas   = pmax(es_si(P7510s1), es_si(P7510s2)),
      recibe_ayudas_instituciones = es_si(P7510s3),
      recibe_rentas_capital = pmax(es_si(P7510s5), es_si(P7510s6))
    )
}

# Agrega la información de personas al nivel hogar.
# Argumentos: personas = salida de limpiar_personas().
# Devuelve: tibble con una fila por id de hogar.
agregar_personas_a_hogar <- function(personas) {
  # Educación de adultos: solo personas de 18 años o más
  personas <- personas |>
    mutate(anos_educ_adulto = if_else(adulto == 1, anos_educ, NA_real_))

  hogares <- personas |>
    group_by(id) |>
    summarise(
      n_miembros         = n(),
      n_menores          = sum(menor_edad, na.rm = TRUE),
      n_menores_5        = sum(menor_5, na.rm = TRUE),
      n_adultos_mayores  = sum(adulto_mayor, na.rm = TRUE),
      n_edad_trabajar    = sum(edad_trabajar, na.rm = TRUE),
      n_ocupados         = sum(ocupado, na.rm = TRUE),
      n_desocupados      = sum(desocupado, na.rm = TRUE),
      prop_ocupados      = n_ocupados / n_miembros,
      prop_mujeres       = mean(mujer, na.rm = TRUE),
      educ_maxima        = max(nivel_educ, na.rm = TRUE),
      educ_promedio      = mean(nivel_educ, na.rm = TRUE),
      educ_maxima_adultos   = suppressWarnings(max(anos_educ_adulto, na.rm = TRUE)),
      educ_promedio_adultos = mean(anos_educ_adulto, na.rm = TRUE),
      prop_afiliados_salud = mean(afiliado_salud, na.rm = TRUE),
      prop_subsidiado      = mean(regimen_subsidiado, na.rm = TRUE),
      prop_cotiza_pension  = mean(cotiza_pension, na.rm = TRUE),
      n_informales       = sum(informal),
      n_cuenta_propia    = sum(cuenta_propia),
      n_empresa_pequena  = sum(empresa_pequena),
      horas_trabajo_hogar = sum(horas_trabajo),
      across(
        c(subempleo, segundo_trabajo, starts_with("recibe_")),
        max
      ),
      .groups = "drop"
    ) |>
    rename(hay_subempleo = subempleo, hay_segundo_trabajo = segundo_trabajo)

  # Características del jefe: cada hogar tiene exactamente un jefe (P6050 = 1)
  jefes <- personas |>
    filter(jefe_hogar == 1) |>
    distinct(id, .keep_all = TRUE) |>
    transmute(
      id,
      jefe_mujer      = mujer,
      jefe_edad       = edad,
      jefe_educ       = nivel_educ,
      jefe_anos_educ  = anos_educ,
      jefe_ocupado    = ocupado,
      jefe_posicion   = posicion,
      jefe_oficio     = oficio_grupo,
      jefe_subsidiado = regimen_subsidiado,
      jefe_cotiza     = cotiza_pension
    )

  etiquetas_educ <- c("sin_dato", "ninguno", "preescolar", "primaria",
                      "secundaria", "media", "superior")

  hogares |>
    left_join(jefes, by = "id") |>
    mutate(
      tasa_dependencia = (n_menores + n_adultos_mayores) / pmax(n_edad_trabajar, 1),
      fuerza_laboral = n_ocupados + n_desocupados,
    tasa_desempleo_hogar = if_else(fuerza_laboral > 0, n_desocupados / fuerza_laboral, 0),
      prop_ocupados    = n_ocupados / n_miembros,
      # Proporciones sobre ocupados (0 si nadie trabaja)
      prop_informales      = if_else(n_ocupados > 0, n_informales / n_ocupados, 0),
      prop_cuenta_propia   = if_else(n_ocupados > 0, n_cuenta_propia / n_ocupados, 0),
      prop_empresa_pequena = if_else(n_ocupados > 0, n_empresa_pequena / n_ocupados, 0),
      horas_por_miembro    = horas_trabajo_hogar / n_miembros,
      educ_maxima_adultos   = if_else(is.finite(educ_maxima_adultos), educ_maxima_adultos, NA_real_),
      educ_promedio_adultos = if_else(is.finite(educ_promedio_adultos), educ_promedio_adultos, NA_real_),
      jefe_educ     = factor(jefe_educ, levels = 0:6, labels = etiquetas_educ),
      jefe_posicion = factor(jefe_posicion),
      jefe_oficio   = factor(jefe_oficio)
    )
}

# Limpia la base de hogares y renombra variables al español.
# any_of() permite que la misma función sirva para train (con pobre) y test.
# Argumentos: hogares = base de hogares cruda.
# Devuelve: tibble de hogares limpio.
limpiar_hogares <- function(hogares) {
  nombres <- c(
    pobre             = "Pobre",
    clase             = "Clase",
    dominio           = "Dominio",
    departamento      = "Depto",
    cuartos           = "P5000",
    dormitorios       = "P5010",
    tenencia          = "P5090",
    n_personas        = "Nper",
    personas_ug       = "Npersug",
    linea_pobreza     = "Lp",
    arriendo_estimado = "P5130",
    arriendo_pagado   = "P5140"
  )

  etiquetas_tenencia <- c("propia_pagada", "propia_pagando", "arriendo",
                          "usufructo", "posesion_sin_titulo", "otra")

  hogares |>
    rename(any_of(nombres)) |>
    select(id, any_of(names(nombres))) |>
    mutate(
      # 98 y 99 son "no sabe / no informa"; cualquier valor < 1.000 pesos no es un arriendo
      across(c(arriendo_estimado, arriendo_pagado), ~ if_else(.x < 1000, NA_real_, .x)),
      # Propietarios reportan arriendo estimado y arrendatarios el pagado: son complementarios
      arriendo = coalesce(arriendo_pagado, arriendo_estimado),
      log_arriendo = log(arriendo),
      # Arriendo frente a la línea de pobreza del hogar: proxy barato de capacidad de pago
      log_arriendo_relativo = log(arriendo / (linea_pobreza * n_personas)),
      clase = factor(if_else(clase == 1, "urbano", "rural")),
      tenencia = factor(tenencia, levels = 1:6, labels = etiquetas_tenencia),
      across(c(dominio, departamento), as.factor),
      personas_por_cuarto     = n_personas / pmax(cuartos, 1),
      personas_por_dormitorio = n_personas / pmax(dormitorios, 1),
      hacinamiento_critico = as.integer(personas_por_cuarto > 3)
    ) |>
    select(-arriendo_estimado, -arriendo_pagado)
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

# Conjuntos de variables de cada nivel de la escalera. Cambiar el conjunto de
# variables es cambiar de nivel: dentro de un nivel todos los algoritmos usan
# exactamente estas columnas y la misma receta.
# Devuelve: lista nombrada de vectores de nombres de variables.
definir_niveles <- function() {
  # Nivel 2: pocos predictores teóricos (tamaño, educación y ocupación del jefe)
  nivel_2 <- c("n_personas", "jefe_educ", "jefe_ocupado", "jefe_posicion")

  # Nivel 3: nivel 2 + ingeniería de variables desde personas + vivienda y ubicación
  desde_personas <- c(
    "n_menores", "n_menores_5", "n_adultos_mayores", "n_ocupados", "n_desocupados",
    "prop_ocupados", "tasa_dependencia", "prop_mujeres",
    "educ_maxima_adultos", "educ_promedio_adultos", "jefe_anos_educ",
    "jefe_mujer", "jefe_edad", "jefe_oficio", "jefe_subsidiado", "jefe_cotiza",
    "prop_afiliados_salud", "prop_subsidiado", "prop_cotiza_pension",
    "prop_informales", "prop_cuenta_propia", "prop_empresa_pequena",
    "horas_por_miembro", "hay_subempleo", "hay_segundo_trabajo",
    "recibe_subsidio_laboral", "recibe_prestaciones", "recibe_especie",
    "recibe_arriendos", "recibe_pension", "recibe_remesas",
    "recibe_ayudas_instituciones", "recibe_rentas_capital"
  )
  desde_hogar <- c(
    "clase", "dominio", "departamento", "cuartos", "dormitorios", "tenencia",
    "log_arriendo", "log_arriendo_relativo", "personas_por_cuarto",
    "personas_por_dormitorio", "linea_pobreza"
  )
  complementarias <- c(
    "tasa_desempleo_hogar", "hacinamiento_critico", "fuerza_laboral",
    "n_edad_trabajar", "n_informales", "n_cuenta_propia",
    "n_empresa_pequena", "horas_trabajo_hogar"
  )

  list(
    # Nivel 1: candidatas para la regla de una sola variable
    nivel_1 = c("personas_por_cuarto", "educ_maxima_adultos", "prop_ocupados"),
    nivel_2 = nivel_2,
    nivel_3 = c(nivel_2, desde_personas, desde_hogar),
    nivel_4v = c(nivel_2, desde_personas, desde_hogar, complementarias),
    # Nivel E: encuesta corta, 10 preguntas baratas y verificables en campo
    nivel_e = c(
      "n_personas", "dormitorios", "tenencia", "log_arriendo", "dominio",
      "educ_maxima_adultos", "n_ocupados", "n_menores_5", "prop_subsidiado",
      "jefe_mujer"
    )
  )
}
