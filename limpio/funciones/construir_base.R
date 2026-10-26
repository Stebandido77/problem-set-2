# =============================================================================
# construir_base.R — Base parsimoniosa de variables de hogar.
# Todo agregado de personas se calcula sobre la unidad de gasto: se excluye a
# servicio doméstico, pensionistas y trabajadores (P6050 en 6, 7, 8), igual que
# en Npersug del DANE.
# =============================================================================

PARENTESCOS_FUERA_UG <- c(6L, 7L, 8L)

# Construye la tabla de mes (rango de Lp dentro de cada dominio) y los niveles
# de dominio, ambos a partir del train para que train y test compartan códigos.
# Argumentos: hogares_train, data.table de hogares de train.
# Devuelve: lista con `meses` (Dominio, Lp, mes) y `niveles_dominio`.
crear_referencias <- function(hogares_train) {
  meses <- unique(hogares_train[, .(Dominio, Lp)])
  meses[, mes := frank(Lp, ties.method = "dense"), by = Dominio]
  list(meses = meses, niveles_dominio = sort(unique(hogares_train$Dominio)))
}

# Agrega las personas de la unidad de gasto al nivel de hogar.
# Argumentos: personas, data.table limpia (ver limpiar_codigos()).
# Devuelve: data.table con una fila por id y las variables de demografía,
# educación, mercado laboral y transferencias.
agregar_personas <- function(personas) {
  ug <- personas[!P6050 %in% PARENTESCOS_FUERA_UG]
  ug[, `:=`(
    es_jefe = P6050 == 1L,
    adulto = P6040 >= 18,
    edad_15_64 = P6040 >= 15 & P6040 <= 64
  )]
  hogar <- ug[, .(
    # Demografía
    n_menores_5 = sum(P6040 < 5),
    n_menores_18 = sum(P6040 < 18),
    n_mayores_65 = sum(P6040 >= 65),
    n_edad_trabajar = sum(Pet == 1L),
    tasa_dependencia = sum(P6040 < 15 | P6040 > 64) / max(sum(edad_15_64), 1),
    prop_mujeres = mean(P6020 == 2L),
    jefe_mujer = as.integer(any(es_jefe & P6020 == 2L)),
    edad_jefe = P6040[es_jefe][1],
    jefe_con_pareja = as.integer(any(P6050 == 2L)),
    n_hijos = sum(P6050 == 3L),
    n_otros_parientes = sum(P6050 %in% c(4L, 5L)),
    # Educación
    educ_jefe = P6210[es_jefe][1],
    grado_jefe = P6210s1[es_jefe][1],
    educ_max_hogar = suppressWarnings(as.numeric(max(P6210, na.rm = TRUE))),
    educ_media_adultos = mean(P6210[adulto], na.rm = TRUE),
    prop_adultos_superior = mean(P6210[adulto] %in% 6L),
    n_estudiando = sum(P6240 %in% 3L),
    n_menores_no_estudian = sum(P6040 >= 10 & P6040 <= 17 & !P6240 %in% 3L),
    # Mercado laboral
    n_ocupados = sum(Oc),
    n_desocupados = sum(Des),
    n_inactivos = sum(Ina),
    horas_totales = as.numeric(sum(P6800, na.rm = TRUE)),
    horas_jefe = fcoalesce(as.numeric(P6800[es_jefe][1]), 0),
    n_cotiza_pension = sum(P6920 %in% 1L),
    n_pensionados = sum(P6920 %in% 3L),
    prop_salud_contributiva = mean(P6100 %in% c(1L, 2L)),
    n_salud_subsidiada = sum(P6100 %in% 3L),
    n_sin_afiliacion = sum(P6090 %in% 2L),
    n_empleado_particular = sum(P6430 %in% 1L),
    n_empleado_gobierno = sum(P6430 %in% 2L),
    n_empleado_domestico = sum(P6430 %in% 3L),
    n_cuenta_propia = sum(P6430 %in% 4L),
    n_patron = sum(P6430 %in% 5L),
    n_sin_remuneracion = sum(P6430 %in% c(6L, 7L)),
    n_jornalero = sum(P6430 %in% 8L),
    posicion_jefe = fcoalesce(P6430[es_jefe][1], 0L),
    # Primer dígito del oficio (0-9); 10 = jefe sin oficio (no ocupado)
    oficio_jefe = fcoalesce(as.integer(Oficio[es_jefe][1] %/% 10), 10L),
    n_empresa_grande = sum(P6870 >= 6, na.rm = TRUE),
    n_empresa_unipersonal = sum(P6870 %in% 1L),
    n_quiere_mas_horas = sum(P7090 %in% 1L),
    n_segundo_empleo = sum(P7040 %in% 1L),
    n_recibe_primas = sum(P6545 %in% 1L),
    n_subsidio_familiar = sum(P6585s3 %in% 1L),
    n_subsidio_transporte = sum(P6585s2 %in% 1L),
    # Transferencias e ingresos no laborales (indicadores de hogar)
    recibe_arriendos_pension = as.integer(any(P7495 %in% 1L)),
    recibe_pension_jubilacion = as.integer(any(P7500s2 %in% 1L)),
    recibe_pension_alimenticia = as.integer(any(P7500s3 %in% 1L)),
    recibe_otros_ingresos_12m = as.integer(any(P7505 %in% 1L)),
    recibe_ayuda_hogares_pais = as.integer(any(P7510s1 %in% 1L)),
    recibe_remesas_exterior = as.integer(any(P7510s2 %in% 1L)),
    recibe_ayuda_instituciones = as.integer(any(P7510s3 %in% 1L)),
    recibe_intereses_dividendos = as.integer(any(P7510s5 %in% 1L)),
    recibe_cesantias = as.integer(any(P7510s6 %in% 1L)),
    recibe_otras_fuentes = as.integer(any(P7510s7 %in% 1L))
  ), by = id]
  hogar[is.infinite(educ_max_hogar), educ_max_hogar := NA]
  hogar[is.nan(educ_media_adultos), educ_media_adultos := NA]
  hogar[, n_fuentes_no_laborales := recibe_arriendos_pension +
    recibe_pension_jubilacion + recibe_pension_alimenticia +
    recibe_ayuda_hogares_pais + recibe_remesas_exterior +
    recibe_ayuda_instituciones + recibe_intereses_dividendos +
    recibe_cesantias + recibe_otras_fuentes]
  hogar
}

# Construye la base de hogar completa de una muestra.
# Argumentos:
#   muestra, lista de leer_muestra() (se limpia por referencia).
#   referencias, lista de crear_referencias().
# Devuelve: data.table con id, variables explicativas y, en train, Pobre e
# Ingpcug. Las categóricas quedan como enteros.
construir_base_hogar <- function(muestra, referencias) {
  limpiar_codigos(muestra$hogares, muestra$personas)
  h <- copy(muestra$hogares)
  h <- merge(h, referencias$meses, by = c("Dominio", "Lp"), all.x = TRUE)
  stopifnot(!anyNA(h$mes))
  h[, `:=`(
    dominio = match(Dominio, referencias$niveles_dominio),
    rural = as.integer(Clase == 2L),
    lp = Lp,
    n_personas_ug = Npersug,
    n_fuera_ug = Nper - Npersug,
    cuartos_pc = P5000 / Npersug,
    dormitorios_pc = P5010 / Npersug,
    hacinamiento = Npersug / P5010,
    tenencia = P5090,
    arriendo = fcoalesce(P5140, P5130),
    arriendo_es_estimado = as.integer(is.na(P5140) & !is.na(P5130)),
    tiene_hipoteca = as.integer(P5090 == 2L),
    cuota_hipoteca_lp = P5100 / Lp,
    es_bogota = as.integer(Dominio == DOMINIO_AUSENTE_TEST)
  )]
  h[, `:=`(
    log_arriendo = log(arriendo),
    arriendo_pc_lp = arriendo / Npersug / Lp
  )]
  hogar_personas <- agregar_personas(muestra$personas)
  h <- merge(h, hogar_personas, by = "id", all.x = TRUE)
  stopifnot(nrow(h) == nrow(muestra$hogares))
  h[, `:=`(
    ocupados_pc = n_ocupados / Npersug,
    horas_pc = horas_totales / Npersug,
    cotizantes_pc = n_cotiza_pension / Npersug
  )]
  descartar <- c(COLUMNAS_HOGARES[-1], "Lp")
  h[, (intersect(descartar, names(h))) := NULL]
  setcolorder(h, "id")
  h
}

# Nombres de las variables explicativas de la base (todo menos id y objetivos).
# Argumentos: base, data.table de construir_base_hogar().
# Devuelve: vector de caracteres.
variables_explicativas <- function(base) {
  setdiff(names(base), c("id", "Pobre", "Ingpcug", "es_bogota"))
}
