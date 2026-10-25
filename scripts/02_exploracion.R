# ==============================================================================
# 02_exploracion.R
# Stage 1: desbalance de clases y relación de la pobreza con los predictores
# que sugiere la teoría económica. Salidas en views/.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
bases <- cargar_bases_modelado()
train <- bases$train

# ---- Desbalance de clases ----------------------------------------------------
desbalance <- train |>
  count(pobre) |>
  mutate(proporcion = n / sum(n))
guardar_tabla(desbalance, "desbalance_clases")

# ---- Tasa de pobreza según predictores clave ---------------------------------
# Función auxiliar: tasa de pobreza por categorías de una variable
tasa_pobreza_por <- function(datos, variable, etiqueta) {
  datos |>
    group_by(valor = as.factor({{ variable }})) |>
    summarise(tasa_pobreza = mean(pobre == "pobre"), n = n(), .groups = "drop") |>
    mutate(variable = etiqueta)
}

grafico_educacion <- tasa_pobreza_por(train, jefe_educ, "Educación del jefe") |>
  ggplot(aes(valor, tasa_pobreza)) +
  geom_col(fill = "#2c7fb8") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "La pobreza cae con la educación del jefe de hogar",
    x = "Nivel educativo del jefe (P6210)", y = "Tasa de pobreza"
  )
guardar_figura(grafico_educacion, "pobreza_educacion_jefe")

grafico_dependencia <- train |>
  mutate(tramo = cut(tasa_dependencia, breaks = c(-Inf, 0, 0.5, 1, 2, Inf))) |>
  tasa_pobreza_por(tramo, "Tasa de dependencia") |>
  ggplot(aes(valor, tasa_pobreza)) +
  geom_col(fill = "#2c7fb8") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Más dependientes por adulto en edad de trabajar, más pobreza",
    x = "Tasa de dependencia", y = "Tasa de pobreza"
  )
guardar_figura(grafico_dependencia, "pobreza_tasa_dependencia")

grafico_desempleo <- train |>
  mutate(tramo_desempleo = case_when(
    tasa_desempleo_hogar == 0 ~ "0%",
    tasa_desempleo_hogar > 0 & tasa_desempleo_hogar <= 0.5 ~ "1% - 50%",
    tasa_desempleo_hogar > 0.5 ~ "> 50%"
  )) |>
  tasa_pobreza_por(tramo_desempleo, "Tasa de desempleo del hogar") |>
  ggplot(aes(valor, tasa_pobreza)) +
  geom_col(fill = "#e34a33") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Incidencia de la pobreza según el desempleo del hogar",
    x = "Tasa de desempleo en el hogar", y = "Tasa de pobreza"
  ) +
  theme_minimal()
guardar_figura(grafico_desempleo, "pobreza_desempleo_hogar")

# ---- Resumen descriptivo -----------------------------------------------------
skim(train)
