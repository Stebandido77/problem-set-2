# =============================================================================
# metricas.R — F1, precisión, recall y búsqueda del umbral de F1.
# Precisión baja = filtración (falsos positivos); recall bajo = subcobertura
# (falsos negativos).
# =============================================================================

# Calcula F1, precisión y recall con un umbral dado.
# Argumentos: y (0/1), p (probabilidades), umbral.
# Devuelve: lista con f1, precision, recall y umbral.
metricas_en_umbral <- function(y, p, umbral) {
  pred <- p >= umbral
  vp <- sum(pred & y == 1)
  precision <- if (sum(pred) > 0) vp / sum(pred) else 0
  recall <- vp / sum(y == 1)
  f1 <- if (vp > 0) 2 * precision * recall / (precision + recall) else 0
  list(f1 = f1, precision = precision, recall = recall, umbral = umbral)
}

# Busca el umbral que maximiza F1 recorriendo todas las probabilidades
# observadas (ordenadas de mayor a menor, conteo acumulado de verdaderos
# positivos).
# Argumentos: y (0/1), p (probabilidades).
# Devuelve: lista con f1, precision, recall y umbral óptimos.
buscar_umbral_f1 <- function(y, p) {
  orden <- order(p, decreasing = TRUE)
  y_ord <- y[orden]
  vp <- cumsum(y_ord)
  predichos <- seq_along(y_ord)
  f1 <- 2 * vp / (predichos + sum(y_ord))
  # Solo se puede cortar donde cambia la probabilidad
  validos <- c(diff(p[orden]) != 0, TRUE)
  mejor <- which(validos)[which.max(f1[validos])]
  metricas_en_umbral(y, p, p[orden][mejor])
}
