# Problem Set 2: Predicting Poverty

> *"Wars of nations are fought to change maps. But wars of poverty are fought to map change."* — M. Ali

**Big Data and Machine Learning para Economía Aplicada (MECA 4107)** · Universidad de los Andes · 2026-20 · **Equipo 06**

Predecimos la pobreza de los hogares colombianos con datos de la MESE (DANE) como un problema de
clasificación, y nos preguntamos qué modelo le serviría de verdad a un hacedor de política:
uno que prediga bien **y** que se pueda aplicar con una encuesta corta y barata.

## Integrantes

| Nombre | GitHub |
|---|---|
| Esteban | [@Stebandido77](https://github.com/Stebandido77) |
| Neislen | [@NeislenG](https://github.com/NeislenG) |
| _Por completar_ | |

## Estructura del repositorio

```
.
├── document/
│   ├── enunciado_problem_set_2.pdf   # Enunciado del profesor
│   └── slides/                       # best_equipo_06.pdf y compare_equipo_06.pdf
├── scripts/
│   ├── 00_configuracion.R            # Paquetes, rutas, semilla
│   ├── 01_preparar_datos.R           # Limpieza y unión hogares-personas
│   ├── 02_exploracion.R              # Desbalance y relaciones con predictores
│   ├── 03_modelos_lineales.R         # LPM, Logit, Elastic Net
│   ├── 04_modelos_arboles.R          # CART, Random Forest, Boosting
│   ├── 05_naive_bayes.R              # Naive Bayes
│   ├── 06_balanceo_clases.R          # SMOTE y submuestreo sobre el mejor modelo
│   ├── 07_comparacion.R              # Deck compare: tabla y gráfico de algoritmos
│   ├── 08_mejor_modelo.R             # Deck best: importancia, trade-off de política
│   ├── main.R                        # Corre todo en orden
│   └── funciones/                    # Funciones reutilizables
│       ├── limpieza.R
│       ├── modelado.R
│       └── utilidades.R
├── stores/
│   ├── raw/                          # CSV de Kaggle (no se suben)
│   ├── procesados/                   # Bases limpias (no se suben)
│   ├── resultados/                   # metricas_cv.csv (se sube) y .rds (no)
│   └── submissions/                  # CSV enviados a Kaggle + bitacora_kaggle.csv
└── views/
    ├── figuras/                      # Figuras en PDF y PNG para las slides
    └── tablas/                       # Tablas en CSV y LaTeX para las slides
```

## Cómo reproducir

1. Clonar el repo y abrir `problem-set-2.Rproj` en RStudio.
2. Descargar de Kaggle `train_hogares.csv`, `train_personas.csv`, `test_hogares.csv` y
   `test_personas.csv`, y ponerlos en `stores/raw/`.
3. Correr `scripts/main.R`. Los paquetes se instalan solos con `pacman`.

Cada corrida de modelo escribe su CSV en `stores/submissions/`, agrega una fila a
`bitacora_kaggle.csv` y guarda sus métricas en `stores/resultados/metricas_cv.csv`.
Después de subir a Kaggle, anoten el F1 público a mano en la bitácora.

## Decisiones de diseño

- **Clasificación directa** de `pobre`, como pide el enunciado.
- **Cancha nivelada**: todos los algoritmos comparten la misma base y la misma receta
  (`crear_receta()`), así las diferencias de desempeño reflejan el algoritmo.
- Solo usamos variables presentes en **train y test**.
- Hiperparámetros por **validación cruzada de 5 folds** estratificada, con grilla regular o
  búsqueda *space-filling* según el número de hiperparámetros.
- El **umbral de clasificación** se elige maximizando el F1 fuera de muestra, y se reporta la
  curva completa para discutir filtración vs. subcobertura.

## Checklist de entregables

- [ ] ≥ 15 submissions en Kaggle
- [ ] ≥ 5 algoritmos distintos (LPM, Logit, Elastic Net, CART, RF, Boosting, Naive Bayes)
- [ ] ≥ 5 commits sustanciales por integrante en `main`
- [ ] `document/slides/best_equipo_06.pdf`
- [ ] `document/slides/compare_equipo_06.pdf`
- [ ] Ambos decks con la diapositiva *Best Model Overview* de primera

## Cómo trabajamos

- Mensajes de commit en español y descriptivos (`Agrega tasa de dependencia al hogar`).
- Una iteración del modelo = un commit.
- Si usan ramas, fusionen a `main` con *merge commit* (no *squash*) para que cada commit
  conserve a su autor.
- `.claude/` tiene la configuración compartida de Claude Code: sin firmas de coautoría y con
  las reglas de estilo del proyecto.
