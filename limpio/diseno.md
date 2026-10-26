# Etapa 17 — Diseño desde cero (paso 1)

Documento de diseño previo a construir cualquier cosa. Se escribió leyendo solo
`stores/raw/`, `limpio/ids/` y el diccionario del DANE
(`data/ddi-documentation-spanish-608.pdf`, *Medición de Pobreza Monetaria y
Desigualdad 2018*), sin mirar el trabajo de etapas anteriores.

## 1. Lo que muestran los datos

**Tamaños.** Train: 164 960 hogares y 543 109 personas. Test: 66 168 hogares y
219 644 personas. Todo hogar tiene al menos una persona y exactamente un jefe
(`P6050 == 1`). Los ids de `limpio/ids/` cubren todo el train: holdout de 32 993
hogares más un 80 % de 131 967, y cada partición lo divide en 5 folds de unos 26 400.

**Objetivo.** `Pobre` vale 1 en el 20,0 % de los hogares. Se cumple exactamente que
`Pobre == (Ingpcug < Lp)`, con `Ingpcug = Ingtotugarr / Npersug`. O sea que la
pobreza es un umbral sobre el ingreso per cápita de la unidad de gasto, y `Lp` es
la línea que le toca a cada hogar.

**Hallazgos que cambian el diseño.**

1. **No hay Bogotá en el test.** El train trae 10 567 hogares de Bogotá (Depto 11),
   con una tasa de pobreza del 9,8 %, la más baja después de Manizales y Pereira.
   El test tiene 24 dominios y ninguno es Bogotá. La pobreza del train sin Bogotá es
   del 20,7 %. Bogotá desplaza la tasa base y el umbral óptimo de F1.
2. **`Lp` codifica dominio × mes.** Cada dominio tiene exactamente 12 valores de `Lp`,
   y todas las combinaciones (dominio, `Lp`) del test aparecen en el train. Con eso
   se recupera el mes de la encuesta. La pobreza casi no varía por mes (19 %–21 %),
   así que el mes sirve como control, no como señal.
3. **La unidad de gasto excluye a quienes tienen `P6050 ∈ {6, 7, 8}`**
   (servicio doméstico, pensionistas y trabajadores del hogar). Se cumple
   `Nper − Npersug = #{P6050 ∈ 6,7,8}` en el 100 % de los hogares. Todo agregado
   per cápita tiene que contar solo a los miembros de la unidad de gasto.
4. **Hay códigos de "no sabe" mezclados con montos** (verificado en el diccionario,
   ver §1.1). Train: `P5130` tiene 8 399 hogares con 98 y 1 272 con 99, `P5140`
   tiene 609 y 141, `P5100` tiene 388 y 108, y `P5000` tiene un caso con 98. En
   las categóricas, el 9 aparece de decenas a cientos de veces por variable. Hay que
   pasarlos a NA antes de hacer cualquier cuenta. Los mismos códigos aparecen en el
   test en proporciones parecidas.
5. **El arriendo es el único monto disponible en el test, y nunca falta.**
   `P5130` (arriendo estimado para propietarios, usufructuarios y ocupantes de hecho)
   y `P5140` (arriendo pagado, sin administración) son excluyentes. Juntos cubren al
   100 % de los hogares, salvo los códigos 98/99. Todos los montos son **pesos
   corrientes mensuales** de 2018. `P5100` (cuota de hipoteca) falta en el 97 %.
6. **Las personas del test no traen ningún ingreso en pesos.** Solo traen
   indicadores de si la persona recibió cada fuente (`P6585s*`, `P6545`, `P7495`,
   `P7505`, `P7510s*`, ...). Los montos (`P6500`, `*s1a1`, `Ingtot*`, `Imp*`, `Iof*`)
   y `Estrato1` solo existen en el train y quedan fuera. En las 63 variables comunes,
   los patrones de faltantes del train y del test coinciden casi al decimal, porque
   esos NA son saltos del cuestionario y no pérdida de información.

**Señales univariadas (AUC, train completo).**

| Variable candidata (hogar)                            | AUC   |
|-------------------------------------------------------|-------|
| Personas con salud contributiva / Nper                | 0,78  |
| Arriendo (`P5140` o `P5130`) / Npersug / Lp           | 0,78  |
| Cuartos / Npersug                                     | 0,73  |
| Ocupados / Npersug                                    | 0,73  |
| Ocupados que cotizan pensión / Npersug                | 0,72  |
| Horas trabajadas / Npersug / Lp                       | 0,72  |
| Menores de 18 / Nper (signo inverso)                  | 0,72  |
| Algún miembro con educación superior                  | 0,64  |

Entre los jefes de hogar, la pobreza va del 8 % (superior) al 40 % (ninguno) según
la educación y del 4 % (especial) al 37 % (subsidiado) según el régimen de salud.
Según la posición ocupacional va del 1 % al 34 %: empleado del gobierno 1 %, patrón
8 %, empleado particular 10 %, jornalero 16 %, cuenta propia 27 % y trabajador
familiar sin remuneración 34 %. Llega al 44 % si el jefe recibe ayudas de
instituciones (`P7510s3`) y al 43 % si está desocupado.

### 1.1 Definiciones verificadas en el diccionario

| Variable | Códigos (según el diccionario) | Tratamiento |
|---|---|---|
| `P5000`, `P5010` | cuartos; 98 no sabe, 99 no informa | 98/99 → NA |
| `P5090` | 1 propia pagada, 2 propia pagando, 3 arriendo/subarriendo, 4 usufructo, 5 posesión sin título (ocupante de hecho), 6 otra | factor de 6 niveles |
| `P5100` | cuota mensual de amortización, $; 98 no sabe, 99 no informa | 98/99 → NA |
| `P5130` | arriendo estimado mensual, $ (solo si `P5090 ∈ {1,2,4,5,6}`); 98/99 | 98/99 → NA |
| `P5140` | arriendo pagado mensual, $ (solo si `P5090 = 3`); 99 no sabe, 98 paga pero no informa | 98/99 → NA |
| `P6050` | 1 jefe, 2 pareja, 3 hijo/hijastro, 4 nieto, 5 otro pariente, 6 empleado del servicio doméstico, 7 pensionista, 8 trabajador, 9 otro no pariente | 6–8 fuera de la unidad de gasto (el 9 sí cuenta) |
| `P6090` | 1 sí, 2 no, 9 no sabe | 9 → NA |
| `P6100` | 1 contributivo, 2 especial, 3 subsidiado, 9 no sabe | 9 → NA |
| `P6210` | 1 ninguno, 2 preescolar, 3 primaria, 4 secundaria, 5 media, 6 superior, 9 no sabe | 9 → NA; ordinal |
| `P6240` | 1 trabajando, 2 buscando trabajo, 3 estudiando, 4 oficios del hogar, 5 incapacitado, 6 otra | factor |
| `P6430` | 1 empleado particular, 2 empleado del gobierno, 3 empleado doméstico, 4 cuenta propia, 5 patrón, 6 familiar sin remuneración, 7 sin remuneración en otro hogar, 8 jornalero o peón, 9 otro | factor |
| `P6870` | 1 trabaja solo, 2 de 2 a 3, 3 de 4 a 5, 4 de 6 a 10, 5 de 11 a 19, 6 de 20 a 30, 7 de 31 a 50, 8 de 51 a 100, 9 más de 100 personas | ordinal |
| `P6920` | 1 cotiza, 2 no cotiza, 3 ya pensionado | factor |
| `P6545`, `P6580`, `P6585s1–s4`, `P6590–P6620`, `P7500s*`, `P7510s*` | 1 sí, 2 no, 9 no sabe | 9 → NA |
| `P6630s*`, `P7040`, `P7090`, `P7495`, `P7505` | 1 sí, 2 no | sin códigos especiales |
| `P6040` | años; 99 = 99 o más | se deja igual (máx. observado 110) |
| `P6426` | meses en la empresa; 998 / 999 | no aparecen en los datos |
| `P6800` | horas semanales normales | sin código especial; valores ≥ 98 (972 casos) se dejan, con tope en 120 |
| `Oficio` | código de 2 dígitos, sin etiquetas en el diccionario (84 niveles) | agrupar por primer dígito (grupo mayor CIUO-68, supuesto) |
| `Pet`, `Oc`, `Des`, `Ina` | 1 sí; NA cuando no aplica | NA → 0 |
| `Lp`, `Li` | pesos mensuales por persona | `Lp` como normalizador |

## 2. Hipótesis

No podemos ver el ingreso, pero sí sus tres componentes, y los tres son baratos de
preguntar:

```
ingreso per cápita ≈ (cuántos generan ingreso / cuántos dependen de él)
                     × (calidad del empleo de quienes trabajan)
                     + transferencias
```

Lo que define la pobreza es la **posición relativa a `Lp`**, no el nivel. Un modelo
que vea el arriendo, la intensidad del trabajo y la formalidad *en relación con la
línea y con el tamaño de la unidad de gasto* debería separar mejor que uno que ve
conteos crudos. El arriendo funciona como proxy del ingreso permanente: es el único
monto en pesos del test y por sí solo ya da un AUC de 0,78.

La hipótesis que se puede poner a prueba: **una base de unas 40 variables de hogar,
bien construida (unidad de gasto correcta, códigos 98/99 limpios, razones respecto
de Lp), con un solo gradient boosting, iguala o supera a una base ancha de cientos
de agregados automáticos.** Si eso se cumple, también gana la historia de costo de
recolección.

## 3. Plan de variables

Todo se calcula por hogar sobre los **miembros de la unidad de gasto**
(`P6050 ∉ {6, 7, 8}`), salvo que se diga otra cosa. Cada variable lleva su costo de
recolección: **B** (bajo: una pregunta al informante), **M** (medio: módulo laboral
de cada persona).

### 3.1 Hogar y vivienda (tabla de hogares) — B
- `n_personas_ug` (`Npersug`), `n_fuera_ug` (`Nper − Npersug`)
- `cuartos_pc` = `P5000 / Npersug`, `dormitorios_pc` = `P5010 / Npersug`, `hacinamiento` = `Npersug / P5010`
- `tenencia` (`P5090`, 6 categorías; "posesión sin título" tiene un 48 % de pobres)
- `arriendo` = `coalesce(P5140, P5130)` con 98/99 → NA; `log_arriendo`;
  `arriendo_pc_lp` = `arriendo / Npersug / Lp`; `arriendo_es_estimado` (vino de P5130)
- `tiene_hipoteca` = `!is.na(P5100)`, `cuota_hipoteca_lp`
- Ubicación: `dominio` (24 niveles + Bogotá solo en train), `rural` (`Clase == 2`),
  `mes` (rango de `Lp` dentro del dominio), `lp` como numérica

### 3.2 Demografía y dependencia (personas) — B
- `n_menores_5`, `n_menores_18`, `n_mayores_65`, `n_edad_trabajar`
- `tasa_dependencia` = (menores de 15 + mayores de 64) / personas de 15 a 64
- `prop_mujeres`, `jefe_mujer`, `edad_jefe`, `jefe_con_pareja` (algún `P6050 == 2`)
- `hogar_unipersonal`, `n_hijos`, `n_otros_parientes`

### 3.3 Educación — B
- `educ_jefe` (`P6210`, ordinal), `anios_educ_jefe` (`P6210s1`)
- `educ_max_hogar`, `educ_media_adultos` (mayores de 18), `prop_adultos_superior`
- `n_estudiando` (`P6240 == 3`) y si algún menor de 6 a 17 años **no** estudia

### 3.4 Mercado laboral — M
- `n_ocupados`, `n_desocupados`, `n_inactivos`; `ocupados_pc` = `n_ocupados / Npersug`
- `horas_totales_pc_lp` = Σ`P6800` / Npersug / Lp; `horas_jefe`
- Formalidad: `n_cotiza_pension` (`P6920 == 1`), `n_pensionados` (`P6920 == 3`),
  `prop_salud_contributiva` (`P6100 ∈ {1, 2}`), `n_salud_subsidiada` (`P6100 == 3`),
  `n_sin_afiliacion` (`P6090 == 2`)
- Posición: `n_asalariados_privados`, `n_publicos`, `n_cuenta_propia`, `n_patrones`,
  `n_jornaleros`, `n_sin_remuneracion` (`P6430`); `posicion_jefe`
- `n_empresa_grande` (`P6870 ≥ 6`, 20 o más personas), `n_empresa_unipersonal` (`P6870 == 1`)
- `n_quiere_mas_horas` (`P7090 == 1`), `n_segundo_empleo` (`P7040 == 1`)
- `oficio_jefe` (`Oficio`, agrupado en su primer dígito CIUO para no tener 90 niveles)
- Prestaciones como señal de empleo formal: `n_recibe_primas` (`P6545`),
  `n_subsidio_familiar` (`P6585s3`), `n_subsidio_transporte` (`P6585s2`)

### 3.5 Transferencias e ingresos no laborales (indicadores sí/no) — B
- `recibe_arriendos_pension` (`P7495`, mes pasado), `recibe_pension_jubilacion`
  (`P7500s2`), `recibe_pension_alimenticia` (`P7500s3`)
- `recibe_otros_ingresos_12m` (`P7505`: dinero de hogares o de ONG, intereses,
  dividendos o cesantías; es la pregunta filtro de los `P7510s*`)
- Detalle de los `P7510s*` (últimos 12 meses): `recibe_ayuda_hogares_pais` (s1),
  `recibe_remesas_exterior` (s2), `recibe_ayuda_instituciones` (s3, instituciones del
  país o de fuera), `recibe_intereses_dividendos` (s5), `recibe_cesantias` (s6),
  `recibe_otras_fuentes` (s7)
- `n_fuentes_no_laborales` (conteo)

### 3.6 Lo que queda fuera (y por qué)
- Montos de ingreso, `Estrato1`, `Ingtot*`, `Npobres`, `Indigente`, `Li` como
  objetivo: no están en el test o son el resultado mismo.
- `Fex_c` y `Fex_dpto` no entran como predictores (son diseño muestral, no
  características del hogar). Quedan disponibles como pesos en un experimento aparte.
- Agregados automáticos de las 63 columnas (media/máx/mín de todo): son justo la base
  ancha contra la que se compara la hipótesis.

## 4. Validación y reglas de juego

- **Particiones:** las de `limpio/ids/`. Se busca en CV de 5 folds sobre
  `particion_1` y se confirma con `particion_2` y `particion_3`. El holdout se toca
  **una sola vez**, al final, para el modelo elegido y su umbral.
- **Umbral de F1:** se elige con las predicciones fuera de fold, nunca en el holdout.
- **Bogotá:** se compara (a) entrenar con Bogotá y (b) sin Bogotá, evaluando en folds
  **sin Bogotá** en los dos casos, porque así se parece más al test. El umbral se fija
  con la distribución del test en mente (sin Bogotá).
- **Sin fuga:** toda variable se construye solo con columnas comunes a train y test;
  ningún agregado usa `Pobre` (nada de target encoding fuera de fold).
- **Cancha nivelada:** una sola función que construye la base y una sola receta para
  los algoritmos de la lista del enunciado (Logit, Elastic Net, CART, RF, Boosting,
  Naive Bayes, LPM).

## 5. Orden propuesto de construcción (pasos siguientes, no ejecutados)

1. `construir_base_hogar()`: limpieza de 98/99, unidad de gasto y variables 3.1 a 3.5.
   Pruebas: cero NA inesperados, `Npersug` reconstruido = el original, mismas columnas
   en train y test.
2. Línea base: Logit con 10 variables (las de la tabla de AUC) → F1 en CV.
3. LightGBM sobre la base completa del punto 3 → F1 en CV, por dominio y con/sin Bogotá.
4. Versión reducida a "encuesta corta" (solo variables B) para medir cuánto cuesta en
   F1 quitar el módulo laboral.

## 6. Dudas abiertas para ti

- **Diccionario:** no estaba en `stores/raw/` ni en `limpio/diccionario/`. Usé la
  copia que hay en `data/ddi-documentation-spanish-608.pdf`, que también está en
  Descargas. Las definiciones de §1.1 ya están verificadas contra ese PDF. Lo único
  que el PDF no trae son las etiquetas de `Oficio`, así que agruparlo por primer
  dígito es un supuesto.
- ¿El experimento de Bogotá entra en esta etapa o lo dejamos fijo (entrenar con
  Bogotá) para no abrir otro frente?
- Lenguaje: el proyecto es en R y aquí hay R 4.6 con `data.table`, `tidymodels` y
  `lightgbm`. Mi plan es hacer todo en R dentro de `limpio/`. ¿Te sirve?
