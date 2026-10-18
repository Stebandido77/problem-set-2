"""Variables v2 del nivel 7, por grupos, a partir de las bases de personas y hogar.

Solo usan columnas presentes en train Y test (las exporta 11_exportar_python.R).
Los códigos siguen el diccionario del DANE (MESE):
  P6050 parentesco (1 jefe, 2 cónyuge); P6100 régimen (1 contributivo, 2 especial,
  3 subsidiado); P6240 actividad principal (1 trabaja, 2 busca, 3 estudia,
  4 oficios del hogar, 5 incapacitado, 6 otra); P6426 meses en el empleo; P6430
  posición (1 privado, 2 gobierno, 3 doméstico, 4 cuenta propia, 5 patrón, 6-7 sin
  remuneración, 8 jornalero); P6870 tamaño de empresa (1 solo ... 9 = 101 o más);
  P6920 pensión (1 cotiza, 3 pensionado); P7505 recibió dinero de otros hogares o
  instituciones; P7500s3 pensión alimenticia; P7510s7 otros ingresos; P7472 ingreso
  laboral de desocupados/inactivos; P6510 horas extra; P7110 quiere más horas;
  P7150 busca cambiar de trabajo; P5100 cuota de amortización.
"""

import numpy as np
import pandas as pd

# Variables del nivel 3 (definir_niveles() en scripts/funciones/limpieza.R)
VARIABLES_NIVEL_3 = [
    "n_personas", "jefe_educ", "jefe_ocupado", "jefe_posicion",
    "n_menores", "n_menores_5", "n_adultos_mayores", "n_ocupados", "n_desocupados",
    "prop_ocupados", "tasa_dependencia", "prop_mujeres",
    "educ_maxima_adultos", "educ_promedio_adultos", "jefe_anos_educ",
    "jefe_mujer", "jefe_edad", "jefe_oficio", "jefe_subsidiado", "jefe_cotiza",
    "prop_afiliados_salud", "prop_subsidiado", "prop_cotiza_pension",
    "prop_informales", "prop_cuenta_propia", "prop_empresa_pequena",
    "horas_por_miembro", "hay_subempleo", "hay_segundo_trabajo",
    "recibe_subsidio_laboral", "recibe_prestaciones", "recibe_especie",
    "recibe_arriendos", "recibe_pension", "recibe_remesas",
    "recibe_ayudas_instituciones", "recibe_rentas_capital",
    "clase", "dominio", "departamento", "cuartos", "dormitorios", "tenencia",
    "log_arriendo", "log_arriendo_relativo", "personas_por_cuarto",
    "personas_por_dormitorio", "linea_pobreza",
]

VARIABLES_FEX = ["Fex_c", "Fex_dpto"]


def es_si(x):
    """1 si la respuesta es 1 (sí), 0 en otro caso (incluye NA = no aplica)."""
    return (x == 1).astype(np.int8)


def preparar_personas(personas):
    """Columnas auxiliares a nivel persona que usan varios grupos.

    Argumentos: personas = salida de cargar_personas().
    Devuelve: copia de personas con indicadores y horas totales.
    """
    p = personas.copy()
    ocupado = p["Oc"] == 1
    p["horas_totales"] = np.where(ocupado, p["P6800"].fillna(0) + p["P7045"].fillna(0), 0.0)
    p["edad_6_17"] = ((p["P6040"] >= 6) & (p["P6040"] < 18)).astype(np.int8)
    p["edad_18_24"] = ((p["P6040"] >= 18) & (p["P6040"] < 25)).astype(np.int8)
    p["edad_25_59"] = ((p["P6040"] >= 25) & (p["P6040"] < 60)).astype(np.int8)
    p["edad_60_mas"] = (p["P6040"] >= 60).astype(np.int8)
    p["edad_trabajar_18_64"] = ((p["P6040"] >= 18) & (p["P6040"] < 65)).astype(np.int8)
    p["pensionado"] = (p["P6920"] == 3).astype(np.int8)
    return p


def construir_laborales(p, hogar):
    """Grupo 1: actividad, estabilidad laboral, formalidad, tamaño de empresa,
    posición de todos los ocupados, horas, ingresos sí/no y costo de vivienda.

    Argumentos: p = salida de preparar_personas(); hogar = base de hogar.
    Devuelve: DataFrame indexado por id.
    """
    ocupado = p["Oc"] == 1
    actividad = p["P6240"]
    columnas = pd.DataFrame({
        "id": p["id"],
        "n_estudian": (actividad == 3).astype(np.int8),
        "n_oficios_hogar": (actividad == 4).astype(np.int8),
        "n_incapacitados": (actividad == 5).astype(np.int8),
        "n_buscan_trabajo": (actividad == 2).astype(np.int8),
        "n_pet": es_si(p["Pet"]),
        "n_inactivos": es_si(p["Ina"]),
        "n_contributivo": (p["P6100"] == 1).astype(np.int8),
        "n_regimen_especial": (p["P6100"] == 2).astype(np.int8),
        "n_cotizan": (p["P6920"] == 1).astype(np.int8),
        "n_pensionados": p["pensionado"],
        "n_empresa_grande": (ocupado & (p["P6870"] == 9)).astype(np.int8),
        "n_empresa_mediana": (ocupado & p["P6870"].between(4, 8)).astype(np.int8),
        "n_asalariado_privado": (ocupado & (p["P6430"] == 1)).astype(np.int8),
        "n_gobierno": (ocupado & (p["P6430"] == 2)).astype(np.int8),
        "n_domestico": (ocupado & (p["P6430"] == 3)).astype(np.int8),
        "n_patron": (ocupado & (p["P6430"] == 5)).astype(np.int8),
        "n_sin_remuneracion": (ocupado & p["P6430"].isin([6, 7])).astype(np.int8),
        "n_jornalero": (ocupado & (p["P6430"] == 8)).astype(np.int8),
        "n_dinero_otros_hogares": es_si(p["P7505"]),
        "n_pension_alimenticia": es_si(p["P7500s3"]),
        "n_otros_ingresos": es_si(p["P7510s7"]),
        "n_ingreso_laboral_inactivo": es_si(p["P7472"]),
        "n_horas_extra": es_si(p["P6510"]),
        "n_quiere_mas_horas": es_si(p["P7110"]),
        "n_busca_cambiar": es_si(p["P7150"]),
        "horas_totales": p["horas_totales"],
        "horas_jefe": np.where(p["P6050"] == 1, p["horas_totales"], 0.0),
        "horas_conyuge": np.where(p["P6050"] == 2, p["horas_totales"], 0.0),
        "edad_6_17": p["edad_6_17"], "edad_18_24": p["edad_18_24"],
        "edad_25_59": p["edad_25_59"], "edad_60_mas": p["edad_60_mas"],
        "meses_empleo": np.where(ocupado, p["P6426"], np.nan),
        "tamano_empresa": np.where(ocupado, p["P6870"], np.nan),
        "meses_empleo_jefe": np.where(ocupado & (p["P6050"] == 1), p["P6426"], np.nan),
        "edad": p["P6040"],
    })
    agregado = columnas.groupby("id").agg(
        **{c: (c, "sum") for c in columnas.columns
           if c.startswith("n_") or c.startswith("horas_") or c.startswith("edad_")},
        meses_empleo_max=("meses_empleo", "max"),
        meses_empleo_promedio=("meses_empleo", "mean"),
        meses_empleo_jefe=("meses_empleo_jefe", "max"),
        tamano_empresa_max=("tamano_empresa", "max"),
        tamano_empresa_promedio=("tamano_empresa", "mean"),
        edad_promedio=("edad", "mean"),
    )
    agregado = agregado.reindex(hogar.index)

    n_ug = hogar["personas_ug"].clip(lower=1)
    lp = hogar["linea_pobreza"]
    agregado["ocupados_por_pet"] = hogar["n_ocupados"] / agregado["n_pet"].clip(lower=1)
    agregado["horas_por_pet"] = agregado["horas_totales"] / agregado["n_pet"].clip(lower=1)
    agregado["horas_por_persona_ug"] = agregado["horas_totales"] / n_ug
    agregado["prop_contributivo"] = agregado["n_contributivo"] / hogar["n_personas"]
    agregado["prop_cotizan_ocupados"] = agregado["n_cotizan"] / hogar["n_ocupados"].clip(lower=1)
    agregado["prop_estudian"] = agregado["n_estudian"] / hogar["n_personas"]
    agregado["prop_60_mas"] = agregado["edad_60_mas"] / hogar["n_personas"]

    # Vivienda: amortización (P5100) y arriendo frente a la línea de pobreza
    cuota = hogar["P5100"].where(hogar["P5100"] >= 1000)
    costo_vivienda = hogar["arriendo"].fillna(0) + cuota.fillna(0)
    agregado["cuota_amortizacion_relativa"] = cuota / (lp * n_ug)
    agregado["arriendo_por_persona_lp"] = hogar["arriendo"] / (lp * n_ug)
    agregado["log_costo_vivienda_relativo"] = np.log(
        costo_vivienda.where(costo_vivienda > 0) / (lp * n_ug))
    agregado["cuartos_por_persona"] = hogar["cuartos"] / n_ug
    agregado["dormitorios_por_persona"] = hogar["dormitorios"] / n_ug
    return agregado


def construir_perceptor(p, hogar):
    """Grupo 2: perceptor principal = el ocupado con más horas (desempate: más
    años de educación y luego más edad). Hogares sin ocupados quedan con la
    categoría 'sin_perceptor' y 0 horas.

    Argumentos: p = salida de preparar_personas(); hogar = base de hogar.
    Devuelve: DataFrame indexado por id.
    """
    ocupados = p.loc[p["Oc"] == 1].sort_values(
        ["id", "horas_totales", "anos_educ", "P6040"], ascending=[True, False, False, False])
    perceptor = ocupados.drop_duplicates("id").set_index("id")
    salida = pd.DataFrame({
        "perceptor_edad": perceptor["P6040"],
        "perceptor_anos_educ": perceptor["anos_educ"],
        "perceptor_horas": perceptor["horas_totales"],
        "perceptor_cotiza": (perceptor["P6920"] == 1).astype(np.int8),
        "perceptor_contributivo": (perceptor["P6100"] == 1).astype(np.int8),
        "perceptor_tamano_empresa": perceptor["P6870"],
        "perceptor_meses_empleo": perceptor["P6426"],
        "perceptor_mujer": (perceptor["P6020"] == 2).astype(np.int8),
        "perceptor_es_jefe": (perceptor["P6050"] == 1).astype(np.int8),
        "perceptor_posicion": perceptor["P6430"].map(lambda x: f"p{int(x)}" if pd.notna(x) else "sin_dato"),
        "perceptor_oficio": perceptor["Oficio"].map(lambda x: f"o{int(x):02d}" if pd.notna(x) else "sin_dato"),
    }).reindex(hogar.index)
    salida["perceptor_horas"] = salida["perceptor_horas"].fillna(0)
    for columna in ("perceptor_posicion", "perceptor_oficio"):
        salida[columna] = salida[columna].fillna("sin_perceptor").astype("category")
    return salida


def construir_pequenos(p, hogar):
    """Grupo 3: tipologías de hogares pequeños, donde se concentran los FN.

    Argumentos: p = salida de preparar_personas(); hogar = base de hogar.
    Devuelve: DataFrame indexado por id.
    """
    sin_pension = (p["edad_60_mas"] == 1) & (p["recibe_pension"] == 0) & (p["pensionado"] == 0)
    agregado = pd.DataFrame({
        "id": p["id"],
        "edad_60_mas": p["edad_60_mas"],
        "am_sin_pension": sin_pension.astype(np.int8),
        "edad_trabajar_18_64": p["edad_trabajar_18_64"],
    }).groupby("id").sum().reindex(hogar.index)

    salida = pd.DataFrame(index=hogar.index)
    salida["hogar_unipersonal"] = (hogar["n_personas"] == 1).astype(np.int8)
    salida["solo_adultos_mayores"] = (agregado["edad_60_mas"] == hogar["n_miembros"]).astype(np.int8)
    salida["n_am_sin_pension"] = agregado["am_sin_pension"]
    salida["hay_am_sin_pension"] = (agregado["am_sin_pension"] > 0).astype(np.int8)
    salida["un_adulto_con_menores"] = (
        (agregado["edad_trabajar_18_64"] == 1) & (hogar["n_menores"] > 0)).astype(np.int8)
    salida["unipersonal_sin_ocupacion"] = (
        (salida["hogar_unipersonal"] == 1) & (hogar["n_ocupados"] == 0)).astype(np.int8)
    return salida


def construir_grupos(personas, hogar):
    """Arma todos los grupos de variables v2 que no dependen de modelos auxiliares.

    Argumentos: personas = cargar_personas(); hogar = cargar_hogares().
    Devuelve: diccionario {nombre_grupo: DataFrame indexado por id}.
    """
    p = preparar_personas(personas)
    return {
        "nivel_3": hogar[VARIABLES_NIVEL_3],
        "laborales": construir_laborales(p, hogar),
        "perceptor": construir_perceptor(p, hogar),
        "pequenos": construir_pequenos(p, hogar),
        "fex": hogar[VARIABLES_FEX],
    }
