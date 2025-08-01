import streamlit as st
import gdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import random
import numpy as np
import plotly.express as px
import pycountry
import geopandas as gpd
import plotly.graph_objects as go
import statsmodels.api as sm
import itertools
import time
import warnings
import logging

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
# Opcional: desactiva warnings excesivos
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TP Final - Ciencia de Datos",
    layout="wide"
)

st.title("🌍 TP Final - Ciencia de Datos")
st.subheader("Análisis interactivo de emisiones de gases de efecto invernadero")
st.caption("Dataset FAOSTAT (1961–2021) • Visualización dinámica • Comparación por país y fuente de emisión • Predicción a futuro")


def download_and_extract_zip_from_google_drive(file_id, output_dir, zip_name):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(zip_path)

csv_path = "data/raw/excercise/Emisiones_Totales_S_Todos_los_Datos_(Normalizado).csv"
if not os.path.exists(csv_path):
    st.info("Descargando dataset desde Google Drive...")
    download_and_extract_zip_from_google_drive(
        file_id="1hDcwMcaajRkoaO8trMqqc8DWObGkZl8v",
        output_dir="data/raw/excercise",
        zip_name="Emisiones_Totales_S_Todos_los_Datos_(Normalizado).zip"
    )

df = pd.read_csv(csv_path)
df = df[df['Valor'].notna()]
df['Valor_Mt'] = df['Valor'] / 1e3           # megatoneladas
df['Valor_Gt'] = df['Valor'] / 1e6           # gigatoneladas
df['Año'] = pd.to_datetime(df['Año'], format='%Y')

# KPIs
# Cálculos
total_emisiones = df['Valor_Mt'].sum()
anio_min = df['Año'].dt.year.min()
anio_max = df['Año'].dt.year.max()
total_paises = df['Área'].nunique()

# Diseño con columnas
col1, col2, col3 = st.columns(3)

col1.metric("🌍 Total emisiones (Mt)", f"{total_emisiones:,.2f}")
col2.metric("📅 Años cubiertos", f"{anio_min} – {anio_max}")
col3.metric("🗺️ Países únicos", total_paises)


# Mostrar rango temporal
anio_min = df['Año'].min().strftime('%Y')
anio_max = df['Año'].max().strftime('%Y')

st.subheader(f'📆 Rango temporal de los datos: {anio_min}  a  {anio_max}')


# Evolución por año
df_anual = df.groupby(df['Año'].dt.year).size().reset_index(name='Cantidad')
fig_anual = px.bar(df_anual, x='Año', y='Cantidad', title="Cantidad de registros por año")
st.plotly_chart(fig_anual, use_container_width=True)

df_cleaned = df.drop(columns=['Nota','Código del área', 'Código del área (M49)', 'Código del elemento', 'Código del año', 'Código fuente']).copy()


regiones = [
    'Mundo', 'Países Anexo I', 'Países No-Anexo I', 'Unión Europea (27)', 'Unión Europea (28)',
    'África', 'Américas', 'Europa', 'Asia', 'Oceanía',
    'África occidental', 'África central', 'África oriental', 'África meridional', 'África septentrional',
    'América central', 'América del Sur', 'América septentrional',
    'Europa occidental', 'Europa oriental', 'Europa septentrional', 'Europa meridional',
    'Asia central', 'Asia oriental', 'Asia occidental', 'Asia sudoriental', 'Asia meridional',
    'Australia y Nueva Zelandia', 'El Caribe', 'Melanesia', 'Polinesia', 'Micronesia',
    'OECD', 'URSS', 'Checoslovaq', 'Yugoslav RFS',
    'Los países menos desarrollados', 'Países sin litoral en vías de desarrollo',
    'Países de bajos ingresos y con déficit de alim.', 'Pequeñas islas en vías de Desarrollo',
    'Import netos alim en Des', 'Territorio de las Islas del Pacífico', 'China, Continental'
    ]
df_regiones = df_cleaned[df_cleaned['Área'].isin(regiones)].copy()
df_countries = df_cleaned[~df_cleaned['Área'].isin(regiones)].copy()
st.subheader("Cantidad de países: " + str(df_countries['Área'].nunique()))
# Total original
total_original = len(df_cleaned)

# Total después de separar
total_regiones = len(df_regiones)
total_paises = len(df_countries)

# Verificar
col1, col2 = st.columns(2)

with col1:
    st.metric("🧮 Total original", total_original)
    st.metric("🌍 Total regiones", total_regiones)

with col2:
    st.metric("🌎 Total países", total_paises)
    st.metric("➕ Suma regiones + países", total_regiones + total_paises)


countries = df_countries['Área'].unique()
fixes = {
    'Anguila': 'Anguilla',
    'Bahrein': 'Bahrain',
    'Bermudas': 'Bermuda',
    'Bhután': 'Bhutan',
    'Bélgica': 'Belgium',
    'Cabo Verde': 'Cape Verde',
    'Chequia': 'Czechia',
    'Chipre': 'Cyprus',
    'Comoras': 'Comoros',
    'Emiratos Árabes Unidos': 'United Arab Emirates',
    'Gambia': 'The Gambia',
    'Granada': 'Grenada',
    'Guadalupe': 'Guadeloupe',
    'Guayana Francesa': 'French Guiana',
    'Isla Norfolk': 'Norfolk Island',
    'Isla de Man': 'Isle of Man',
    'Islas Anglonormandas': 'Channel Islands',
    'Islas Caimán': 'Cayman Islands',
    'Islas Cook': 'Cook Islands',
    'Islas Feroe': 'Faroe Islands',
    'Islas Marianas del Norte': 'Northern Mariana Islands',
    'Islas Marshall': 'Marshall Islands',
    'Islas Salomón': 'Solomon Islands',
    'Islas Svalbard y Jan Mayen': 'Svalbard and Jan Mayen',
    'Islas Turcas y Caicos': 'Turks and Caicos Islands',
    'Islas Vírgenes Británicas': 'British Virgin Islands',
    'Islas Vírgenes de los Estados Unidos': 'United States Virgin Islands',
    'Islas Wallis y Futuna': 'Wallis and Futuna',
    'Martinica': 'Martinique',
    'Mónaco': 'Monaco',
    'Nueva Caledonia': 'New Caledonia',
    'Palestina': 'Palestine',
    'Países Bajos (Reino de los)': 'Netherlands',
    'Polinesia Francesa': 'French Polynesia',
    'República Democrática del Congo': 'Democratic Republic of the Congo',
    'República Democrática Popular Lao': "Lao People's Democratic Republic",
    'República Popular Democrática de Corea': "Democratic People's Republic of Korea",
    'Reunión': 'Réunion',
    'Saint Kitts y Nevis': 'Saint Kitts and Nevis',
    'Samoa Americana': 'American Samoa',
    'San Pedro y Miquelón': 'Saint Pierre and Miquelon',
    'Santa Elena, Ascensión y Tristán de Acuña': 'Saint Helena, Ascension and Tristan da Cunha',
    'Santa Sede': 'Holy See',
    'Sierra Leona': 'Sierra Leone',
    'Sudán (ex)': 'Sudan',
    'Sáhara Occidental': 'Western Sahara',
    'Timor-Leste': 'Timor-Leste',
    'Trinidad y Tabago': 'Trinidad and Tobago',
    'Yugoslav RFS': 'Yugoslavia',
    "Afganistán": "Afghanistan",
    "Albania": "Albania",
    "Alemania": "Germany",
    "Angola": "Angola",
    "Antigua y Barbuda": "Antigua and Barbuda",
    "Arabia Saudita": "Saudi Arabia",
    "Argelia": "Algeria",
    "Argentina": "Argentina",
    "Armenia": "Armenia",
    "Austria": "Austria",
    "Azerbaiyán": "Azerbaijan",
    "Bangladés": "Bangladesh",
    "Baréin": "Bahrain",
    "Belice": "Belize",
    "Belarús": "Belarus",
    "Bolivia (Estado Plurinacional de)": "Bolivia",
    "Bosnia y Herzegovina": "Bosnia and Herzegovina",
    "Botsuana": "Botswana",
    "Brasil": "Brazil",
    "Brunéi Darussalam": "Brunei",
    "Bulgaria": "Bulgaria",
    "Burkina Faso": "Burkina Faso",
    "Camboya": "Cambodia",
    "Camerún": "Cameroon",
    "Canadá": "Canada",
    "Chad": "Chad",
    "Chile": "Chile",
    "China": "China",
    "Colombia": "Colombia",
    "Corea, República de": "South Korea",
    "Costa de Marfil": "Ivory Coast",
    "Croacia": "Croatia",
    "Cuba": "Cuba",
    "Dinamarca": "Denmark",
    "Ecuador": "Ecuador",
    "Egipto": "Egypt",
    "El Salvador": "El Salvador",
    "Eritrea": "Eritrea",
    "Eslovaquia": "Slovakia",
    "Eslovenia": "Slovenia",
    "España": "Spain",
    "Estados Unidos de América": "United States of America",
    "Estonia": "Estonia",
    "Esuatini": "Eswatini",
    "Etiopía": "Ethiopia",
    "Filipinas": "Philippines",
    "Finlandia": "Finland",
    "Francia": "France",
    "Gabón": "Gabon",
    "Georgia": "Georgia",
    "Ghana": "Ghana",
    "Grecia": "Greece",
    "Groenlandia": "Greenland",
    "Guatemala": "Guatemala",
    "Guinea": "Guinea",
    "Guinea-Bissau": "Guinea-Bissau",
    "Guinea Ecuatorial": "Equatorial Guinea",
    "Haití": "Haiti",
    "Honduras": "Honduras",
    "Hungría": "Hungary",
    "India": "India",
    "Indonesia": "Indonesia",
    "Irak": "Iraq",
    "Irán (República Islámica del)": "Iran",
    "Irlanda": "Ireland",
    "Islandia": "Iceland",
    "Israel": "Israel",
    "Italia": "Italy",
    "Japón": "Japan",
    "Jordania": "Jordan",
    "Kazajstán": "Kazakhstan",
    "Kenia": "Kenya",
    "Kirguistán": "Kyrgyzstan",
    "Líbano": "Lebanon",
    "Liberia": "Liberia",
    "Libia": "Libya",
    "Lesoto": "Lesotho",
    "Letonia": "Latvia",
    "Lituania": "Lithuania",
    "Luxemburgo": "Luxembourg",
    "Madagascar": "Madagascar",
    "Malasia": "Malaysia",
    "Malawi": "Malawi",
    "Maldivas": "Maldives",
    "Malí": "Mali",
    "Malta": "Malta",
    "Marruecos": "Morocco",
    "Mauricio": "Mauritius",
    "Mauritania": "Mauritania",
    "México": "Mexico",
    "Micronesia (Estados Federados de)": "Federated States of Micronesia",
    "Mongolia": "Mongolia",
    "Montenegro": "Montenegro",
    "Mozambique": "Mozambique",
    "Namibia": "Namibia",
    "Nepal": "Nepal",
    "Nicaragua": "Nicaragua",
    "Níger": "Niger",
    "Nigeria": "Nigeria",
    "Noruega": "Norway",
    "Nueva Zelandia": "New Zealand",
    "Omán": "Oman",
    "Pakistán": "Pakistan",
    "Panamá": "Panama",
    "Papua Nueva Guinea": "Papua New Guinea",
    "Paraguay": "Paraguay",
    "Perú": "Peru",
    "Polonia": "Poland",
    "Portugal": "Portugal",
    "Qatar": "Qatar",
    "Reino Unido de Gran Bretaña e Irlanda del Norte": "United Kingdom",
    "República Árabe Siria": "Syria",
    "República Centroafricana": "Central African Republic",
    "República Checa": "Czech Republic",
    "República de Corea": "South Korea",
    "República de Moldova": "Moldova",
    "República Dominicana": "Dominican Republic",
    'República Unida de Tanzanía': 'Tanzania',
    "RDP Lao": "Laos",
    "Rumanía": "Romania",
    "Federación de Rusia": "Russian Federation",
    "San Cristóbal y Nieves": "Saint Kitts and Nevis",
    "Santa Lucía": "Saint Lucia",
    "San Vicente y las Granadinas": "Saint Vincent and the Grenadines",
    "Santo Tomé y Príncipe": "Sao Tome and Principe",
    "Samoa": "Samoa",
    "Senegal": "Senegal",
    "Serbia": "Serbia",
    "Seychelles": "Seychelles",
    "Singapur": "Singapore",
    "Sri Lanka": "Sri Lanka",
    "Sudáfrica": "South Africa",
    "Sudán": "Sudan",
    "Sudán del Sur": "South Sudan",
    "Suecia": "Sweden",
    "Suiza": "Switzerland",
    "Suriname": "Suriname",
    "Tailandia": "Thailand",
    "Tanzania, República Unida de": "Tanzania",
    "Tayikistán": "Tajikistan",
    "Timor-Leste": "East Timor",
    "Tonga": "Tonga",
    "Trinidad y Tobago": "Trinidad and Tobago",
    "Túnez": "Tunisia",
    "Turkmenistán": "Turkmenistan",
    "Turquía": "Türkiye",
    "Ucrania": "Ukraine",
    "Uruguay": "Uruguay",
    "Uzbekistán": "Uzbekistan",
    "Venezuela (República Bolivariana de)": "Venezuela",
    "Viet Nam": "Vietnam",
    "Yemen": "Yemen",
    "Zambia": "Zambia",
    "Zimbabue": "Zimbabwe",
    "Estado de Palestina": "Palestine",
    "Macedonia del Norte": "North Macedonia",
    'República del Congo': 'Congo',
}

# Aplicar correcciones
fixed_countries = [fixes.get(p, p) for p in countries]

# Obtener código ISO-3 para cada país
def get_iso3(name):
    '''
    Esta función busca cada nombre en la base de datos oficial de pycountry:

    Si encuentra el país  devuelve el código ISO-3 (ARG para Argentina, BRA para Brasil, etc.).

    Si falla (nombre raro, no existe)  devuelve None
    '''
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

iso_codes = [get_iso3(p) for p in fixed_countries] #  obtiene una lista de códigos ISO-3 o None si falló
df_map = pd.DataFrame({'country': countries, 'iso_alpha': iso_codes}) # Crea un nuevo DataFrame con dos columnas:'country' → nombre original (sin corrección), 'iso_alpha' → código ISO-3 resultante (de la versión corregida)
df_map = df_map.dropna()  # Elimina las filas donde iso_alpha es None.

# Añadir una columna "coverage" para indicar cobertura
df_map['coverage'] = 1  # 1 = incluido en el dataset

st.subheader("Cobertura geográfica del dataset FAOSTAT")
fig = px.choropleth(
    df_map,
    locations='iso_alpha',
    color='coverage',
    hover_name='country',
    color_continuous_scale=[[0, 'lightgrey'], [1, 'green']],
    title='Cobertura FAOSTAT'
)
fig.update_geos(bgcolor='lightblue',showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
fig.update_layout(coloraxis_showscale=False, margin={"r":10,"t":40,"l":10,"b":10})
st.plotly_chart(fig, use_container_width=True)



st.subheader("Registros por país por año:")
st.markdown("Expansión en la cobertura de países a partir de 1990*  \n"
            "A partir del año 1990 se observa un incremento significativo en la cantidad de países con datos disponibles. Este cambio no necesariamente implica un aumento real en las emisiones,"
            " sino una mejora en la cobertura geográfica del dataset.  \n"
            "En total, se incorporan 52 nuevos países/regiones después de 1990, lo que puede influir en los análisis agregados si no se controla adecuadamente.*  \n"
            "Para evitar conclusiones erróneas, este notebook incluye filtros y comparaciones que tienen en cuenta este cambio estructural en la base de datos.*")

#Agrupar por país y año
df_anual = df_countries.groupby(['Área', 'Año']).size().reset_index(name='records')
df_group = df_countries.groupby('Área').agg({'Valor_Mt': 'sum'}).reset_index()

#Aplicar equivalencias de nombres
df_anual['country'] = df_anual['Área'].replace(fixes)
df_anual['iso_alpha'] = df_anual['country'].apply(get_iso3)
df_anual = df_anual[df_anual['iso_alpha'].notnull()]


#Crear mapa animado
fig = px.choropleth(
    df_anual,
    locations='iso_alpha',
    color='records',
    hover_name='country',
    color_continuous_scale='Viridis',
    animation_frame='Año',
    title='Evolución anual de registros por país'
)

fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)


# Selector de país
st.subheader("Evolución temporal por país")
pais_sel = st.selectbox("Seleccionar país", sorted(df['Área'].unique()))
df_pais = df[df['Área'] == pais_sel]
df_serie = df_pais.groupby(df_pais['Año'].dt.year)['Valor_Mt'].sum().reset_index()
fig_pais = px.line(df_serie, x='Año', y='Valor_Mt', title=f"Evolución de emisiones en {pais_sel}")
st.plotly_chart(fig_pais, use_container_width=True)

st.markdown("""## Expansión en la cobertura de países a partir de 1990
A partir del año 1990 se observa un incremento significativo en la cantidad de países con datos disponibles. Este cambio no necesariamente implica un aumento real en las emisiones, sino una mejora en la cobertura geográfica del dataset.

En total, se incorporan 52 nuevos países/regiones después de 1990, lo que puede influir en los análisis agregados si no se controla adecuadamente.

Para evitar conclusiones erróneas, este notebook incluye filtros y comparaciones que tienen en cuenta este cambio estructural en la base de datos.""")


st.subheader("Nuevas áreas registradas después de 1990")
# Conjuntos de países por período
df_cleaned['Año'] = df_cleaned['Año'].dt.year
areas_before_1990 = set(df_cleaned[df_cleaned['Año'] < 1990]['Área'].unique())
areas_after_1990 = set(df_cleaned[df_cleaned['Año'] > 1990]['Área'].unique())
# Nuevas áreas que no estaban antes
new_areas = sorted(list(areas_after_1990 - areas_before_1990))
st.write(f"Cantidad de nuevas áreas: **{len(new_areas)}**")
for area in new_areas:
    st.markdown(f"- {area}")


# Top países
st.subheader("Top 10 países con más emisiones")
df_top = df_group.sort_values("Valor_Mt", ascending=False).head(10)
fig_top = px.bar(df_top, x='Área', y='Valor_Mt', title="Top 10 países con más emisiones")
st.plotly_chart(fig_top, use_container_width=True)

st.subheader("Consideraciones sobre Productos Reportados")
st.markdown("El número de productos reportados cambia significativamente con el tiempo:  \n"
            "- **Antes de 1990**: solo 16 productos reportados.  \n"
            "- **Después de 1990**: más de 40 productos.  \n"
            "Este cambio refleja una expansión en el nivel de detalle del inventario de emisiones, tanto en cobertura temática como en precisión metodológica. "
            "Sin embargo, también introduce un **sesgo estructural** en los análisis temporales agregados.")

products_before_1990 = set(df_cleaned[df_cleaned['Año'] < 1990]['Producto'].unique())
products_after_1990 = set(df_cleaned[df_cleaned['Año'] >= 1990]['Producto'].unique())

products = products_before_1990 & products_after_1990
new_products = products_after_1990 - products_before_1990

st.subheader("Comparación de productos por período")

st.write(f"📦 Productos antes de 1990: {len(products_before_1990)}")
st.write(f"📦 Productos después de 1990: {len(products_after_1990)}")
st.write(f"🔁 Productos comunes: {len(products)}")
st.write(f"🆕 Productos nuevos desde 1990: {len(new_products)}")

st.subheader("""Además, el dataset incluye algunos productos agregados. Es decir, productos que incluyen dentro a otros productos. Es importante saber diferenciarlos para que las comparaciones tengan sentido.""")
codes_agg = [6518, 6516, 6517, 6996, 6995, 5084, 5085,
             6825, 6829, 6824, 67292, 67291, 69921, 6821, 6817, 6820, 1707, 1711]
aggregated_products = df[df['Código del producto'].isin(codes_agg)]['Producto'].unique()
st.subheader("Productos agregados")
for producto in aggregated_products:
    st.markdown(f"- {producto}")


st.markdown("""## Delimitación temporal del análisis

Debido a los cambios estructurales observados en la cobertura geográfica y temática del dataset, se ha decidido restringir el análisis a los datos disponibles **a partir del año 1990**.

Esta decisión responde a dos razones principales:

- **Mayor cobertura geográfica**: a partir de 1990 se incorporan 52 nuevos países, alcanzando un total de 238. Esto garantiza que los análisis comparativos entre regiones y países no estén sesgados por datos ausentes en décadas anteriores.
  
- **Mayor cobertura temática**: el número de productos reportados aumenta de 16 (antes de 1990) a más de 40 (después), lo que introduce una mejora en el detalle metodológico, pero también limita la comparabilidad histórica.

### Justificación

Trabajar con el subconjunto de datos posterior a 1990 permite realizar análisis **más consistentes, representativos y comparables** reduciendo el riesgo de conclusiones erróneas causadas por diferencias de cobertura y disponibilidad de información.

En consecuencia, **todas las visualizaciones y estadísticas agregadas en este informe se basarán en datos desde 1990 hasta la actualidad, por lo cual no vamos a tener en cuenta estimaciones futuras**."""
)

df_completed = df_cleaned.copy()
df_01 = df_cleaned[(df_cleaned['Año'] >= 1990) & (df_cleaned['Año'] <= 2025)].copy()


st.markdown("## Variables de Emisión  \n"
            "El conjunto de datos original incluye múltiples tipos de elementos relacionados con las emisiones de gases, entre ellos: ")

elementos = df_01['Elemento'].unique()
st.markdown("### Tipos de emisiones registradas:")
for elem in elementos:
    st.markdown(f"- {elem}")
st.subheader("")
st.markdown("## Comparación de Cobertura por Fuente (FAO vs UNFCCC)")


# Ver países únicos por fuente
fao_data = set(df_01[df_01['Fuente'] == 'FAO TIER 1']['Área'].unique())
unfccc_data = set(df_01[df_01['Fuente'] == 'UNFCCC']['Área'].unique())

st.subheader("Comparación de cobertura por fuente de datos")
st.write(f"🌾 Países y regiones con datos **FAO TIER 1**: {len(fao_data)}")
st.write(f"🌍 Países y regiones con datos **UNFCCC**: {len(unfccc_data)}")
st.write(f"✅ Países y regiones en **ambas fuentes**: {len(fao_data & unfccc_data)}")
st.write(f"🟢 Solo en **FAO TIER 1**: {len(fao_data - unfccc_data)}")
st.write(f"🔵 Solo en **UNFCCC**: {len(unfccc_data - fao_data)}")

st.markdown("""
### 📚 Sobre las fuentes de datos

El conjunto de datos incluye emisiones reportadas por dos fuentes distintas: **FAO TIER 1** y **UNFCCC**. Estas fuentes utilizan metodologías diferentes:

- **FAO TIER 1**: Estimaciones generadas por la FAO usando metodologías estandarizadas (*IPCC Tier 1*). Ofrece cobertura global y permite analizar series temporales largas de manera consistente, aunque con menor precisión país-específica.

- **UNFCCC**: Datos reportados directamente por los países miembros del Convenio Marco de las Naciones Unidas sobre el Cambio Climático. Son más precisos pero no están disponibles para todos los países ni todos los años.

Para garantizar la **consistencia del análisis exploratorio** y evitar duplicidades (múltiples registros para un mismo país, año y tipo de emisión), separamos los datos por fuente.  
En este análisis general utilizaremos principalmente los datos provenientes de **FAO TIER 1**, ya que brindan una cobertura más amplia y continua en el tiempo.

📌 En secciones posteriores, se podrá comparar con los datos de **UNFCCC** para identificar posibles diferencias metodológicas o validar tendencias observadas.
""")

df_fao = df_01[df_01['Fuente'] == 'FAO TIER 1'].copy()

st.markdown("""
## 🧾 Descripción de los indicadores

### 1. Emisiones directas (N₂O)
Emisiones de óxido nitroso (**N₂O**) que se liberan directamente al aire desde su fuente, por ejemplo:

- Aplicación de fertilizantes nitrogenados al suelo.

---

### 2. Emisiones indirectas (N₂O)
Emisiones de **N₂O** que ocurren después de procesos intermedios, como:

- Lixiviación de nitrógeno al agua.  
- Volatilización (evaporación y deposición posterior).

Estas emisiones ocurren fuera del punto de aplicación pero son atribuibles a prácticas agrícolas.

---

### 3. Emisiones (N₂O)
Suma total de **Emisiones directas + Emisiones indirectas** de N₂O.  
Representa la emisión completa de N₂O atribuible a la agricultura/ganadería.

---

### 4. Emisiones (CO₂eq) proveniente de N₂O (AR5)
Las emisiones de N₂O convertidas a **CO₂ equivalente** usando el factor del 5º Informe del IPCC (AR5):  

- N₂O tiene un GWP (Global Warming Potential) de **265**.  
- Ejemplo: 1 kt de N₂O → 265 kt de CO₂eq.

Esto permite comparar gases con diferente efecto climático.

---

### 5. Emisiones (CO₂eq) (AR5)
Este es el indicador total combinado, ya convertido a **CO₂eq (según AR5)**.  
Incluye:

- CO₂  
- CH₄ convertido a CO₂eq  
- N₂O convertido a CO₂eq  
- F-gases  

Es la métrica recomendada para **comparaciones globales** de impacto climático.

---

### 6. Emisiones (CH₄)
Emisiones directas de metano (**CH₄**), especialmente desde:

- Fermentación entérica en ganado.  
- Cultivo de arroz.

📏 Unidad: kilotoneladas de CH₄.

---

### 7. Emisiones (CO₂eq) proveniente de CH₄ (AR5)
CH₄ convertido a **CO₂eq** usando GWP del AR5:

- CH₄ tiene un GWP de **28**.  
- 1 kt de CH₄ → 28 kt de CO₂eq.

Permite estimar el efecto climático del metano en términos comparables.

---

### 8. Emisiones (CO₂)
Emisiones directas de dióxido de carbono (**CO₂**).  
Pueden provenir de maquinaria agrícola, quema de residuos, etc.

📏 Unidad: kilotoneladas de CO₂.

---

### 9. Emisiones (CO₂eq) proveniente de F-gases (AR5)
Gases fluorados (**HFCs, PFCs, SF₆**) convertidos a CO₂eq.  
Aunque no provienen típicamente de agricultura, pueden aparecer si se incluyen procesos industriales vinculados.

💥 Tienen altísimos GWP (hasta miles de veces el del CO₂).
""")


st.markdown("# Análisis Exploratorio de Datos (EDA)  \n")
with st.expander("🌍 Análisis Global y Comparativo por Continente de Emisiones Totales incluyendo LULUCF"):
    st.markdown("""
Para este análisis vamos a tomar en cuenta el indicador **CO₂eq (AR5)**, que es la suma estimada de los tres gases principales, ya convertidos por su impacto climático.

Además, vamos a utilizar el agregado **"Emisiones Totales incluyendo LULUCF"**. Este agregado es la suma de todas las fuentes de gases de efecto invernadero del sistema agroalimentario (farm gate + cambio de uso del suelo + procesos pre- y pos-producción) más el resto de sectores **IPCC**.

**¿Qué significa LULUCF?**  
Land Use, Land-Use Change and Forestry.

Es el sector que captura o emite **CO₂ (u otros gases)** cuando se utiliza la tierra (pasturas, cultivos), se cambia el uso de la tierra (deforestación, expansión urbana) y silvicultura (tala y repoblación forestal, incendios forestales).
""")

gas = 'Emisiones (CO2eq) (AR5)'
continents = [
    'Américas',
    'África',
    'Europa',
    'Asia',
    'Oceanía',
    'Mundo'
]
product_code = 6825 # Emisiones Totales incluyendo LULUCF

df_continents = df_fao[
    (df_fao['Área'].isin(continents)) &
    (df_fao['Elemento'] == gas) &
    (df_fao['Código del producto'] == product_code)
    ].copy()

df_emissions_by_continent_year = df_continents.groupby(['Área', 'Elemento', 'Año'])['Valor_Gt'].sum().reset_index()
df_emissions_by_continent_year.sort_values(by='Valor_Gt', ascending=False, inplace=True)

df_emissions_by_continent = df_continents.groupby(['Área'])['Valor_Gt'].sum().reset_index()
df_emissions_by_continent = df_emissions_by_continent.drop(df_emissions_by_continent[df_emissions_by_continent['Área'] == 'Mundo'].index)

st.subheader("Primeras filas del DataFrame de emisiones por continente")
st.dataframe(df_emissions_by_continent.head())



# Crear paleta
palette = sns.color_palette('Set2', len(continents))

# Copiar datos para graficar
df_plot = df_emissions_by_continent_year.copy()

# Estilo y figura
sns.set_style("whitegrid")
fig, (ax_line, ax_pie) = plt.subplots(1, 2, figsize=(14, 5),
                                      gridspec_kw={"width_ratios":[2, 1]})

# Gráfico de líneas por continente
for idx, cont in enumerate(continents):
    sub = df_plot[df_plot['Área'] == cont].sort_values('Año')
    ax_line.plot(sub['Año'], sub['Valor_Gt'], marker='o', linewidth=1.4,
                 label=cont, color=palette[idx])

ax_line.set_title('CO₂-eq (AR5) — serie temporal')
ax_line.set_ylabel('Gt CO₂-eq')
ax_line.set_xlabel('')
ax_line.grid(ls='--', alpha=.4)
ax_line.legend(title='Continente', ncol=3, fontsize=8)

# Gráfico de torta
ax_pie.pie(df_emissions_by_continent['Valor_Gt'],
           labels=df_emissions_by_continent['Área'],
           autopct='%1.1f%%',
           startangle=90,
           colors=palette)
ax_pie.set_title('Porcentaje de emisiones (CO₂eq AR5) por continente')
ax_pie.axis('equal')

# Título general
plt.suptitle('Emisiones Totales incluyendo LULUCF — CO₂-eq 1990 - 2022',
             fontsize=14, y=1.03)
plt.tight_layout()

# ✅ Mostrar en Streamlit
st.pyplot(fig)


st.markdown("""
### 🧠 Interpretación y conclusiones del gráfico

**Interpretación:**

En el gráfico de la izquierda, podemos observar cómo fueron evolucionando las emisiones totales de **CO₂eq (AR5)** en cada continente desde **1990 a 2022**.  
Cada línea representa las emisiones en gigatoneladas por continente.  
En el gráfico de torta, se visualiza el **aporte proporcional de cada continente** a la suma total de emisiones de CO₂eq (AR5) en el mismo período.

---

**Conclusiones:**

- **🌏 Asia**  
  - Aporta el **46%** del total mundial y continúa en crecimiento.  
  - Su curva crece de forma continua, pasando de **10 Gt en 1990 a 30 Gt en 2022**.

- **🌎 América**  
  - Aporta el **26%** del total mundial.  
  - Presenta un ligero crecimiento hasta 2005 (~12 Gt), luego una meseta, y a partir de 2010, un descenso.

- **🌍 Europa**  
  - Muestra una **disminución sostenida** desde 1990 hasta la actualidad.

- **🌍 África**  
  - Representa el **9%** del total de emisiones.  
  - Tiene un crecimiento sostenido de aproximadamente **3 Gt a 5 Gt** (un aumento del 60%).  
  - Sin embargo, la aceleración es mucho menor comparada con Asia.

- **🌏 Oceanía**  
  - Es el continente con **menos emisiones**.  
  - Su línea permanece prácticamente **plana** desde 1990 hasta hoy.

---

📌 **Resumen general:**  
**Asia y América representan el 72%** del total de emisiones desde 1990 a 2022.  
**Europa** mantiene una tendencia de **reducción constante** en sus emisiones.
""")


df_covid = df_fao[
    (df_fao['Área'].isin(continents)) &
    (df_fao['Año'].between(2017, 2022)) &
    (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_fao['Producto'].isin(['Emisiones totales incluyendo LULUCF',
                              'Pre y\xa0post-producción',
                              'Farm gate',
                              'Cambios de uso de la tierra']))
].copy()

df_pct = (
    df_covid
    .groupby(['Área','Producto','Año'])['Valor_Mt'].sum()
    .unstack('Año')
    .pipe(lambda d: d.pct_change(axis=1)*100)
    [[2019,2020, 2021,2022]]            # columnas de interés
    .reset_index()
)
# Título y descripción
st.subheader("Variación porcentual anual de emisiones CO₂eq (AR5)")
st.caption("Comparación año a año (2019-2022) por continente y tipo de producto.")

# Asegurarse que los nombres de columnas numéricas sean strings
df_pct.columns = [str(col.year) if isinstance(col, pd.Timestamp) else str(col) for col in df_pct.columns]

# Detectar columnas numéricas por su nombre (años en string)
cols_numericas = [col for col in df_pct.columns if col.isdigit()]

# Estilo zebra + formato numérico
styled_df = (
    df_pct.style
    .format({col: "{:.2f}" for col in cols_numericas})
    .set_properties(**{'background-color': '#f9f9f9'}, subset=pd.IndexSlice[::2, :])
)

st.dataframe(styled_df, use_container_width=True)

st.subheader("Distribución porcentual anual de emisiones por continente.")
world = df_emissions_by_continent_year[df_emissions_by_continent_year['Área'] == 'Mundo']
conts = df_emissions_by_continent_year[df_emissions_by_continent_year['Área'] != 'Mundo']
df_share = conts.merge(world, on='Año', suffixes=('_cont', '_world'))
df_share['share'] = df_share['Valor_Gt_cont'] / df_share['Valor_Gt_world'] * 100
pivot = (
    df_share.pivot(index='Año', columns='Área_cont', values='share')
            .loc[:, ['Asia','Américas','Europa','África','Oceanía']]
            .fillna(0)
)
# Estilo y paleta
sns.set_style('whitegrid')
palette = sns.color_palette('Set2', len(pivot.columns))

# Crear figura y eje
fig, ax = plt.subplots(figsize=(12, 6))

# Gráfico de barras apiladas
pivot.plot(kind='bar', stacked=True, color=palette, width=0.9, ax=ax)

# Títulos y formato
ax.set_title('Distribución porcentual de emisiones agroalimentarias por continente (1990-2025)')
ax.set_ylabel('% del total global')
ax.set_xlabel('Año')
ax.set_ylim(0, 100)
ax.legend(title='Continente', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig)

# Texto interpretativo
st.markdown("""
### Interpretación:
- Cada barra del gráfico representa el **100 % del total global de emisiones**. Los colores muestran el porcentaje que ocupa cada continente en ese año.

### Conclusiones:
- **Asia**: pasa de representar un 30 % en 1990 a aproximadamente un 55 % en la actualidad.
- **América**: mantiene aproximadamente un 30 % en toda la década del 90, luego cae al 23 % en 2010 y se estabiliza.
- **Europa**: pasa de 29 % en 1990 a menos del 15 % en 2022. La franja azul confirma la eficacia de sus políticas contra la emisión de gases de efecto invernadero.
- **África**: crece muy lentamente, del 8 % al 9 %.
- **Oceanía**: en 32 años (1990 - 2022) nunca superó el 2 %.

Como se puede observar en el gráfico, **el eje de las emisiones se desplazó del Atlántico (Europa - América) al Índico - Pacífico**.  
**Asia es hoy el principal emisor absoluto y relativo. Además, es el motor del crecimiento de las emisiones a nivel global.**
""")


st.subheader('Promedio Anual de Emisiones Totales por década y continente')
df_dec = df_emissions_by_continent_year.copy()

# Crear columna de década
df_dec['Década'] = (df_dec['Año'] // 10) * 10

# Excluir 'Mundo'
df_dec = df_dec[df_dec['Área'] != 'Mundo']

# Agrupar por década y área
pivot_dec = (
    df_dec.groupby(['Década', 'Área'])['Valor_Gt']
          .mean()
          .reset_index()
)

# --- Gráfico ---
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=pivot_dec,
    x='Década', y='Valor_Gt', hue='Área',
    palette='Set2',
    ax=ax
)

ax.set_title('Promedio anual de CO₂-eq por década y continente')
ax.set_ylabel('Gt CO₂-eq')
ax.set_xlabel('Década')
ax.legend(title='Continente', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

plt.tight_layout()
st.pyplot(fig)
st.markdown("""Interpretación:
- Década del 90: el mapa de las emisiones está muy distribuido, no se observa una dominancia marcada por algún continente.
- Década del 2000: Asia despega, aumenta el promedio anual en un 45% respecto a la década anterior (del 12 Gt a 18 Gt). Además, América aumenta un 10% y Europa tiene una caída del 15%. Comienza el cambio a nivel mundial del mapa de emisiones, las mismas se trasladan hacía el pacífico.
- Decada del 2010: Asia acelera nuevamente aumentando otro 45% (de 18gt a 26gt) el promedio anual en la década. En esta década Asia tiene un promedio mayor al doble que América. Europa continúa con su descenso y África aumenta su promedio en un 25% respecto a la década anterior.
- Década del 2020: el promedio anual de Asia se ubica cerca de las 30Gt, triplicando a América. Europa se estabiliza y Oceanía mantiene su promedio anual < 1 Gt, al igual que en décadas anteriores.""")

st.subheader("Comparativa de Emisiones por Continente y por componente (1990 - 2010 - 2022)")
st.markdown("""Para este análisis se seleccionaron los siguientes componentes:

- Farm Gate: fermentación entérica, gestión de estiércol, fertilizantes sintéticos, uso de energía en la finca, etc.
Es decir, es todo lo que ocurre dentro del establecimiento agropecuario.
- Cambios en el uso de la tierra: deforestación, conversión neta de bosques, drenaje de suelos orgánicos, incendios, etc.
- Pre y post-producción: procesado, envasado, transporte, venta y desperdicio de alimentos. Todo lo que sucede antes y después de la puerta de la finca

Estos componentes agrupados representan las Emisiones Totales incluyendo LULUCF. Al analizarlos por separado, podemos definir con precisión qué porción de las emisiones proviene de la finca, de la conversión de ecosistemas o de la cadena de suministro, lo cual es información importante para definir politicas eficaces en cada región.
""")

# --- Parámetros ---
continents = ['Américas', 'Asia', 'Europa', 'Oceanía', 'África']
products = ['Farm gate', 'Cambios de uso de la tierra', 'Pre y\xa0post-producción']
gas = "Emisiones (CO2eq) (AR5)"
years = [1990, 2010, 2022]

# --- Filtro del DataFrame ---
df_products_continents = df_fao[
    (df_fao['Producto'].isin(products)) &
    (df_fao['Área'].isin(continents)) &
    (df_fao['Año'].isin(years)) &
    (df_fao['Elemento'] == gas)
].copy()

# --- Pivot ---
pivot = (
    df_products_continents
    .pivot_table(index=['Año', 'Área'], columns='Producto', values='Valor_Gt', aggfunc='sum')
    .sort_index(level=1)
    .reset_index()
)

pivot = pivot.sort_values(['Área', 'Año'], ascending=[True, False]).reset_index(drop=True)

# --- Configuración del gráfico ---
colors = ['#0066CC', '#0eca1c', '#ff5733']
bar_h = 0.8
gap = 1
offset = bar_h
n_y = len(years)
y_pos = []

# --- Posiciones verticales para barras por continente ---
for g in range(len(continents)):
    base = offset + g * (n_y * bar_h + gap)
    y_pos.extend(base + np.arange(n_y) * bar_h)

fig, ax = plt.subplots(figsize=(10, 7))
left = np.zeros(len(y_pos))

# --- Barras horizontales apiladas ---
for (col, color) in zip(products, colors):
    ax.barh(y_pos, pivot[col], left=left,
            height=bar_h, color=color, edgecolor='white', label=col)
    left += pivot[col].values

# --- Ejes Y principales (años) ---
ax.set_yticks(y_pos)
ax.set_yticklabels(pivot['Año'])
ax.set_axisbelow(True)
ax.grid(axis='y')

# --- Eje Y secundario (continentes) ---
ax2 = ax.twinx()
mid_pos = [offset * 0.1 + g * (n_y * bar_h + gap) + (n_y * bar_h) - bar_h
           for g in range(len(continents))]
ax2.set_yticks(mid_pos)
ax2.set_yticklabels(continents, fontsize=15, weight='bold')
ax2.yaxis.set_ticks_position('left')
ax2.spines['left'].set_position(('outward', 70))
ax2.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(False)

# --- Títulos y leyenda ---
ax.set_xlabel('Gt CO₂-eq (AR5)')
ax.set_title('Total Emisiones CO₂-eq por componente 1990 · 2010 · 2022')
ax.legend(title='Producto',
          loc='upper right',
          frameon=True,
          framealpha=.9,
          borderpad=.6, fontsize=9)

plt.tight_layout()

# --- Mostrar en Streamlit ---
st.subheader("📊 Total Emisiones CO₂-eq por componente (1990, 2010, 2022)")
st.pyplot(fig)

st.markdown("""
---

### 🧾 Interpretación del gráfico

El gráfico muestra barras horizontales apiladas que cuantifican las emisiones agro-alimentarias de CO₂-equivalente (Gt CO₂-eq) para cada continente en tres cortes temporales: **1990**, **2010** y **2022**.

#### 🔍 Interpretación por continente:

- **Asia**:
  - En 1990, las emisiones a causa de *Farm gate* son la porción más grande.
  - Para 2022, el bloque rojo (*pre-/post-producción*) crece con rapidez (aumenta × 7 en 32 años), reflejando la industrialización de la cadena alimentaria asiática y el aumento del consumo urbano.
  - Las emisiones por *cambios en el uso de la tierra* se reducen moderadamente tras 2010 gracias al freno parcial de la deforestación en el Sudeste Asiático.

- **Américas**:
  - En 1990 se observa una dominancia de las emisiones por *cambios en el uso de la tierra*.
  - Para el año 2022, el bloque verde disminuye notablemente, mientras las emisiones por *pre y post-producción* se multiplican. Esto significa que hubo una transición de deforestación a cadena de suministro.
  - Al mismo tiempo, azul se mantiene estable y rojo se multiplica. Significa que la presión climática migra de la frontera agropecuaria hacia la logística y el consumo.

- **Europa**:
  - Desde 1990 a la actualidad, las emisiones por *cambios en el uso de la tierra* fueron marginales.
  - Se observa una disminución de las emisiones por *farm gate* desde 1990 a 2010 y luego se mantienen estables hasta 2022.
  - El bloque rojo demuestra que la mayor parte del problema europeo reside hoy en las emisiones por *pre y post-producción*.

- **África**:
  - Las emisiones por *farm gate* ganan peso década tras década.
  - En 1990, las emisiones por *cambios en el uso de la tierra* dominan con amplia ventaja.
  - En 2010, las emisiones por *farm gate* recortan distancia.
  - En 2022, el componente ligado a deforestación e incendios sigue siendo la principal fuente de emisiones. Además, la franja roja (*pre y post-producción*) muestra que la cadena de valor —procesado, transporte, venta— está comenzando a pesar y puede acelerarse.

- **Oceanía**: emisiones muy bajas y estables.

---

### 🧠 Conclusión

El problema climático del sistema agro‑alimentario mundial se ha desplazado del **“dónde sembramos”** (*deforestación*) al **“cómo producimos y consumimos”** (*industria y consumo*).

Las estrategias para reducir las emisiones de gases deben, por tanto, abarcar la **cadena completa**, con prioridades distintas según la fase en que se encuentre cada región.
""")

st.subheader("Top 10 Paises con mayores Emisiones (CO2eq) (AR5) (2022)")
st.markdown("""Para este análisis se seleccionó el elemento Emisiones (CO2eq) (AR5), ya que:
- Es una métrica que convierte todas las emisiones de gases de efecto invernadero (GEI) —como dióxido de carbono (CO₂), metano (CH₄) y óxido nitroso (N₂O)— en toneladas equivalentes de CO₂.
- Permite realizar análisis agregados, consistentes y comparables entre países

""")
gas = 'Emisiones (CO2eq) (AR5)'
product_code = 6825 # Emisiones Totales incluyendo LULUCF

df_countries_2022 = df_fao[
    (~df_fao['Área'].isin(regiones)) &
    (df_fao['Código del producto'] == product_code) &
    (df_fao['Elemento'] == gas) &
    (df_fao['Año'] == 2022)
    ]

df_top_countries_emission = df_countries_2022.groupby(['Área'])['Valor'].sum().reset_index()
df_top_countries_emission.sort_values(by= 'Valor',ascending=False, inplace=True)
df_top_countries_emission = df_top_countries_emission.head(10)
df_top_countries_emission

total_global = df_fao[
    (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_fao['Área'] == 'Mundo') &
     (df_fao['Año'] == 2022) &
    (df_fao['Código del producto'] == product_code)
    ]['Valor'].sum()
total_top_10 = df_top_countries_emission['Valor'].sum()

pct_top_10 = (total_top_10 / total_global) * 100
pct_rest = 100 - pct_top_10

st.markdown(f"## **Top 10 países** representan el **{pct_top_10:.1f}%** del total global.")
st.markdown(f"## **Resto del mundo** representa el **{pct_rest:.1f}%**.")

# Espaciado visual
st.write("")  # una línea en blanco
st.write("")  # otra si se quiere más espacio

fix = {
    "Estados Unidos de América": "EEUU",
    "Irán (República Islámica del)": 'Irán',
    "Federación de Rusia": "Rusia"
}
df_top_countries_emission['Área'] = df_top_countries_emission['Área'].replace(fix)

# Crear figura y subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6),gridspec_kw={'width_ratios': [1.2, 1]})

# --- Subplot 1: Gráfico de barras ---
sns.barplot(data=df_top_countries_emission, x='Área', y='Valor', hue='Área',
            palette='Greens_r', ax=axs[0], width=1.0)

# Quitar la leyenda manualmente (en lugar de usar legend=False que lanza error)
axs[0].legend_.remove()

axs[0].set_title(' Total Emisiones (CO2eq) (AR5)', fontsize=13)
axs[0].set_xlabel('País')
axs[0].set_ylabel('Emisiones Totales (kilotones)')
axs[0].tick_params(axis='x', rotation=45)

# --- Subplot 2: Gráfico de torta ---
labels = [f'Top 10 países ({pct_top_10:.1f}%)', f'Resto del mundo ({pct_rest:.1f}%)']
values = [pct_top_10, pct_rest]

axs[1].pie(
    x=values,
    labels=labels,
    colors=sns.color_palette('Greens_r', len(labels)),
    startangle=140,
    autopct=lambda p: f'{p:.1f}%'
)
axs[1].set_title('Participación del Top 10 países emisores sobre el total global', fontsize=13)

# Título global y ajuste de espacio
fig.suptitle('Top 10 Paises Emisiones (CO2eq) (AR5) (2022)', fontsize=16, y=1.05)
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)

# Mostrar en Streamlit
st.pyplot(fig)

st.markdown("""Interpretación:

- China	Con aproximadamente 14 millones de kt lidera con enorme ventaja ( 2,5 × EEUU).
- EE. UU: es el segundo país con mas emisiones en el mundo pero, aun así emite menos de la mitad que China.
- India: se consolida en el tercer lugar, reflejando crecimiento poblacional.
- Rusia, Indonesia, Brasil, Japón, Irán: tienen valores intermedios (0,6 – 2 millones kt). Mezcla de grandes potencias agrícolas (Brasil, Indonesia) y economías industriales/energéticas (Rusia, Irán, Japón).
- Arabia Saudita y México	ocupan el puesto 9 y 10 el ranking. Sus emisiones son < 10 % de las chinas.

Se puede observar una desigualdad extrema: el primer país (China) emite casi 75 veces más que el décimo.

Además, en el gráfico de la derecha podemos observar que, en la actualidad, dos tercios de todas las emisiones agro‑alimentarias se concentran en solo diez países (63%). Por lo tanto, el resto de los paises (180 aprox) aportan el otro tercio.
El gráfico demuestra que las politicas de mitigación global deben hacer foco en unas pocas jurisdicciones. Sin acciones contundentes en esos países, el resto del mundo difícilmente compensará el volumen de emisiones que ahí se genera.""")

st.subheader('Emisiones 2022 — Productos más emisores por país')
st.markdown("""En el próximo Heatmap de productos x paises

cada celda nos muestra las Emisiones (CO2eq) (AR5) en el año 2022 de cada producto en cada país.
Esto nos permite ver que paises son más emisores en cada proceso y ver los procesos críticos de cada región, lo cual permite priorizar acciones.

Para este análisis, seleccionamos los 6 paises con más emisiones en el año 2022 (China, EEUU, India, Rusia, Indonesia y Brasil). Estos países representan el 56% del total de emisiones incluyendo LULUCF a nivel global en el año 2022.')
""")

# Definir regiones de interés
regions = ['China', 'Estados Unidos de América', 'India', 'Indonesia', 'Brasil', 'Federación de Rusia']

# Filtrar datos para 2022 excluyendo productos agregados
df_2022 = (
    df_fao[
        (df_fao['Año'] == 2022) &
        (df_fao['Área'].isin(regions)) &
        (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
        (~df_fao['Código del producto'].isin(codes_agg))
    ]
    .copy()
)

# Correcciones de nombres
fix = {
    "Eliminación de desechos de sistemas agroalimentarios": "Eliminación desechos sist. agro",
}
fix_countries = {
    'Estados Unidos de América': 'EEUU',
    'Federación de Rusia': 'Rusia',
}
df_2022['Producto'] = df_2022['Producto'].replace(fix)
df_2022['Área'] = df_2022['Área'].replace(fix_countries)

# Seleccionar los 15 productos más emisores
top_products = (
    df_2022.groupby('Producto')['Valor_Mt'].sum()
           .nlargest(15).index
)

df_2022 = df_2022[df_2022['Producto'].isin(top_products)]

# Crear pivot para el heatmap
pivot = (
    df_2022
    .pivot_table(index='Producto', columns='Área',
                 values='Valor_Mt', aggfunc='sum')
    .fillna(0)
    .sort_values('China', ascending=False)
)

# Crear figura y heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Greens',
            linewidths=.5, linecolor='white', cbar_kws={'label': 'Mt CO₂-eq'}, ax=ax)

ax.set_title('Emisiones CO2 eq 2022 — Productos más emisores en el Top 6 Países')
ax.set_xlabel('País')
ax.set_ylabel('')
plt.tight_layout()

# Mostrar en Streamlit
st.subheader("🌍 Emisiones CO₂-eq por producto y país (2022)")
st.pyplot(fig)

st.markdown("""#### Conslusiones:
- China: el grueso de las emisiones no proviene del campo, sino de la cadena poscosecha y el consumo urbano (desechos y consumo de alimentos).
- EEUU: tiene valores medianos-altos en casi todas las filas. No tiene un pico, por lo tanto las medidas de acción deberían ser aplicadas de manera multisectorial.
- India: las emisiones se encuentran diversificadas. Es el mayor emisor por fermentación entérica. Además, el consumo de energía dentro de la finca ya supera a la mayoría de los procesos pos-cosecha.
- Indonesia: picos sobresalientes en suelos orgánicos drenados y conversión neta de bosques. El drenaje y la quema de turberas  para palma aceitera liberan grandes cantidades de CO2.
https://rspo.org/es/the-challenges-of-growing-oil-palm-on-peatlands/
- Rusia: sin celdas mayores a 200 Mt, se destacan las emisiones por desechos y suelos orgánicos. Tiene un perfil más parecido al de Europa que al de Brasil/Indonesia.
- Brasil: el cambio en el uso del suelo es el motor de las emisiones. Otro pico alto es la fermentación entérica.


Conclusiones generales:
- No hay una única fuente de emisión que domine en todos los paises. Cada economía tiene su debilidad.
- China y EEUU tienen mayor contaminación en procesos pos-producción (desechos, consumo), mientras Brasil e Indonesia tienen mayores problemas en el sector agrícola.
- Cuatro de los seis paises tienen emisiones mayores a 150 mt por fermentación entérica, esto indica que la ganadería es un motor de contaminación a nivel global.""")


st.subheader("América (Actualidad)")
anio = 2022
countries = ['Estados Unidos de América', 'Brasil', 'Argentina', 'México', 'Colombia', 'Canadá', 'Perú']
product_code = 6825 # Emisiones totales incluyendo LULUCF
gas = 'Emisiones (CO2eq) (AR5)'

df_most_population_america = df_fao[
    (df_fao['Área'].isin(countries)) &
    (df_fao['Año'] == anio) &
    (df_fao['Elemento'] == gas) &
    (df_fao['Código del producto'] == product_code)]

df_top_countries_emission = df_most_population_america.groupby(['Área'])['Valor_Mt'].sum().reset_index()
df_top_countries_emission.sort_values(by= 'Valor_Mt',ascending=False, inplace=True)

total_america = df_fao[
    (df_fao['Área'] == 'Américas') &
    (df_fao['Año'] == anio) &
     (df_fao['Elemento'] == gas) &
    (df_fao['Código del producto'] == product_code)
    ]['Valor_Mt'].sum()

rest_of_america = total_america - df_top_countries_emission['Valor_Mt'].sum()
st.write("")
st.write("")
st.markdown(f"### **Resto de América** representa  **{rest_of_america:.1f}%**.")
st.write("")
st.write("")


df_plot = (pd.concat(
    [df_top_countries_emission ,
     pd.DataFrame({'Área': ['Resto de América'], 'Valor_Mt': [rest_of_america]})
     ], ignore_index=True)
)

df_plot['Share'] = (df_plot['Valor_Mt'] / df_plot['Valor_Mt'].sum()) * 100
df_plot.sort_values('Valor_Mt', ascending=False, inplace=True)

sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
sns.barplot(data=df_plot, y='Área', x='Share', hue='Área',
            palette='Greens_r', edgecolor='black')

for i, row in df_plot.iterrows():
    plt.text(row['Share']+5, i,
             f"{row['Valor_Mt']:,.0f} Mt", va='center')

plt.title(f'Emisiones Totales incluyendo LULUCF CO₂-eq en América ({anio})')
plt.xlabel('% del total continental'); plt.ylabel('')
plt.xlim(0, 100)
plt.tight_layout(); st.pyplot(plt)

st.write("")
st.write("")
st.markdown("""### Análisis por Tipo de Gas y Continente""")

gases = ['Emisiones (CO2)', 'Emisiones (N2O)', 'Emisiones (CH4)']

df_continents_gas = df_fao[
    (df_fao['Área'].isin(continents)) &
    (df_fao['Elemento'].isin(gases)) &
    (df_fao['Producto'] == 'Emisiones totales incluyendo LULUCF')
    ].copy()

df_continents_gas_by_year = df_continents_gas.groupby(['Área', 'Elemento', 'Año'])['Valor_Mt'].sum().reset_index()
df_continents_gas_by_year = df_continents_gas_by_year[df_continents_gas_by_year['Área'] != 'Mundo']
df_continents_gas_by_year.sort_values(by='Valor_Mt', ascending=False, inplace=True)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
axs = axs.ravel()

palette = sns.color_palette('Set2', len(continents) - 1)

# Dibujar cada gas en un eje
for i, gas in enumerate(gases):
    ax = axs[i]
    sns.lineplot(data=df_continents_gas_by_year[df_continents_gas_by_year['Elemento'] == gas],
                 x='Año', y='Valor_Mt', hue='Área',
                 marker='o', palette=palette, ax=ax)

    ax.set_title(gas, fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('Mt CO₂-eq')
    ax.grid(ls='--', alpha=.4)
    if i == 0:
        ax.set_ylabel('Mt CO₂-eq')
    else:
        ax.set_ylabel('')
fig.suptitle('Evolución anual de emisiones por continente (1990-2022)',
             fontsize=15, weight='bold', y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.95])
st.pyplot(plt)

st.write("")
st.write("")

st.markdown("""Asia: es el motor del alza mundial de los tres gases. La pendiente apenas se modera a partir de 2015. Las emisiones de CO2 crecen de de 6 mil Mt a casi 23 mil Mt.

América: las emisiones de CO2 crecen hasta 2005 y luego hay una meseta-descenso a 8000 mt. Las emisiones de NO2 tienen un crecimiento suave de 2 mil Mt a casi 3mil Mt. Mientras que las emisiones de CH4, se mantienen estables.

Europa: es el único continente donde se observa una caída en las emisiones de los tres gases.

África: el salto porcentual es grande (sobretodo en CH4 y N2O), pero la magnitud absoluta sigue muy por debajo de Asia o América. Las emisiones de CO2 crecen muy lentamente.

Oceanía: impacto global muy bajo. Las variaciones anuales están relacionadas a incendios o sequías.


Conclusiones generales:
- N2O es el gas con la pendiente proporcional más alta en África y Asia: es la línea que mas rápido crece en ambos continentes.
- CH4 presenta tendencia ascendente suave excepto en Europa que hay un descenso.
-CO₂ es, con mucha diferencia, el gas dominante en todas las regiones.""")

st.markdown("""### Análisis de Productos Desagregados y su relación con los diferentes tipos de Gas
Para este análisis vamos a tener en cuenta los productos desagregados""")

gases = ['Emisiones (CO2)', 'Emisiones (N2O)', 'Emisiones (CH4)']
anio = 2022
codes_agg = [6518, 6516, 6517, 6996, 6995, 5084, 5085,
             6825, 6829, 6824, 67292, 67291, 69921, 6821, 6817, 6820, 1707, 1711]

df_non_agg_products = df_fao[
    (~df_fao['Código del producto'].isin(codes_agg)) &
    (df_fao['Año'] == anio) &
    (df_fao['Elemento'].isin(gases))
    ]

df_products_top_emissions = df_non_agg_products.groupby(['Producto', 'Elemento'])['Valor_Mt'].sum().reset_index()

top_n = 10
lista_top = []
for g in gases:
    top_g = (df_non_agg_products[df_non_agg_products['Elemento'] == g]
                .groupby('Producto')['Valor_Mt']
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index())
    top_g['Gas'] = g
    lista_top.append(top_g)

df_top_gases = pd.concat(lista_top, ignore_index=True)

df_top_gases = (df_top_gases.set_index(['Gas','Producto'])
                              .sort_index(level='Gas'))

df_top_gases = df_top_gases.rename(
    index={'Eliminación de desechos de sistemas agroalimentarios':
           'Eliminación de desechos de S. Agro.'},
    level='Producto'
)

sns.set_style('whitegrid')
base_palette = sns.color_palette('Greens_r', top_n)
fig, axes = plt.subplots(3, 1, figsize=(10,10), sharex=False)

for ax, g in zip(axes, gases):
    sub = df_top_gases.loc[g].sort_values('Valor_Mt')
    colors = base_palette
    sub.plot.barh(ax=ax, color=colors, edgecolor='black')
    ax.set_title(f'{g.split()[1]}')
    ax.set_xlabel('Mt'); ax.set_ylabel('')
    ax.grid(axis='x', ls='--', alpha=.4)
    ax.get_legend().remove()

fig.suptitle('Top 10 Productos más emisores por gas – Año 2022', y=1.02, fontsize=13)
plt.tight_layout()
st.pyplot(plt)

years = [1990, 2022]
df_crec_prod = df_fao[
    (df_fao['Año'].isin(years)) &
    (df_fao['Área'] == 'Mundo') &
    (~df_fao['Código del producto'].isin(codes_agg)) &
    (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)')
    ].copy()

totals_by_year = df_crec_prod.groupby(['Año', 'Producto'])['Valor_Mt'].sum().unstack('Año').reset_index()

st.markdown("### Evolución de emisiones por producto (1990 vs 2022)")

totals_by_year["Crecimiento absoluto"] = totals_by_year[2022] - totals_by_year[1990]
totals_by_year = totals_by_year.sort_values("Crecimiento absoluto", ascending=False)
st.dataframe(totals_by_year, use_container_width=True)

st.markdown("# **Modelo Predictivo**")
st.markdown("## Utilizando ARIMA:")
st.markdown("#### El siguiente modelo estima las emisiones totales incluyendo LULUCF (Uso de la Tierra, Cambio de Uso de la Tierra y Silvicultura) para cada continente. Se utilizan los datos desde 1990 a 2022 para proyectar como van a evolucionar esas emisiones en los proximos años.")

serie = df_fao[
    (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_fao['Código del producto'] == 6825)
  ].copy()

serie_america = serie[serie['Área'] == 'Américas']
serie_asia = serie[serie['Área'] == 'Asia']
serie_europa = serie[serie['Área'] == 'Europa']
serie_oceania = serie[serie['Área'] == 'Oceanía']
serie_africa = serie[serie['Área'] == 'África']

st.markdown("### 📊 Curva de emisiones en kilotoneladas a lo largo de los años para los distintos continentes. "
            "Evolución de las emisiones agroalimentarias en cada continente a lo largo del tiempo")

st.markdown("""
---
### Cada gráfico representa la evolución de las **emisiones agroalimentarias totales** (en kilotoneladas de CO₂-eq, metodología AR5) desde el año 1990 hasta 2022 en cada continente. A continuación se presentan algunas observaciones generales:

- **América**:
  - Muestra una tendencia creciente con ciertas oscilaciones.
  - Se destacan picos asociados a deforestación y uso intensivo de fertilizantes en décadas recientes.

- **Asia**:
  - Presenta un **crecimiento sostenido y fuerte**.
  - La industrialización y el aumento del consumo urbano explican buena parte del incremento.

- **Europa**:
  - Se observa una **tendencia a la estabilización o incluso leve reducción**.
  - Las políticas ambientales y agrícolas parecen estar moderando las emisiones.

- **Oceanía**:
  - Tiene niveles **bajos y relativamente estables**.
  - La menor población y menor superficie cultivable influyen en estos valores.

- **África**:
  - Las emisiones han crecido de forma progresiva.
  - El aumento se debe al avance de la frontera agropecuaria y la presión sobre ecosistemas.

---

### 🧠 Conclusión general

Estos gráficos permiten **visualizar patrones históricos** que sirven como base para aplicar modelos predictivos (por ejemplo, regresiones o modelos ARIMA) y anticipar el impacto futuro de las políticas agrícolas y alimentarias en cada región.
""")



# Crear figura y subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Graficar cada continente
sns.lineplot(data=serie_america, x='Año', y='Valor', ax=axes[0])
axes[0].set_title('América')

sns.lineplot(data=serie_asia, x='Año', y='Valor', ax=axes[1])
axes[1].set_title('Asia')

sns.lineplot(data=serie_europa, x='Año', y='Valor', ax=axes[2])
axes[2].set_title('Europa')

sns.lineplot(data=serie_oceania, x='Año', y='Valor', ax=axes[3])
axes[3].set_title('Oceanía')

sns.lineplot(data=serie_africa, x='Año', y='Valor', ax=axes[4])
axes[4].set_title('África')

# Eliminar el subplot vacío
fig.delaxes(axes[5])

# Ajustar diseño y mostrar en Streamlit
plt.tight_layout()
st.subheader("📈 Evolución anual de emisiones por continente")
st.pyplot(fig)


st.markdown("""## Análisis de ACF y PACF por continente

En este bloque realizamos un análisis de series temporales sobre las emisiones de CO₂-eq para cada continente, utilizando dos herramientas estadísticas fundamentales:

### 🔁 Función de autocorrelación (ACF)
- La **ACF (Autocorrelation Function)** mide la correlación de la serie con sus propios retardos (lags).
- Permite detectar patrones de repetición o dependencia temporal.
- Si hay autocorrelación significativa en ciertos lags, es señal de que el pasado influye sobre el futuro.

### 📈 Función de autocorrelación parcial (PACF)
- La **PACF (Partial Autocorrelation Function)** mide la correlación entre una observación y sus lags, **controlando por las correlaciones intermedias**.
- Es útil para identificar el orden AR (autoregresivo) en modelos ARIMA.
- Ayuda a decidir cuántos términos autoregresivos incluir (cuántos lags tienen efecto directo).

### ⚠️ Filtro de calidad de datos
- Si una serie tiene menos de 10 observaciones no nulas, **no se grafica** ACF/PACF por falta de datos para un análisis confiable.

### 📌 Aplicación
Este análisis se realiza de forma individual para cada continente (América, Asia, Europa, Oceanía y África) y se muestra un gráfico con dos subplots: ACF y PACF con hasta 15 lags.

Esto es fundamental para modelar emisiones futuras, detectar estacionalidad o dependencia, y elegir modelos estadísticos adecuados.
""")
series = [
    ('América', serie_america),
    ('Asia', serie_asia),
    ('Europa', serie_europa),
    ('Oceanía', serie_oceania),
    ('África', serie_africa)
]

for (nombre, df) in series:
    serie = df['Valor'].dropna()

    if len(serie) < 10:
        st.warning(f"⚠️ Muy pocos datos para mostrar ACF/PACF confiables para **{nombre}**")
        continue

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'ACF y PACF - {nombre}', fontsize=14)

    plot_acf(serie, lags=15, ax=ax[0])
    plot_pacf(serie, lags=15, ax=ax[1])

    ax[0].set_title('ACF')
    ax[1].set_title('PACF')

    plt.tight_layout()
    st.subheader(f"📊 ACF y PACF — {nombre}")
    st.pyplot(fig)

st.markdown("""Conclusiones:

ACF: mide cuánta memoria tiene la serie. En los gráficos se puede ver que las barras bajan de a poco, esto quiere decir que lo que pasó años anteriores todavía pesa hoy. Es una característica de series no estacionarias.

PACF: muestra las influencias directas. Ejemplo: Cuánto empuja 2019 a 2020 directamente, sin contar con la ayuda de 2018, 2017…? En los gráficos se observa una barra alta en el lag 1 y 2 que luego caen.

""")



st.markdown("""
---

### 🔄 Componente estacional de las emisiones por continente

Estos gráficos muestran la **variación estacional** de las emisiones agroalimentarias en cada continente, obtenida mediante un modelo de descomposición STL (Seasonal-Trend decomposition using Loess). A diferencia del gráfico anterior que muestra la tendencia global, este aísla **los patrones cíclicos o repetitivos** presentes en las emisiones a lo largo del tiempo.

#### 🧩 ¿Qué representa la curva en cada gráfico?
- El eje X representa los años.
- El eje Y representa la **fluctuación estacional** de las emisiones (desvinculada de la tendencia general).
- Valores positivos o negativos indican cuánto se desvía la serie por efecto estacional en distintos momentos.

#### 🔍 Observaciones clave:

- **América y Asia** muestran oscilaciones cíclicas claras, lo que sugiere que hay factores repetitivos (como campañas agrícolas o políticas energéticas) que influyen periódicamente.
- **Europa** presenta una componente estacional más tenue, reflejo de políticas más estables y menor variabilidad estructural.
- **África y Oceanía** tienen variaciones más irregulares o menos pronunciadas, posiblemente asociadas a factores climáticos o económicos puntuales.

---

### 📌 Conclusión

El análisis estacional es fundamental para detectar **patrones ocultos** que se repiten en el tiempo. Estos insights permiten afinar los modelos predictivos, entender la influencia del calendario agrícola, y anticipar picos o caídas sistemáticas en las emisiones.
""")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Lista de series y títulos
series = [
    ('América', serie_america['Valor']),
    ('Asia', serie_asia['Valor']),
    ('Europa', serie_europa['Valor']),
    ('Oceanía', serie_oceania['Valor']),
    ('África', serie_africa['Valor'])
]

# Graficar componente 'seasonal' de cada STL
for i, (titulo, serie) in enumerate(series):
    stl = STL(serie, period=5).fit()
    axes[i].plot(serie.index, stl.seasonal)
    axes[i].set_title(f'Tendencia - {titulo}')
    axes[i].set_xlabel('Año')
    axes[i].set_ylabel('Valor')

# Eliminar el subplot vacío (el sexto)
fig.delaxes(axes[5])

plt.tight_layout()
st.pyplot(plt)

st.markdown("### 📊 Proporción de varianza explicada por la estacionalidad")

# Lista de pares (nombre, serie)
series = [
    ('América', serie_america['Valor']),
    ('Asia', serie_asia['Valor']),
    ('Europa', serie_europa['Valor']),
    ('Oceanía', serie_oceania['Valor']),
    ('África', serie_africa['Valor'])
]

# Calcular proporción varianza estacional / total
resultados = []
for nombre, serie in series:
    stl = STL(serie, period=5).fit()
    proporción = stl.seasonal.var() / serie.var()
    resultados.append((nombre, round(proporción, 3)))

# Mostrar en tabla
df_var = pd.DataFrame(resultados, columns=["Región", "Var(seasonal) / Var(total)"])
st.dataframe(df_var, use_container_width=True)

st.markdown("""La varianza explicada por la componente estacional en todas las regiones se encuentra entre el **0.1% y el 2.8%** de la varianza total. Esto indica que **no existe un patrón estacional fuerte** en ninguna de las series.

En particular:

- Asia, África y América muestran una **estacionalidad muy débil** (≤ 1.5%).
- Europa, aunque algo mayor, también se mantiene por debajo del 3%.
- En Oceanía no se pudo calcular por falta de variabilidad o datos incompletos.

### 🧠 Conclusión:
Dado que la **contribución estacional es insignificante**, tratamos las series como **no estacionales**, y podemos modelarlas directamente con un modelo **ARIMA convencional** (o un SARIMA con `s = 1`, que equivale a lo mismo). Solo es necesario diferenciar la serie para eliminar la tendencia.
""")

st.markdown("""### Pruebas de estacionaridad""")
st.markdown("""# 📉 Prueba ADF (Augmented Dickey-Fuller)

La **prueba ADF (Augmented Dickey-Fuller)** es una prueba estadística fundamental en el análisis de series temporales. Se utiliza para determinar si una serie es **estacionaria**, es decir, si sus propiedades estadísticas (como la media, la varianza y la autocorrelación) **se mantienen constantes en el tiempo**.

---

## 🔍 ¿Por qué es importante la estacionariedad?

Muchos modelos de series temporales —como **ARIMA**, **SARIMA**, etc.— requieren que la serie sea estacionaria para funcionar correctamente. Si una serie no es estacionaria (por ejemplo, tiene una tendencia o estacionalidad no corregida), los modelos pueden producir **predicciones sesgadas o erráticas**.

---

## 📐 Fundamento de la prueba ADF

La prueba ADF es una extensión de la **prueba de Dickey-Fuller**, que incluye **términos adicionales de rezago (lags)** de la variable dependiente para capturar autocorrelación residual y mejorar la robustez del test.

La forma general de la regresión que se estima es:

\[
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t
\]

Donde:

- \( y_t \) es la serie original.
- \( \Delta y_t = y_t - y_{t-1} \) representa la primera diferencia.
- \( t \) es una tendencia determinista (opcional).
- \( \gamma \) es el parámetro clave que se analiza.
- \( \varepsilon_t \) es el error aleatorio.
- \( p \) es el número de rezagos incluidos.

---

## 🎯 Hipótesis de la prueba ADF

| Hipótesis nula \( H_0 \)            | Hipótesis alternativa \( H_1 \)         |
|-------------------------------------|-----------------------------------------|
| La serie tiene una raíz unitaria → no es estacionaria. | La serie es estacionaria (no tiene raíz unitaria). |

---

## 🧪 Interpretación de resultados

- **Si el valor p es menor que un nivel de significancia (por ejemplo, 0.05):**
  - Se **rechaza la hipótesis nula**.
  - Concluimos que la serie **es estacionaria**.

- **Si el valor p es mayor que 0.05:**
  - **No se puede rechazar** la hipótesis nula.
  - La serie **no es estacionaria**.

También puede compararse el **estadístico ADF** con los **valores críticos** (critical values) al 1%, 5% y 10%.

---
""")
st.markdown("### 📉 Test de Estacionariedad ADF por continente")

series_continentales = {
    'América': serie_america['Valor'],
    'Asia': serie_asia['Valor'],
    'Europa': serie_europa['Valor'],
    'Oceanía': serie_oceania['Valor'],
    'África': serie_africa['Valor'],
}

# Para tabla resumen
resumen_adf = []

# Evaluar ADF
for nombre, serie in series_continentales.items():
    st.markdown(f"#### 🌍 {nombre}")
    serie_sin_na = serie.dropna()

    if len(serie_sin_na) < 3:
        st.warning("Serie vacía o con muy pocos datos, se omite.")
        resumen_adf.append({
            "Región": nombre,
            "ADF Statistic": "N/A",
            "p-value": "N/A",
            "Estacionaria": "No evaluada"
        })
        continue

    try:
        result = adfuller(serie_sin_na)
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]

        st.markdown(f"- **ADF Statistic**: `{adf_stat:.4f}`")
        st.markdown(f"- **p-value**: `{p_value:.4f}`")
        st.markdown("- **Valores críticos:**")
        for key, value in critical_values.items():
            st.markdown(f"  - {key}: `{value:.4f}`")

        if p_value < 0.05:
            st.success("✅ La serie **es estacionaria** (se rechaza H0)")
            conclusion = "✅ Sí"
        else:
            st.error("🚫 La serie **NO es estacionaria** (no se rechaza H0)")
            conclusion = "🚫 No"

        resumen_adf.append({
            "Región": nombre,
            "ADF Statistic": round(adf_stat, 4),
            "p-value": round(p_value, 4),
            "Estacionaria": conclusion
        })

    except Exception as e:
        st.error(f"⚠️ Error al procesar la serie: {e}")
        resumen_adf.append({
            "Región": nombre,
            "ADF Statistic": "Error",
            "p-value": "Error",
            "Estacionaria": "Error"
        })

# Mostrar tabla resumen
st.markdown("### 📋 Resumen del test ADF")
df_resumen_adf = pd.DataFrame(resumen_adf)
st.dataframe(df_resumen_adf, use_container_width=True)

st.write("")
st.write("")

st.markdown("""# 📊 Prueba KPSS (Kwiatkowski–Phillips–Schmidt–Shin)

La **prueba KPSS** es una herramienta estadística que se utiliza para verificar la **estacionariedad** de una serie temporal. A diferencia de la prueba ADF (Augmented Dickey-Fuller), la prueba KPSS parte de una hipótesis **opuesta**.

---

## 🧠 ¿Por qué KPSS?

Mientras que la prueba ADF evalúa si una serie tiene **una raíz unitaria** (es decir, si **no es estacionaria**), la prueba KPSS evalúa si la serie es **estacionaria en nivel o tendencia**.

---

## ⚖️ Hipótesis de la prueba KPSS

| Hipótesis nula \( H_0 \)                    | Hipótesis alternativa \( H_1 \)                      |
|---------------------------------------------|------------------------------------------------------|
| La serie **es estacionaria** (en nivel o tendencia). | La serie **no es estacionaria** (tiene raíz unitaria). |

> ⚠️ ¡Esto es exactamente lo contrario de la prueba ADF!

---

## 📐 Formulación

La prueba se basa en la descomposición de una serie temporal como:

\[
y_t = r_t + \beta t + \varepsilon_t
\]

Donde:
- \( r_t \): componente estacionaria (o aleatoria).
- \( \beta t \): tendencia determinista.
- \( \varepsilon_t \): error aleatorio.

Se calcula un estadístico de prueba que mide la varianza acumulada de los residuos de una regresión de \( y_t \) sobre \( t \), y se compara contra valores críticos.

---

## 🧪 Interpretación de resultados

- Si el **p-valor es bajo (p < 0.05)**:
  - Se **rechaza la hipótesis nula**.
  - Concluimos que la serie **no es estacionaria**.
  
- Si el **p-valor es alto (p ≥ 0.05)**:
  - **No se rechaza** la hipótesis nula.
  - Se considera que la serie **es estacionaria**.

""")

st.markdown("### 📉 Test de Estacionariedad KPSS por continente")


### ✅ 2. Versión en **Streamlit** del código que ejecuta la prueba KPSS para cada serie continental

alpha = 0.05  # nivel de significancia

for nombre, serie in series_continentales.items():
    st.subheader(f"🌍 {nombre}")

    serie = serie.dropna()

    if len(serie) < 3:
        st.warning("⚠️ Serie vacía o muy corta, se omite.")
        continue

    try:
        stat, p, lags, crit = kpss(serie, regression='ct')
        st.write(f"**Estadístico KPSS:** {stat:.3f}")
        st.write(f"**p-valor:** {p:.3f}")
        st.write(f"**Número de rezagos:** {lags}")

        if p < alpha:
            st.error("❌ **NO estacionaria** (se rechaza H₀)")
        else:
            st.success("✅ Sin evidencia contra la estacionaridad (no se rechaza H₀)")

        # Mostrar los valores críticos como tabla
        st.markdown("**Valores críticos:**")
        st.table(pd.DataFrame(crit.items(), columns=["Nivel", "Valor crítico"]))

    except Exception as e:
        st.warning(f"⚠️ Error al procesar {nombre}: {e}")



st.markdown("""### Resultados de las Pruebas de Estacionariedad:

- América: tanto la prueba ADF como KPSS coinciden en que la serie no es estacionaria. Tratamiento: vamos a realizar una diferenciación (d = 1).
- Asia: existe un conflicto leve entre las pruebas. ADF rechaza la estacionariedad, mientras KPSS no la rechaza. Tratamiento: vamos a partir de una primera diferenciación y hacer pruebas.
- Europa: conflicto. KPSS indica que no hay estacionariedad mientras ADF rechaza la H0, indicando lo contrario. Tratamiento: vamos a hacer pruebas luego de una primera diferenciación.
- Oceanía: las pruebas se contradicen. Tratamiento: d = 1.
- África: ADF concluye que la serie no es estacionaria y KPSS no tiene evidencia contra la estacionariedad. Tratamiento: primera diferenciación.""")

st.markdown("### 🔍 Estacionariedad y diferenciación para modelado ARIMA")
st.markdown("""## 📉 Diferenciación

### ¿Para qué sirve?

La **diferenciación** es la operación más sencilla y común para "arreglar" una serie temporal que **no es estacionaria**.

Consiste en restar cada valor de la serie con su valor inmediatamente anterior:

\[
\Delta y_t = y_t - y_{t-1}
\]

Esta operación elimina tendencias lineales o estructuras de crecimiento acumulativo, haciendo que la serie tenga **media y varianza más estables en el tiempo**.

Es una herramienta clave en el modelado con ARIMA, donde el parámetro \( d \) indica cuántas veces se debe diferenciar la serie para volverla estacionaria.
""")

alpha = 0.05  # Nivel de significancia

# Diccionario de series por continente
differenced_series = {
    'América': serie_america,
    'Asia': serie_asia,
    'Europa': serie_europa,
    'Oceanía': serie_oceania,
    'África': serie_africa,
}

# Lista para resumen final
resultados_adf_kpss = []

# Función auxiliar
def test_adf(serie):
    result = adfuller(serie.dropna())
    adf_stat, p_value, _, _, critical_values, _ = result
    return adf_stat, p_value, critical_values

for nombre, df in differenced_series.items():
    st.markdown(f"#### 🌍 {nombre}")

    df = df.copy()
    df['Valor_diff'] = df['Valor'].diff()
    df.dropna(subset=['Valor_diff'], inplace=True)
    serie_diff = df['Valor_diff']

    if serie_diff.size < 3:
        st.warning("⚠️ No hay suficientes datos para testear la serie diferenciada.")
        resultados_adf_kpss.append({
            "Región": nombre,
            "ADF diferenciada": "–",
            "p-valor ADF": "–",
            "Estacionaria ADF": "No evaluada",
            "KPSS diferenciada": "–",
            "p-valor KPSS": "–",
            "Estacionaria KPSS": "No evaluada",
        })
        continue

    # --- ADF ---
    adf_stat, pval_adf, crit_adf = test_adf(serie_diff)
    est_adf = "Sí" if pval_adf < alpha else "No"
    st.markdown(f"- **ADF:** estadístico = `{adf_stat:.4f}`, p-valor = `{pval_adf:.4f}` → Estacionaria: **{'✅ Sí' if est_adf == 'Sí' else '🚫 No'}**")

    # --- KPSS ---
    try:
        kpss_stat, pval_kpss, lags_kpss, crit_kpss = kpss(serie_diff, regression='ct')
        est_kpss = "No" if pval_kpss < alpha else "Sí"
        st.markdown(f"- **KPSS:** estadístico = `{kpss_stat:.4f}`, p-valor = `{pval_kpss:.4f}` → Estacionaria: **{'✅ Sí' if est_kpss == 'Sí' else '🚫 No'}**")
    except Exception as e:
        st.warning(f"⚠️ Error en prueba KPSS para {nombre}: {e}")
        kpss_stat = pval_kpss = est_kpss = "–"

    # Guardar en resumen
    resultados_adf_kpss.append({
        "Región": nombre,
        "ADF diferenciada": round(adf_stat, 4),
        "p-valor ADF": round(pval_adf, 4),
        "Estacionaria ADF": est_adf,
        "KPSS diferenciada": round(kpss_stat, 4) if isinstance(kpss_stat, float) else kpss_stat,
        "p-valor KPSS": round(pval_kpss, 4) if isinstance(pval_kpss, float) else pval_kpss,
        "Estacionaria KPSS": est_kpss
    })

# --- Mostrar resumen final ---
st.markdown("### 📋 Resumen de pruebas ADF y KPSS sobre la serie diferenciada")
df_diff_resumen = pd.DataFrame(resultados_adf_kpss)
st.dataframe(df_diff_resumen, use_container_width=True)




st.markdown("""
---
## 📌 Conclusiones luego de testear la estacionariedad en las series diferenciadas

- **América:** ambos tests concuerdan. Serie **estacionaria**.
- **Asia:** resultado **mixto**. ADF afirma que es estacionaria, mientras KPSS indica que **no lo es**.
- **Europa:** los tests coinciden. Serie **estacionaria**.
- **Oceanía:** los tests coinciden. Serie **estacionaria**.
- **África:** los tests coinciden. Serie **estacionaria**.

---

## ⚙️ Tratamientos propuestos

Se modelará cada serie con una **primera diferenciación** (`d=1`). Luego, se evaluará el comportamiento de los **residuos del modelo**:

- Si los residuos se comportan como **ruido blanco** (sin autocorrelación),
- entonces se considerará que la elección de `d=1` fue **adecuada**,
- **independientemente** de que un test (ADF o KPSS) aislado sugiera lo contrario.
""")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

series_diff = [
    ('América', serie_america),
    ('Asia', serie_asia),
    ('Europa', serie_europa),
    ('Oceanía', serie_oceania),
    ('África', serie_africa),
]

for i, (nombre, df) in enumerate(series_diff):
    df = df.sort_values('Año')  # <-- Asegura que el eje X esté bien ordenado

    if 'Valor_diff' in df and df['Valor_diff'].dropna().size >= 3:
        sns.lineplot(x=df['Año'], y=df['Valor_diff'], ax=axes[i])
        axes[i].set_title(f'{nombre} - Valor diferenciado')
        axes[i].set_ylabel('Δ Valor')
    elif 'Valor' in df and df['Valor'].dropna().size >= 3:
        sns.lineplot(x=df['Año'], y=df['Valor'], ax=axes[i])
        axes[i].set_title(f'{nombre} - Serie original (estacionaria)')
        axes[i].set_ylabel('Valor')
    else:
        axes[i].set_title(f'{nombre} - Sin datos')
        axes[i].axis('off')

    axes[i].set_xlabel('Año')

fig.delaxes(axes[5])
plt.tight_layout()
st.subheader("📉 Visualización de series diferenciadas u originales por continente")
st.pyplot(fig)

st.markdown("""## 📊 ACF y PACF en series temporales

### 🔄 ¿Qué es la ACF (Autocorrelation Function)?

La **Función de Autocorrelación (ACF)** mide la correlación lineal entre una serie temporal y **sus propios rezagos** (valores pasados).

- **ACF(k)** indica cuánto se correlaciona la serie consigo misma desplazada `k` pasos.
- Se representa con un gráfico que muestra los coeficientes de correlación para distintos rezagos.
- Incluye tanto los efectos **directos como indirectos** (es decir, puede verse afectada por rezagos intermedios).

#### 📌 ¿Para qué se usa?

- Para identificar la presencia de **dependencia temporal**.
- Para ayudar a definir el orden `q` en modelos **ARIMA** (componente MA: media móvil).
- Para detectar patrones de estacionalidad o ciclos.

---

### 🔁 ¿Qué es la PACF (Partial Autocorrelation Function)?

La **Función de Autocorrelación Parcial (PACF)** mide la correlación entre la serie y sus rezagos, **controlando los efectos de los rezagos intermedios**.

- PACF(k) muestra la relación entre `X_t` y `X_{t-k}` *una vez eliminada* la influencia de los rezagos `1` hasta `k-1`.
- Aísla la contribución directa de cada rezago.

#### 📌 ¿Para qué se usa?

- Para estimar el orden `p` en un modelo **ARIMA** (componente AR: autorregresivo).
- Permite entender **cuál es el número mínimo de rezagos necesarios** para explicar la dependencia.

---

### 📈 ¿Cómo se interpretan los gráficos?

Ambas funciones se grafican con líneas verticales por cada rezago, junto a un intervalo de confianza (por ejemplo, 95%).

- Si un valor **supera el límite de confianza**, se considera **estadísticamente significativo**.
- Una **caída brusca** en ACF o PACF sugiere el orden adecuado para `q` o `p` respectivamente.

---

### 🧠 En resumen

| Función | Evalúa...                        | Ayuda a definir... | Considera efectos indirectos |
|---------|----------------------------------|---------------------|------------------------------|
| **ACF** | Correlación con rezagos          | `q` en ARIMA        | ✅ Sí                         |
| **PACF**| Correlación parcial (solo directa)| `p` en ARIMA        | ❌ No                         |
""")

# 1. Diferenciar primero
for nombre, df in series_diff:
    if 'Valor' in df and 'Valor_diff' not in df:
        df['Valor_diff'] = df['Valor'].diff()
        df.dropna(inplace=True)

# 2. Luego graficar
for nombre, df in series_diff:
    st.markdown(f"### 🌎 {nombre}")

    if 'Valor_diff' not in df:
        st.warning("⚠️ No tiene columna `Valor_diff`, se omite.")
        continue

    serie = df['Valor_diff'].dropna()

    if len(serie) < 10:
        st.warning("⚠️ Muy pocos datos para mostrar ACF/PACF confiables.")
        continue

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'ACF y PACF - {nombre} (Valor diferenciado)', fontsize=14)

    plot_acf(serie, lags=15, ax=ax[0])
    plot_pacf(serie, lags=15, ax=ax[1])

    ax[0].set_title('ACF')
    ax[1].set_title('PACF')

    st.pyplot(fig)




st.markdown("""## ✅ Estacionariedad lograda con primera diferencia

La **primera diferencia** logró que las series sean estacionarias:

- Las **colas largas** en la función de autocorrelación (ACF) desaparecieron.
- La mayoría de las barras están dentro de la **franja de confianza**.
- Todas las series cumplen con el requisito de **varianza y media constantes**.

Por lo tanto, ya podemos aplicar un modelo **ARIMA** a cada una.

---

## 🔍 Selección del mejor modelo ARIMA

Luego de lograr la estacionariedad, vamos a buscar el **mejor modelo ARIMA** para cada serie.

El siguiente procedimiento permite comparar múltiples modelos con:

- `d = 1` (primera diferencia fija),
- `p` y `q` variando entre 0 y 3.

Se presentan los **3 mejores modelos** según:

1. **Precisión** del pronóstico (*MAPE*),
2. **Simplicidad** del modelo (*AIC*),
3. **Validez estadística** del ajuste (residuos como **ruido blanco**).
""")


# Ignorar warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore")

# Diccionario de series originales (ya diferenciadas previamente si corresponde)
series = {
    'América': serie_america['Valor'].astype(float),
    'Asia':    serie_asia['Valor'].astype(float),
    'Europa':  serie_europa['Valor'].astype(float),
    'Oceanía': serie_oceania['Valor'].astype(float),
    'África':  serie_africa['Valor'].astype(float)
}

# Hiperparámetros de test y diferenciación
h_test = 5
d = 1

# Función para buscar los mejores modelos ARIMA
def grid_search_arima(y_train, y_test, p_max=3, q_max=3, top_k=3, lb_lags=10):
    resultados = []
    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        try:
            modelo = SARIMAX(y_train, order=(p, d, q),
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)

            forecast = modelo.get_forecast(steps=len(y_test)).predicted_mean
            mape = mean_absolute_percentage_error(y_test, forecast) * 100

            lb_p = acorr_ljungbox(modelo.resid, lags=[lb_lags],
                                  return_df=True)['lb_pvalue'].iloc[-1]

            resultados.append({
                'order': (p, d, q),
                'aic': modelo.aic,
                'mape': mape,
                'lb_p': lb_p,
                'ok': lb_p > 0.05,
                'mod': modelo
            })
        except Exception:
            continue

    resultados = sorted(resultados, key=lambda x: (x['mape'], x['aic']))
    return resultados[:top_k]

# Título principal en Streamlit
st.title("📊 Comparación de modelos ARIMA")

# Loop sobre todas las series
resultados = {}
for nombre, serie in series.items():
    st.markdown(f"## 🌎 {nombre}")

    y_train, y_test = serie.iloc[:-h_test], serie.iloc[-h_test:]
    top_modelos = grid_search_arima(y_train, y_test)

    resultados[nombre] = top_modelos

    for i, modelo in enumerate(top_modelos, 1):
        st.markdown(f"""
        **{i}. ARIMA{modelo['order']}**  
        • 📉 AIC = `{modelo['aic']:.2f}`  
        • 🎯 MAPE (test) = `{modelo['mape']:.2f}%`  
        • 🧪 Ljung‑Box p-valor = `{modelo['lb_p']:.3f}` → {'✅ OK' if modelo['ok'] else '❌ NO'}
        """)
st.markdown("""Se exploraron modelos **ARIMA (p,1,q)** con `p` y `q` entre 0 y 3.  
Para cada continente se muestran los **3 modelos con menor error de pronóstico** sin sobreajustar la serie (medido mediante el **AIC**).  
Además, se evaluó que los residuales **no presenten autocorrelación** mediante el test de **Ljung-Box**.

##### 📌 Modelos Seleccionados:

- **América**: ARIMA(0,1,3)  
- **Asia**: ARIMA(1,1,1)  
- **Europa**: ARIMA(1,1,3)  
- **Oceanía**: ARIMA(2,1,3)  
- **África**: ARIMA(2,1,3)
""")

import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Parámetros por continente
parametros_arima = {
    'América': (0, 1, 3),
    'Asia': (1, 1, 1),
    'Europa': (1, 1, 3),
    'Oceanía': (2, 1, 3),
    'África': (2, 1, 3)
}

# Series de tiempo
series_dict = {
    'América': serie_america,
    'Asia': serie_asia,
    'Europa': serie_europa,
    'Oceanía': serie_oceania,
    'África': serie_africa
}

h_test = 5

for nombre, df in series_dict.items():
    st.markdown(f"### 🌍 {nombre}")

    if 'Valor' not in df.columns:
        st.warning("⚠️ No tiene columna `Valor`, se omite.")
        continue

    y = df['Valor'].dropna()
    y_train, y_test = y.iloc[:-h_test], y.iloc[-h_test:]

    if len(y_train) < 10:
        st.warning("⚠️ Muy pocos datos para ajustar el modelo.")
        continue

    try:
        p, d, q = parametros_arima[nombre]
        model = SARIMAX(
            y_train,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)

        # Mostrar resumen como texto en bloque
        st.code(res.summary().as_text(), language='text')

        # Diagnóstico gráfico
        fig = res.plot_diagnostics(figsize=(10, 5))
        plt.suptitle(f'Diagnóstico del modelo ARIMA para {nombre}', fontsize=14)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error al ajustar modelo para {nombre}: {e}")
st.markdown("""## Análisis de los Residuales del Modelo ARIMA

- **Línea de Residuales:**  
  Podemos observar que, en los gráficos de línea de residuales, los errores del modelo año a año oscilan alrededor de cero, salvo algún pico aislado en *América*.

- **Histograma:**  
  En el histograma se aprecia la distribución de los errores.  
  - En *África* y *Asia*, la distribución coincide bastante con la línea verde que representa una distribución normal.  
  - En *América* y *Europa*, se observan colas un poco más anchas, indicando cierta desviación de la normalidad.  
  - *Oceanía* presenta el mejor ajuste, con una distribución muy cercana a la normal.

- **Q-Q Plot:**  
  El gráfico Q-Q compara los errores reales con los que tendría una distribución normal perfecta.  
  - La mayoría de los puntos siguen la línea roja, lo que indica normalidad en los residuos.  
  - Se observan leves desvíos en algunos casos, pero no son significativos.

- **Correlograma de Residuales (ACF):**  
  Permite observar si hay autocorrelación en los errores.  
  - En todos los gráficos, las barras se encuentran dentro de la franja azul de confianza, lo que sugiere que **los residuos se comportan como ruido blanco**, sin memoria temporal significativa.
""")

st.markdown("""
### 🔮 Predicciones

En el caso de **África** y **Asia**, luego de la primera diferenciación observamos que los **cambios anuales son casi siempre positivos**, es decir, cada año se emite un poco más de gases de efecto invernadero.

Por este motivo, se añade el parámetro `trend='t'` al modelo ARIMA, lo cual le permite **proyectar una tendencia creciente** teniendo en cuenta la dinámica reciente de la serie.

> 💡 **Alternativa:**  
> Otra opción sería aplicar una **segunda diferenciación** (`d=2`), aunque esto podría introducir más ruido y hacer que el modelo pierda información relevante sobre la tendencia subyacente.
""")
series_dict = {
    'América': serie_america,
    'Asia': serie_asia,
    'Europa': serie_europa,
    'Oceanía': serie_oceania,
    'África': serie_africa
}

modelos_config = {
    'América':  {'order': (0, 1, 3)},
    'Asia':     {'order': (1, 1, 1), 'trend': 't'},
    'Europa':   {'order': (1, 1, 3)},
    'Oceanía':  {'order': (2, 1, 3)},
    'África':   {'order': (2, 1, 3), 'trend': 't'}
}

h_test, h_future = 5, 5

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (nombre, df) in enumerate(series_dict.items()):
    serie = df.set_index('Año')['Valor'].sort_index().dropna().astype(float)
    serie.index = pd.PeriodIndex(serie.index, freq='Y')

    y_train, y_test = serie.iloc[:-h_test], serie.iloc[-h_test:]
    cfg = modelos_config[nombre]
    trend = cfg.get('trend', 'n')

    res = SARIMAX(
        y_train,
        order=cfg['order'],
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    fitted = res.fittedvalues
    fc_test = res.get_forecast(h_test)
    pred_test = fc_test.predicted_mean
    ci_test = fc_test.conf_int()

    fc_fut = res.get_forecast(h_future)
    pred_fut = fc_fut.predicted_mean
    ci_fut = fc_fut.conf_int()

    pred_test.index = pd.period_range(y_train.index[-1] + 1, periods=h_test, freq='Y')
    pred_fut.index = pd.period_range(pred_test.index[-1] + 1, periods=h_future, freq='Y')
    ci_test.index, ci_fut.index = pred_test.index, pred_fut.index

    pred_full = pd.concat([fitted, pred_test, pred_fut])
    mape = mean_absolute_percentage_error(y_test, pred_test) * 100

    ax = axes[i]
    ax.plot(serie.index.to_timestamp(), serie, label='Observado', color='steelblue', alpha=.6)
    ax.plot(pred_full.index.to_timestamp(), pred_full, color='firebrick', lw=2, label='Modelo ARIMA')
    ax.fill_between(pred_test.index.to_timestamp(), ci_test.iloc[:, 0], ci_test.iloc[:, 1],
                    color='firebrick', alpha=.25, label='IC 95 % (test)')
    ax.fill_between(pred_fut.index.to_timestamp(), ci_fut.iloc[:, 0], ci_fut.iloc[:, 1],
                    color='darkorange', alpha=.20, label='IC 95 % (futuro)')

    ax.axvline(y_train.index[-1].to_timestamp(), color='grey', ls='--')
    ax.axvline(y_test.index[-1].to_timestamp(), color='grey', ls='--')

    ax.set(title=f'{nombre} (MAPE: {mape:.2f}%)',
           xlabel='Año', ylabel='kt CO₂-eq')
    ax.legend()
    ax.grid(ls='--', alpha=.3)

# Si hay menos de 6 gráficos, eliminar ejes sobrantes
if len(series_dict) < len(axes):
    for j in range(len(series_dict), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)


st.markdown("""
### 🌊 Caso Oceanía

En la serie de Oceanía habíamos observado que la **varianza explicada por un ciclo de 5 años supera el 38%**.  
Sin embargo, luego de aplicar una primera diferenciación, observamos en los gráficos **ACF** y **PACF** que los *lags* 4, 5 y 10 **caen dentro de la franja azul**, lo que indica que **la estacionalidad ya está explicada**.

Aplicar un modelo **SARIMA** implicaría agregar más parámetros para representar solo **6 ciclos observados (30 años)**, lo cual podría llevar a un **sobreajuste** por la escasez de datos.

Además, los resultados obtenidos con el modelo **ARIMA simple** muestran un **ajuste adecuado** y consistente, por lo tanto **no se justifica complejizar el modelo** con un componente estacional.
""")


st.markdown("""## Utilizando Prophet""")

st.markdown("""
### 🔮 ¿Qué es Prophet?

**Prophet** es una herramienta de pronóstico de series temporales desarrollada por **Facebook (Meta)**. Está pensada para:

- Modelar **tendencias no lineales**.
- Capturar **cambios de régimen** o inflexiones en la evolución histórica.
- Incluir opcionalmente **estacionalidades** (diarias, semanales, anuales).
- Ser **fácil de usar** para analistas sin conocimientos avanzados en estadística.

---

### 🧮 Prophet descompone la serie temporal de la siguiente forma:

- y(t) = g(t) + s(t) + h(t) + ε_t            

Donde:

- **g(t)** → Tendencia (puede ser lineal o logística, con posibles "cambios de pendiente").
- **s(t)** → Estacionalidad (opcional, puede ser anual, semanal, diaria).
- **h(t)** → Efectos por fechas especiales (festivos, eventos).
- **ε_t** → Ruido aleatorio (residuo no explicado).

---

### ✅ Beneficios clave frente a ARIMA

| Beneficio clave                       | ¿Por qué importa en tu dataset?                                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
|  Captura **cambios de tendencia**     | Las emisiones no evolucionan linealmente. Prophet detecta **cambios de pendiente automáticamente**, lo que ARIMA no hace bien.  |
|  No requiere **estacionariedad**      | Prophet **no exige diferenciar** ni transformar la serie. SARIMA sí, y esto puede distorsionar el significado del pronóstico.   |
|  Funciona bien con **datos anuales**  | Las series son anuales. Prophet acepta fácilmente series con cualquier frecuencia sin reconfigurar nada.                        |
|  Maneja bien la **incertidumbre**     | Prophet devuelve automáticamente **intervalos de confianza del 95%**, facilitando la comunicación de riesgo/incertidumbre.      |
|  Automatizable por región             | Se puede aplicar el mismo modelo a cada continente sin tunear manualmente los parámetros. Ideal para **automatización**.        |
|  Interpretabilidad de componentes     | Prophet permite ver **la tendencia sola**, algo útil para análisis visual y argumentación.                                      |
""")

h_test = 5
h_future = 5
resultados = {}

fig, axes = plt.subplots(2, 3, figsize=(22, 10))
axes = axes.flatten()

for idx, (nombre, df) in enumerate(series_dict.items()):
    # Preprocesamiento
    serie = df[['Año', 'Valor']].dropna().copy()
    serie = serie.sort_values('Año')
    serie['Año'] = serie['Año'].astype(int)
    serie = serie.rename(columns={'Año': 'ds', 'Valor': 'y'})
    serie['ds'] = pd.to_datetime(serie['ds'], format='%Y')

    y_train = serie.iloc[:-h_test]
    y_test = serie.iloc[-h_test:]

    # Modelo Prophet
    model = Prophet(
        yearly_seasonality=False,
        changepoint_range=0.9,
        n_changepoints=5
    )
    model.add_seasonality(name='quinquenal', period=5, fourier_order=2)
    model.fit(y_train)

    future_test = model.make_future_dataframe(periods=h_test, freq='Y', include_history=False)
    forecast_test = model.predict(future_test)

    future_total = model.make_future_dataframe(periods=h_test + h_future, freq='Y')
    forecast_total = model.predict(future_total)

    pred_test = forecast_test.set_index('ds')['yhat']
    pred_future = forecast_total.set_index('ds')['yhat'].iloc[-h_future:]
    ci_test = forecast_test.set_index('ds')[['yhat_lower', 'yhat_upper']]
    ci_future = forecast_total.set_index('ds')[['yhat_lower', 'yhat_upper']].iloc[-h_future:]

    mape = mean_absolute_percentage_error(y_test['y'].values, pred_test.values) * 100

    # Gráfico
    ax = axes[idx]
    ax.plot(y_train['ds'], y_train['y'], label='Histórico (train)', color='steelblue')
    ax.plot(y_test['ds'],  y_test['y'], label='Real (test)', color='black', lw=2)
    ax.plot(pred_test.index, pred_test.values, label='Pronóstico test', color='firebrick')
    ax.fill_between(pred_test.index, ci_test['yhat_lower'], ci_test['yhat_upper'],
                    color='firebrick', alpha=.25)
    ax.plot(pred_future.index, pred_future.values, label='Proyección 8 años', color='orange')
    ax.fill_between(pred_future.index, ci_future['yhat_lower'], ci_future['yhat_upper'],
                    color='orange', alpha=.20)
    ax.axvline(y_train['ds'].iloc[-1], color='grey', ls='--', lw=1)
    ax.axvline(y_test['ds'].iloc[-1],  color='grey', ls='--', lw=1)
    ax.set(title=f'{nombre} (MAPE={mape:.2f}%)', xlabel='Año', ylabel='Kt CO₂‑eq')
    ax.grid(ls='--', alpha=.4)

    resultados[nombre] = {
        'modelo': model,
        'pred_test': pred_test,
        'ci_test': ci_test,
        'pred_future': pred_future,
        'ci_future': ci_future,
        'mape': mape
    }

# Ocultar ejes sobrantes si hay menos de 6
if len(series_dict) < 6:
    for j in range(len(series_dict), 6):
        fig.delaxes(axes[j])

fig.tight_layout()
st.pyplot(fig)


st.markdown("""
### ✅ Conclusiones

Aplicando **Prophet** con una tendencia quinquenal, observamos que en la mayoría de los casos el **📉 MAPE es mayor** que en ARIMA, salvo en Asia donde es casi similar.

🔍 En este caso, **preferimos mantener el modelo ARIMA**, ya que Prophet es más confiable en series con:

- ⏱️ **Mayor frecuencia temporal** (diaria, mensual),
- 📈 **Más observaciones históricas**.

> ⚠️ Con solo ~30 observaciones por serie, **Prophet tiende a sobreajustarse**, interpretando como patrones reales lo que probablemente es solo ruido.

---

Con **datos anuales** y sin estacionalidades dentro del año:

❌ **Prophet se ajusta de más** y comete más errores.  
✅ **ARIMA**, en cambio:

- ✔️ Es más **sencillo**,
- ✔️ Usa menos **supuestos**,
- ✔️ Y ofrece **mejores pronósticos** para este tipo de series.

🎯 Por estas razones, **ARIMA es el modelo más adecuado** en este contexto.
""")
