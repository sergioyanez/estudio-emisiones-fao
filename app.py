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

# Opcional: desactiva warnings excesivos
import warnings; warnings.filterwarnings("ignore")

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

products_before_1990 = set(df_cleaned[df_cleaned['Año'] < '1990']['Producto'].unique())
products_after_1990 = set(df_cleaned[df_cleaned['Año'] >= '1990']['Producto'].unique())

products = products_before_1990 & products_after_1990
new_products = products_after_1990 - products_before_1990

st.subheader("Comparación de productos por período")

st.write(f"📦 Productos antes de 1990: {len(products_before_1990)}")
st.write(f"📦 Productos después de 1990: {len(products_after_1990)}")
st.write(f"🔁 Productos comunes: {len(products)}")
st.write(f"🆕 Productos nuevos desde 1990: {len(new_products)}")



st.markdown("### Delimitación temporal del análisis  \n"
            "Debido a los cambios estructurales observados en la cobertura geográfica y temática del dataset, se ha decidido restringir el análisis a los datos disponibles **a partir del año 1990**."
            "Esta decisión responde a dos razones principales:  \n"
            "- **Mayor cobertura geográfica**: a partir de 1990 se incorporan 52 nuevos países, alcanzando un total de 238. Esto garantiza que los análisis comparativos entre regiones y países no estén sesgados por datos ausentes en décadas anteriores.  \n"
            "- **Mayor cobertura temática**: el número de productos reportados aumenta de 16 (antes de 1990) a más de 40 (después), lo que introduce una mejora en el detalle metodológico, pero también limita la comparabilidad histórica.  \n"
            "### Justificación  \n"
            "Trabajar con el subconjunto de datos posterior a 1990 permite realizar análisis **más consistentes, representativos y comparables** reduciendo el riesgo de conclusiones erróneas causadas por diferencias de cobertura y disponibilidad de información."
            "En consecuencia, **todas las visualizaciones y estadísticas agregadas en este informe se basarán en datos desde 1990 a 2025, por lo cual no vamos a tener en cuenta estimaciones futuras**."
)

df_completed = df_cleaned.copy()
df_01 = df_cleaned[(df_cleaned['Año'] >= '1990') & (df_cleaned['Año'] <= '2025')].copy()

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
    (df_fao['Año'].between('2017', '2022')) &
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
    [['2019','2020', '2021','2022']]            # columnas de interés
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

# Saca las horas, minutos y segundo, sólo deja el año
pivot = pivot.rename_axis('Año').reset_index()
pivot['Año'] = pivot['Año'].dt.year
pivot.set_index('Año', inplace=True)


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

df_dec = df_emissions_by_continent_year.copy()

# Convertir 'Año' a datetime si no lo está
if not pd.api.types.is_datetime64_any_dtype(df_dec['Año']):
    df_dec['Año'] = pd.to_datetime(df_dec['Año'], errors='coerce')

# Extraer década como número
df_dec['Década'] = df_dec['Año'].dt.year // 10 * 10

# Excluir 'Mundo'
df_dec = df_dec[df_dec['Área'] != 'Mundo']

# Agrupar
pivot_dec = (
    df_dec.groupby(['Década','Área'])['Valor_Gt']
          .mean()
          .reset_index()
)

# Estilo
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(data=pivot_dec,
            x='Década', y='Valor_Gt', hue='Área', palette='Set2', ax=ax)

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

Estos componentes agrupados representan las Emisiones Totales incluyendo LULUCF. Al analizarlos por separado, podemos definir con precisión qué porción de las emisiones proviene de la finca, de la conversión de ecosistemas o de la cadena de suministro, lo cual es información importante para definir politicas eficaces en cada región.
""")



# ---------- CONVERSIÓN DE DATOS ----------
df_fao['Año'] = pd.to_datetime(df_fao['Año'], errors='coerce').dt.year
df_fao['Elemento'] = df_fao['Elemento'].astype(str)

# ---------- PARÁMETROS ----------
regions = ['Américas', 'Asia', 'Europa', 'Oceanía', 'África']
products = ["Farm gate", "Cambios de uso de la tierra", "Pre y\xa0post-producción"]
gas = "CO2eq"
years = [1990, 2010, 2022]

# ---------- FILTRADO ----------
df_products_continents = df_fao[
    df_fao['Producto'].isin(products) &
    df_fao['Área'].isin(regions) &
    df_fao['Año'].isin(years) &
    df_fao['Elemento'].str.contains(gas, case=False, na=False)
].copy()

# ---------- PIVOTEO ----------
pivot = (
    df_products_continents
    .pivot_table(index=['Año', 'Área'],
                 columns='Producto',
                 values='Valor_Gt',
                 aggfunc='sum')
    .sort_index(level=1)
    .reset_index()
    .sort_values(['Área', 'Año'], ascending=[True, False])
    .reset_index(drop=True)
)

if pivot.empty:
    st.warning("No hay datos disponibles con los filtros actuales.")
else:
    colors = ['#0066CC', '#0eca1c', '#ff5733']
    bar_h = 0.8
    gap = 1
    offset = bar_h
    n_y = len(years)
    y_pos = []
    for g in range(len(regions)):
        base = offset + g*(n_y*bar_h + gap)
        y_pos.extend(base + np.arange(n_y)*bar_h)

    fig, ax = plt.subplots(figsize=(10, 7))
    left = np.zeros(len(y_pos))

    for (col, color) in zip(products, colors):
        if col in pivot.columns:
            ax.barh(y_pos, pivot[col], left=left,
                    height=bar_h, color=color, edgecolor='white', label=col)
            left += pivot[col].fillna(0).values
        else:
            st.warning(f"'{col}' no está presente en los datos filtrados.")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot['Año'])
    ax.set_axisbelow(True)
    ax.grid(axis='y')

    ax2 = ax.twinx()
    mid_pos = [offset*0.1 + g*(n_y*bar_h + gap) + (n_y*bar_h) - bar_h
               for g in range(len(regions))]
    ax2.set_yticks(mid_pos)
    ax2.set_yticklabels(regions, fontsize=15, weight='bold')

    ax2.yaxis.set_ticks_position('left')
    ax2.spines['left'].set_position(('outward', 70))
    ax2.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(False)

    ax.set_xlabel('Gt CO₂-eq (AR5)')
    ax.set_title('Total Emisiones incluyendo LULUCF CO₂-eq por componente 1990 · 2010 · 2022')
    ax.legend(title='Producto',
              loc='upper right',
              frameon=True,
              framealpha=.9,
              borderpad=.6, fontsize=9)
    plt.tight_layout()
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
sns.barplot(data=df_top_countries_emission,x='Área', y='Valor', hue='Área',
            palette='Greens_r', ax=axs[0], legend=False)
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

fig.suptitle('Top 10 Paises Emisiones (CO2eq) (AR5) (2022)', fontsize=16, y=1.05)
st.pyplot(fig)
st.markdown("""Interpretación:

- China	Con aproximadamente 14 millones de kt lidera con enorme ventaja ( 2,5 × EEUU).
- EE. UU: es el segundo país con mas emisiones en el mundo pero, aun así emite menos de la mitad que China.
- India: se consolida en el tercer lugar, reflejando crecimiento poblacional.
- Rusia, Indonesia, Brasil, Japón, Irán: tienen valores intermedios (0,6 – 2 millones kt). Mezcla de grandes potencias agrícolas (Brasil, Indonesia) y economías industriales/energéticas (Rusia, Irán, Japón).
- Arabia Saudita y México	ocupan el puesto 9 y 10 el ranking. Sus emisiones son <10 % de las chinas.

Se puede observar una desigualdad extrema: el primer país (China) emite casi 75 veces más que el décimo.

Además, en el gráfico de la derecha podemos observar que, en la actualidad, dos tercios de todas las emisiones agro‑alimentarias se concentran en solo diez países (63%). Por lo tanto, el resto de los paises (180 aprox) aportan el otro tercio.
El gráfico demuestra que las politicas de mitigación global deben hacer foco en unas pocas jurisdicciones. Sin acciones contundentes en esos países, el resto del mundo difícilmente compensará el volumen de emisiones que ahí se genera.""")


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

st.markdown("# Modelo predictivo")

serie_america = df_cleaned[
    (df_cleaned['Área'] == 'Américas') &
    (df_cleaned['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_cleaned['Código del producto'] == 6825)
    ]
serie_asia = df_cleaned[
    (df_cleaned['Área'] == 'Asia') &
    (df_cleaned['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_cleaned['Código del producto'] == 6825)
    ]

serie_europa = df_cleaned[
    (df_cleaned['Área'] == 'Europa') &
    (df_cleaned['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_cleaned['Código del producto'] == 6825)
    ]
serie_oceania = df_cleaned[
    (df_cleaned['Área'] == 'Oceania') &
    (df_cleaned['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_cleaned['Código del producto'] == 6825)
    ]

serie_africa = df_cleaned[
    (df_cleaned['Área'] == 'África') &
    (df_cleaned['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
    (df_cleaned['Código del producto'] == 6825)
    ]

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


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Aplanar para acceder con índice simple

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

# Eliminar el sexto subplot (vacío)
fig.delaxes(axes[5])

plt.tight_layout()
st.pyplot(plt)

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

# ────── Supongamos que estas series ya están definidas ──────
# Asegurate de definirlas antes de correr esta app
series_continentales = {
    'América': serie_america['Valor'],
    'Asia': serie_asia['Valor'],
    'Europa': serie_europa['Valor'],
    'Oceanía': serie_oceania['Valor'],
    'África': serie_africa['Valor'],
}

# ────── Interfaz ──────
st.title("Test KPSS de Estacionaridad")
st.write("Análisis de estacionaridad con regresión lineal y término constante")

alpha = 0.05  # Nivel de significancia

# Botón para iniciar análisis
if st.button("Ejecutar test KPSS"):
    for nombre, serie in series_continentales.items():
        st.subheader(f"{nombre}")
        serie = serie.dropna()

        if len(serie) < 3:
            st.info("Serie vacía o muy corta, se omite.")
            continue

        try:
            stat, p, lags, crit = kpss(serie, regression='ct')
            st.write(f"**KPSS stat** = {stat:.3f} | **p** = {p:.3f} | **lags** = {lags}")
            if p < alpha:
                st.error("**NO estacionaria** (se rechaza H₀)")
            else:
                st.success("Sin evidencia contra la estacionaridad (no se rechaza H₀)")
        except Exception as e:
            st.warning(f"Error al procesar {nombre}: {e}")

st.markdown("""### Resultados de las Pruebas de Estacionariedad:

- América: tanto la prueba ADF como KPSS coinciden en que la serie no es estacionaria. Tratamiento: vamos a realizar una diferenciación (d = 1).
- Asia: existe un conflicto leve entre las pruebas. ADF rechaza la estacionariedad, mientras KPSS no la rechaza. Tratamiento: vamos a partir de una primera diferenciación y hacer pruebas.
- Europa: conflicto. KPSS indica que no hay estacionariedad mientras ADF rechaza la H0, indicando lo contrario. Tratamiento: vamos a hacer pruebas luego de una primera diferenciación.
- Oceanía: las pruebas se contradicen. Tratamiento: d = 1.
- África: ADF concluye que la serie no es estacionaria y KPSS no tiene evidencia contra la estacionariedad. Tratamiento: primera diferenciación.""")

st.markdown("### 🔍 Estacionariedad y diferenciación para modelado ARIMA")

series_continentales = {
    'América': serie_america,
    'Asia': serie_asia,
    'Europa': serie_europa,
    'Oceanía': serie_oceania,
    'África': serie_africa,
}


def test_adf(serie):
    result = adfuller(serie.dropna())
    adf_stat, p_value, _, _, critical_values, _ = result
    return adf_stat, p_value, critical_values


# Resultados para tabla
resultados_adf = []

for nombre, df in series_continentales.items():
    st.markdown(f"#### 🌎 {nombre}")

    if 'Valor' not in df or df['Valor'].dropna().size < 3:
        st.warning("⚠️ Serie vacía o con muy pocos datos.")
        resultados_adf.append({
            "Región": nombre,
            "ADF original": "–",
            "p-value original": "–",
            "Estacionaria original": "No evaluada",
            "ADF diferenciada": "–",
            "p-value diferenciada": "–",
            "Estacionaria diferenciada": "No evaluada",
            "Diferencias necesarias (d)": "–"
        })
        continue

    serie = df['Valor'].dropna()

    # ADF original
    adf_stat_orig, pval_orig, _ = test_adf(serie)
    estacionaria_orig = "Sí" if pval_orig < 0.05 else "No"

    st.markdown(f"- ADF original: `{adf_stat_orig:.4f}`, p-value: `{pval_orig:.4f}`")
    st.markdown(f"→ ¿Es estacionaria? **{'✅ Sí' if pval_orig < 0.05 else '🚫 No'}**")

    # Si es estacionaria, no se diferencia
    if pval_orig < 0.05:
        resultados_adf.append({
            "Región": nombre,
            "ADF original": round(adf_stat_orig, 4),
            "p-value original": round(pval_orig, 4),
            "Estacionaria original": "Sí",
            "ADF diferenciada": "–",
            "p-value diferenciada": "–",
            "Estacionaria diferenciada": "–",
            "Diferencias necesarias (d)": 0
        })
        continue

    # Diferenciar y volver a testear
    df['Valor_diff'] = df['Valor'].diff()
    serie_diff = df['Valor_diff'].dropna()

    if serie_diff.size < 3:
        st.warning("⚠️ No hay suficientes datos tras diferenciar.")
        resultados_adf.append({
            "Región": nombre,
            "ADF original": round(adf_stat_orig, 4),
            "p-value original": round(pval_orig, 4),
            "Estacionaria original": "No",
            "ADF diferenciada": "–",
            "p-value diferenciada": "–",
            "Estacionaria diferenciada": "No evaluada",
            "Diferencias necesarias (d)": "?"
        })
        continue

    adf_stat_diff, pval_diff, _ = test_adf(serie_diff)
    estacionaria_diff = "Sí" if pval_diff < 0.05 else "No"

    st.markdown(f"- ADF diferenciada: `{adf_stat_diff:.4f}`, p-value: `{pval_diff:.4f}`")
    st.markdown(f"→ ¿Es estacionaria tras diferenciar? **{'✅ Sí' if pval_diff < 0.05 else '🚫 No'}**")

    resultados_adf.append({
        "Región": nombre,
        "ADF original": round(adf_stat_orig, 4),
        "p-value original": round(pval_orig, 4),
        "Estacionaria original": "No",
        "ADF diferenciada": round(adf_stat_diff, 4),
        "p-value diferenciada": round(pval_diff, 4),
        "Estacionaria diferenciada": estacionaria_diff,
        "Diferencias necesarias (d)": 1 if pval_diff < 0.05 else "≥2"
    })

# Mostrar tabla resumen
st.markdown("### 📋 Resumen de diferenciación requerida para ARIMA")
df_adf_resumen = pd.DataFrame(resultados_adf)
st.dataframe(df_adf_resumen, use_container_width=True)

st.markdown("""
---

### 🧠 ¿Por qué se realiza esta prueba?

Antes de aplicar un modelo ARIMA, es necesario trabajar con series **estacionarias**, es decir, series cuya media y varianza se mantienen constantes en el tiempo.

El **test de Dickey-Fuller aumentado (ADF)** permite verificar si una serie:

- 🔹 **Ya es estacionaria** → se puede modelar directamente (ARIMA con `d = 0`).
- 🔹 **No es estacionaria** → requiere ser **diferenciada** (restar cada valor con el anterior) para eliminar tendencia.

---

### ⚙️ ¿Qué significa diferenciar una serie?

Diferenciar una serie es transformar los valores absolutos en **cambios entre periodos consecutivos**. Esto permite:

- Eliminar la tendencia creciente o decreciente.
- Hacer que la serie fluctúe alrededor de una media constante.
- Lograr que el test ADF detecte estacionariedad en la nueva serie.

---

### 📌 Conclusión

Con esta prueba determinamos el parámetro `d` que cada serie necesita en el modelo ARIMA.  
Si una serie no se vuelve estacionaria ni con la primera diferencia (`d = 1`), puede requerir transformaciones adicionales (`d ≥ 2`) o un enfoque diferente como modelado no lineal.
""")

# Crear figura
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Lista de DataFrames y nombres
series_diff = [
    ('América', serie_america),
    ('Asia', serie_asia),
    ('Europa', serie_europa),
    ('Oceanía', serie_oceania),
    ('África', serie_africa),
]

# Graficar cada serie
for i, (nombre, df) in enumerate(series_diff):
    if 'Valor_diff' in df and df['Valor_diff'].dropna().size >= 3:
        sns.lineplot(x=df.index, y=df['Valor_diff'], ax=axes[i])
        axes[i].set_title(f'{nombre} - Valor diferenciado')
        axes[i].set_ylabel('Δ Valor')
    elif 'Valor' in df and df['Valor'].dropna().size >= 3:
        sns.lineplot(x=df.index, y=df['Valor'], ax=axes[i])
        axes[i].set_title(f'{nombre} - Serie original (estacionaria)')
        axes[i].set_ylabel('Valor')
    else:
        axes[i].set_title(f'{nombre} - Sin datos')
        axes[i].axis('off')

    axes[i].set_xlabel('Año')

# Eliminar subplot vacío (el sexto)
fig.delaxes(axes[5])

plt.tight_layout()
st.pyplot(fig)

st.markdown("""### Tras quitar la tendencia (diferenciación de primer orden), los valores ya no muestran crecimiento sistemático. Oscilan alrededor de cero.

1. América – Valor diferenciado
    Pico negativo profundo (~1965–1970) y pico positivo después

      Posibles causas:
    
      * Crisis del petróleo de los años 70: aunque afectó más a los países industrializados, en América Latina produjo cambios abruptos en consumo de energía.
    
      * Desindustrialización parcial en algunos países y reformas estructurales.
    
      * Regulación ambiental inicial en EE.UU. con la creación de la EPA (1970), que marcó una reducción en emisiones industriales.
    
      * Volatilidad macroeconómica en Sudamérica (hiperinflación, dictaduras, caídas del PIB) también puede explicar estos saltos abruptos.

2. Asia – Valor diferenciado
    Muchos picos grandes desde ~1980 hasta 2000+

      Posibles causas:
    
      * Aceleración de la industrialización china desde la apertura de Deng Xiaoping en 1978.
    
      * Crecimiento de India a partir de los 90.
    
      * Urbanización masiva, expansión de infraestructura, transporte y consumo energético.
    
      * Cambios abruptos en políticas de producción energética (transiciones carbón → otras fuentes).
    
      * Alta variabilidad: Asia tiene países con dinámicas muy distintas (desde Japón y Corea hasta Indonesia y Pakistán).

3. Europa – Serie original (estacionaria)
    Tendencia clara, sin picos grandes post-diferenciación porque no se aplicó

    Tendencia decreciente sostenida:

    * Desindustrialización y terciarización de la economía desde los 80.

    * Leyes ambientales fuertes desde el Protocolo de Kioto (1997) y Pacto Verde Europeo.

    * Reducción del uso de carbón y transición energética más temprana que en otras regiones.

4. Oceanía – Sin datos
    No se puede evaluar, pero si se completa luego, podríamos investigar:
    Australia como emisor dominante (por minería, carbón).
    Posibles picos: políticas ambientales, sequías, incendios forestales, tratados.

5. África – Valor diferenciado
    Picos dispersos, sin tendencia clara, pero visibles oscilaciones

      Posibles causas:
      * Variabilidad en consumo energético sin un patrón homogéneo: muchos países dependen de biomasa, con bajo uso industrial.
    
      * Países con extracción de petróleo (Nigeria, Angola) pueden provocar saltos cuando abren/cambian producción.
    
      * Conflictos armados (e.g., guerras civiles, crisis políticas) afectan bruscamente la actividad económica y por ende las emisiones.
      * Crecimiento poblacional sin desarrollo industrial intensivo: no hay un patrón de aumento estable como en Asia.


""")



st.markdown("### 🔁 Test de estacionariedad sobre series diferenciadas (d = 1)")

series_diff = [
    ('América', serie_america),
    ('Asia', serie_asia),
    ('Europa', serie_europa),
    ('Oceanía', serie_oceania),
    ('África', serie_africa),
]

# Tabla de resultados
resultados_diff = []

for nombre, df in series_diff:
    st.markdown(f"#### 🌍 {nombre}")

    if 'Valor_diff' not in df.columns:
        st.warning("⚠️ No tiene columna 'Valor_diff'. No se puede evaluar.")
        resultados_diff.append({
            "Región": nombre,
            "ADF Differenced": "–",
            "p-value": "–",
            "Estacionaria (d=1)": "No evaluada"
        })
        continue

    serie = df['Valor_diff'].dropna()
    if len(serie) < 3:
        st.warning("⚠️ Serie diferenciada con pocos datos.")
        resultados_diff.append({
            "Región": nombre,
            "ADF Differenced": "–",
            "p-value": "–",
            "Estacionaria (d=1)": "No evaluada"
        })
        continue

    try:
        result_diff = adfuller(serie)
        adf_stat = result_diff[0]
        pval = result_diff[1]
        crit_vals = result_diff[4]
        es_estacionaria = "Sí" if pval < 0.05 else "No"

        st.markdown(f"- ADF Statistic: `{adf_stat:.4f}`")
        st.markdown(f"- p-value: `{pval:.4f}`")
        st.markdown("- Valores críticos:")
        for key, val in crit_vals.items():
            st.markdown(f"  - {key}: `{val:.4f}`")

        if pval < 0.05:
            st.success("✅ La serie diferenciada es **estacionaria** (se rechaza H₀)")
        else:
            st.error("🚫 La serie diferenciada **NO es estacionaria** (no se rechaza H₀)")

        resultados_diff.append({
            "Región": nombre,
            "ADF Differenced": round(adf_stat, 4),
            "p-value": round(pval, 4),
            "Estacionaria (d=1)": es_estacionaria
        })

    except Exception as e:
        st.error(f"⚠️ Error al procesar: {e}")
        resultados_diff.append({
            "Región": nombre,
            "ADF Differenced": "Error",
            "p-value": "Error",
            "Estacionaria (d=1)": "Error"
        })

# Mostrar tabla resumen
st.markdown("### 📋 Resumen: Estacionariedad tras 1 diferenciación")
df_resultados_diff = pd.DataFrame(resultados_diff)
st.dataframe(df_resultados_diff, use_container_width=True)

st.markdown("""
---

### 📌 ¿Por qué realizamos este análisis?

Una vez que determinamos que la serie original **no es estacionaria**, la primera estrategia para estabilizarla es **aplicar una diferenciación**: restar cada valor con su valor anterior.

Este paso elimina la tendencia y transforma la serie en una que fluctúe alrededor de una media constante.  
Pero no siempre es suficiente: por eso, debemos aplicar nuevamente el **test ADF** sobre la **serie diferenciada** para verificar si ya es apta para un modelo ARIMA.

---

### 🎯 ¿Qué buscamos?

Con este análisis buscamos determinar si:

- 🔹 **Una sola diferenciación (`d=1`)** ya basta → podemos usar ARIMA(…, **d=1**, …)
- 🔹 **Todavía no es estacionaria** → podría requerir una segunda diferenciación (`d=2`) o un modelo alternativo

Esto garantiza que el modelo ARIMA sea **válido y confiable**, ya que uno de sus supuestos clave es que la serie sea estacionaria.

---
""")

st.markdown("### 📈 Comparación: Serie original vs diferenciada (d=1)")

series_diff = [
    ('América', serie_america),
    ('Asia', serie_asia),
    ('Europa', serie_europa),
    ('Oceanía', serie_oceania),
    ('África', serie_africa),
]

for nombre, df in series_diff:
    st.markdown(f"#### 🌍 {nombre}")

    if 'Valor' not in df.columns or df['Valor'].dropna().size < 3:
        st.warning("⚠️ Serie original vacía o insuficiente.")
        continue

    if 'Valor_diff' not in df.columns or df['Valor_diff'].dropna().size < 3:
        st.warning("⚠️ Serie diferenciada vacía o no generada.")
        continue

    fig, ax = plt.subplots(figsize=(12, 4))

    # Graficar serie original
    ax.plot(df['Año'], df['Valor'], label='Serie original', color='steelblue')

    # Graficar serie diferenciada (alinea por año a partir del segundo punto)
    ax.plot(df['Año'], df['Valor_diff'], label='Serie diferenciada (d=1)', color='firebrick', linestyle='--')

    ax.set_title(f'{nombre} - Comparación de serie original vs diferenciada')
    ax.set_xlabel('Año')
    ax.set_ylabel('Emisiones (CO2eq)')
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

st.markdown("""
---

### 📊 Comparación visual: Serie original vs. Serie diferenciada

Para aplicar un modelo ARIMA válido, es necesario que las series sean **estacionarias**, es decir, que no presenten tendencia sostenida en el tiempo.

Una técnica común para lograr esto es la **diferenciación**, que consiste en restar cada valor con el anterior. Esto transforma una serie creciente o decreciente en una que **oscila alrededor de una media estable**, idealmente cercana a cero.

---

### 🎯 ¿Qué muestran estos gráficos?

Cada gráfico compara:

- 📘 **Serie original** (línea azul): representa los valores absolutos de emisiones a lo largo del tiempo.
- 🔴 **Serie diferenciada** (línea roja punteada): representa los **cambios entre años consecutivos**.

---

### 🧠 ¿Para qué sirve?

Visualizar ambas series permite:

- Confirmar si la **tendencia fue eliminada correctamente**.
- Ver si la serie diferenciada presenta una **fluctuación estable**, condición necesaria para que un modelo ARIMA con `d=1` sea válido.
- Detectar visualmente **outliers o variaciones bruscas** que podrían requerir un tratamiento adicional.

---
""")

st.markdown("### Como ya se logran que las series sean estacionarias (tomo d= 1 para Américas, Asia, África y Oceanía, y d= 0 para Europa que ya era estacionaria)"
            "luego calculo  ACF para obtener el valor de q y PACF para obtener el valor de p ")

st.markdown("## ACF y PACF - Series diferenciadas por región")


series_diff = [
    ('América', serie_america),
    ('Asia', serie_asia),
    ('Europa', serie_europa),
    ('Oceanía', serie_oceania),
    ('África', serie_africa),
]

for nombre, df in series_diff:
    st.markdown(f"### {nombre}")

    if 'Valor_diff' not in df:
        st.warning("⚠️ No tiene columna `Valor_diff`, se omite.")
        continue

    serie = df['Valor_diff'].dropna()
    if len(serie) < 3:
        st.warning("⚠️ Muy pocos datos para mostrar ACF/PACF confiables.")
        continue

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'ACF y PACF - {nombre} (Valor diferenciado)', fontsize=14)

    plot_acf(serie, lags=15, ax=ax[0])
    plot_pacf(serie, lags=15, ax=ax[1])

    ax[0].set_title('ACF')
    ax[1].set_title('PACF')

    plt.tight_layout()
    st.pyplot(fig)


st.markdown("""
### ¿Por qué analizamos ACF y PACF?

Una vez que la serie ha sido diferenciada para volverla estacionaria, se analizan los patrones de autocorrelación:

- **ACF (Autocorrelation Function)**: muestra cuánto se relaciona cada valor con sus rezagos. Sirve para sugerir el parámetro `q` del modelo ARIMA (componente de media móvil).
- **PACF (Partial ACF)**: muestra la correlación con rezagos directos, eliminando la influencia de intermedios. Se usa para determinar el parámetro `p` (componente autorregresiva).

Estos gráficos nos permiten **identificar el orden adecuado del modelo ARIMA (p, d, q)** observando en qué rezagos se cortan las correlaciones.

> En general:
> - Si ACF se corta bruscamente en rezago k ⇒ `q = k`
> - Si PACF se corta bruscamente en rezago k ⇒ `p = k`
""")


st.markdown("""
### ¿Cómo se eligen los parámetros `p` y `q` para ARIMA?

ACF (Autocorrelation Function)
    El primer lag es significativamente distinto de cero, luego se corta.
    Esto sugiere un componente MA (q) de 1.

PACF (Partial ACF)
    El primer lag también es significativamente distinto de cero, y luego se corta.
    Esto sugiere un componente AR (p) de 1.

Como ya aplicamos un diff es sugerido un componente (d) de 1

Modelo candidato: ARIMA(p=1, d=1, q=1)
""")
st.markdown("""ANÁLISIS REGIÓN POR REGIÓN

  América

    ACF: lag 1 claramente significativo → q = 1

    PACF: lag 1 claramente significativo → p = 1

    Sugerencia: ARIMA(1, 1, 1)

  Asia

    ACF: lag 1 es el único fuera del azul → q = 1

    PACF: lag 1 también destaca → p = 1

    Sugerencia: ARIMA(1, 1, 1)

  Oceanía

    ACF: lags 2 y 5 sobresalen → probar q = 2 o q = 5

    PACF: lag 5 y lag 10 visibles → p = 5 o p = 10

    Serie más ruidosa, se puede empezar con ARIMA(5, 1, 2) y afinar.

  África
  
    ACF: lag 1 y lag 7 apenas sobresalen → q = 1

    PACF: lag 5 y 6 aparecen débiles, pero no demasiado claros → p = 1 o p = 2

    Sugerencia conservadora: ARIMA(1, 1, 1) y si no da buen resultado, probar con ARIMA(2, 1, 1)
""")


#################################################################################

# acá deberia calular el AIC pero no puedo

# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=ValueWarning)
#
# p = q = range(0,3)
# d = 1
# pdq = list(itertools.product(p,[d],q))
# resultados = {}
#
# for nombre, serie in series_continentales.items():
#     best_aic   = np.inf
#     best_order = None
#
#     for order in pdq:
#         try:
#             res = SARIMAX(serie, order=order,
#                           enforce_stationarity=False,
#                           enforce_invertibility=False
#                          ).fit(disp=False)
#             if res.aic < best_aic:
#                 best_aic, best_order = res.aic, order
#         except Exception:
#             continue
#
#     resultados[nombre] = {'order': best_order, 'AIC': best_aic}
#
# # — Muestra el resumen por pantalla
# for k, v in resultados.items():
#     print(f"{k:8s}  ->  (p,d,q) = {v['order']}   |  AIC = {v['AIC']:.2f}")

#############################################################################################3


# ───── Suprimir warnings ─────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

# ───── Parámetros ARIMA y SARIMA ─────
parametros_arima = {
    'América': (0, 1, 1),
    'Asia': (1, 1, 2),
    'Europa': (1, 1, 2),
    'Oceanía': (0, 1, 2),
    'África': (2, 1, 2)
}

parametros_estacionales = {
    'América': [(1, 0, 1, 4), (1, 1, 1, 4)],
    'Asia': [(1, 1, 1, 4)],
    'Oceanía': [(1, 1, 1, 4)]
}

# ───── Series por continente (ya definidas) ─────
series_dict = {
    'América': serie_america,
    'Asia': serie_asia,
    'Europa': serie_europa,
    'Oceanía': serie_oceania,
    'África': serie_africa
}

# ───── Ajustar modelos y mostrar resultados ─────
st.title("📈 Ajuste de modelos ARIMA/SARIMA por continente")

for nombre, df in series_dict.items():
    st.subheader(f"🌍 {nombre}")
    y = df['Valor'].dropna()

    if len(y) < 10:
        st.warning("⚠️ Muy pocos datos para ajustar el modelo.")
        continue

    try:
        p, d, q = parametros_arima[nombre]

        if nombre in parametros_estacionales:
            mejores_resultados = None
            mejor_aic = float('inf')

            for (P, D, Q, s) in parametros_estacionales[nombre]:
                st.markdown(f"Probar SARIMA({p},{d},{q})×({P},{D},{Q},{s})")

                model = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)

                st.write(f"AIC = {res.aic:.2f}")
                if res.aic < mejor_aic:
                    mejor_aic = res.aic
                    mejores_resultados = res

            if mejores_resultados:
                st.success("✅ Mejor modelo elegido")
                st.text(mejores_resultados.summary())

                fig = mejores_resultados.plot_diagnostics(figsize=(10, 6))
                fig.suptitle(f'Diagnóstico de residuos - {nombre}', fontsize=14)
                st.pyplot(fig)

        else:
            model = SARIMAX(
                y,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)

            st.success("✅ Modelo ajustado (ARIMA)")
            st.text(res.summary())

            fig = res.plot_diagnostics(figsize=(10, 6))
            fig.suptitle(f'Diagnóstico de residuos - {nombre}', fontsize=14)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error al ajustar modelo para {nombre}: {e}")




st.markdown("""
## Evaluación general por región

Luego de probar varios modelos distintos concluímos que éstos últimos podrían ser los más indicados a utilizar

| Continente  | Modelo elegido      | AIC        | JB test (Normalidad) | ¿Modelo útil?               |
| ----------- | ------------------- | ---------- | -------------------- | --------------------------- |
| **América** | `(0,1,1)x(1,1,1,4)` | **610.47** |  JB=8.73 (p=0.01)   |  Sí |
| **Asia**    | `(1,1,2)x(1,1,1,4)` | 598.03     |  JB=0.38 (p=0.83)   |  Sí                       |
| **Europa**  | `(1,1,2)`           | 778.23     |  JB=0.42 (p=0.81)   |  Aceptable pero débil     |
| **Oceanía** | `(0,1,2)x(1,1,1,4)` | 501.76     |  JB=1.42 (p=0.49)   |  Sólido                   |
| **África**  | `(2,1,2)`           | 723.59     |  JB=2.96 (p=0.23)   |  Aceptable                |
""")







st.markdown("""### 🔍 PREDICCIONES SEGÚN MODELOS CONSIDERADOS""")


# ────── Suprimir warnings ──────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

# ────── Series por continente ──────
series_dict = {
    'América': serie_america,
    'Asia': serie_asia,
    'Europa': serie_europa,
    'Oceanía': serie_oceania,
    'África': serie_africa
}

# ────── Configuración de modelos finales ──────
modelos_config = {
    'América':  {'order': (0, 1, 1), 'seasonal_order': (1, 1, 1, 4)},
    'Asia':     {'order': (1, 1, 2), 'seasonal_order': (1, 1, 1, 4)},
    'Europa':   {'order': (1, 1, 2), 'seasonal_order': None},
    'Oceanía':  {'order': (0, 1, 2), 'seasonal_order': (1, 1, 1, 4)},
    'África':   {'order': (2, 1, 2), 'seasonal_order': None}
}

# ────── Título de la app ──────
st.title("📈 Pronóstico SARIMA por continente")
st.markdown("## Predicción de los próximos 5 años usando modelos configurados manualmente.")

# ────── Loop de pronóstico ──────
for nombre, df in series_dict.items():
    if nombre ==  'Oceanía':
        continue
    else:
        st.subheader(f"🌍 {nombre}")

        try:
            # Verificar que haya datos y columnas necesarias
            if df.empty or 'Año' not in df.columns or 'Valor' not in df.columns:
                st.warning(f"{nombre}: Datos incompletos o vacíos.")
                continue

            # Asegurar que 'Año' esté en formato datetime
            if not np.issubdtype(df['Año'].dtype, np.datetime64):
                df['Año'] = pd.to_datetime(df['Año'].astype(str), format='%Y')

            serie = df.set_index('Año')['Valor'].dropna()

            # Para todos excepto Oceanía: exigir mínimo de datos
            if len(serie) < 10 and nombre != 'Oceanía':
                st.warning(f"{nombre}: Muy pocos datos para modelar.")
                continue

            if len(serie) < 10 and nombre == 'Oceanía':
                st.warning(f"⚠️ {nombre}: Se fuerza el modelo con pocos datos ({len(serie)} valores).")

            # Obtener configuración del modelo
            config = modelos_config[nombre]
            order = config['order']
            seasonal_order = config.get('seasonal_order', (0, 0, 0, 0))

            # Ajustar modelo
            model = SARIMAX(
                serie,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)

            # Pronóstico
            forecast = res.get_forecast(steps=5)
            pred = forecast.predicted_mean
            conf_int = forecast.conf_int()

            # Índice para años futuros
            last_year = serie.index[-1].year
            pred.index = pd.date_range(start=f'{last_year + 1}', periods=5, freq='Y')
            conf_int.index = pred.index

            # Graficar
            fig, ax = plt.subplots(figsize=(10, 5))
            serie.plot(ax=ax, label='Histórico', color='blue', linewidth=2)
            pred.plot(ax=ax, label='Pronóstico', color='red', linewidth=2)
            ax.fill_between(pred.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)

            # Estética general achicada
            ax.set_title(f'Predicción con intervalo de confianza - {nombre}', fontsize=8)
            ax.set_xlabel('Año', fontsize=7)
            ax.set_ylabel('Valor estimado', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
            ax.legend(fontsize=6, loc='upper left')
            ax.grid(True, linewidth=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error al procesar {nombre}: {e}")

series_dict2 = {}

# Recorrer cada continente y construir series temporales
for cont in continents:
    df_serie = df_fao[
        (df_fao['Área'] == cont) &
        (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
        (df_fao['Código del producto'] == 6825)
    ].copy()

    # Crear serie temporal completa con índice anual
    y = df_serie.set_index(pd.PeriodIndex(df_serie['Año'], freq='Y'))['Valor']

    # División: entrenamiento vs validación
    y_train = y[:'2018']
    y_test  = y['2018':]

    series_dict2[cont] = {
        'y_full': y,
        'y_train': y_train,
        'y_test': y_test
    }

if 'Oceanía' in series_dict2:
    try:
        # Extraer la serie temporal y eliminar nulos
        datos = series_dict2['Oceanía']
        serie = datos['y_full'].dropna()

        if len(serie) < 3:
            st.warning("⚠️ Oceanía: muy pocos datos para ajustar el modelo.")
        else:
            st.info(f"🔢 Datos cargados: {len(serie)}")

            # Configuración del modelo (puede adaptarse)
            order = (0, 1, 2)
            seasonal_order = (1, 1, 1, 4)

            # Ajustar modelo
            model = SARIMAX(
                serie,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)

            # Pronóstico
            forecast = res.get_forecast(steps=5)
            pred = forecast.predicted_mean
            conf_int = forecast.conf_int()

            # Fechas futuras
            last_year = serie.index[-1].year
            pred.index = pd.date_range(start=f'{last_year + 1}', periods=5, freq='Y')
            conf_int.index = pred.index

            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 5))
            serie.plot(ax=ax, label='Histórico', color='blue', linewidth=2)
            pred.plot(ax=ax, label='Pronóstico', color='red', linewidth=2)
            ax.fill_between(pred.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)

            # Estética general achicada
            ax.set_title(f'Predicción - con intervalo de confianza Oceanía', fontsize=8)
            ax.set_xlabel('Año', fontsize=7)
            ax.set_ylabel('Valor estimado', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
            ax.legend(fontsize=6, loc='upper left')
            ax.grid(True, linewidth=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error al procesar Oceanía: {e}")

st.markdown("""
## 🌍 Análisis comparativo de proyecciones por continente (SARIMAX con tendencia lineal)

A continuación se presenta una lectura interpretativa de los pronósticos de emisiones de CO₂eq (AR5) por continente, modelados con SARIMAX sin estacionalidad y con tendencia lineal.

---

### 🌎 **Américas**
- 📉 **Tendencia histórica**: Aumento hasta aproximadamente 2010 y luego descenso sostenido.
- 🔮 **Pronóstico**: El modelo predice una **caída continua** en las emisiones hasta 2043.
- ✅ Buen ajuste, con **IC relativamente acotado**, lo que sugiere confianza razonable en la predicción.

---

### 🌍 **África**
- 📈 **Tendencia histórica**: Crecimiento constante y progresivo desde 1990.
- 🔮 **Pronóstico**: Se espera un **fuerte aumento** de emisiones en las próximas dos décadas.
- ⚠️ Zona crítica, ya que **no hay señales de desaceleración** en el modelo. El IC también se expande con el tiempo, indicando **incertidumbre creciente**.

---

### 🌐 **Europa**
- 📉 **Tendencia histórica**: Descenso constante en las emisiones desde 1990.
- 🔮 **Pronóstico**: Continúa la **tendencia descendente**, aunque con una leve curva de inflexión hacia el alza.
- ✅ Muy buen ajuste. Es el **modelo más robusto** entre todos los continentes, con **IC angosto y centrado**.

---

### 🌏 **Asia**
- 📈 **Tendencia histórica**: Crecimiento muy pronunciado, especialmente desde el año 2000.
- 🔮 **Pronóstico**: El modelo proyecta un **aumento constante** y fuerte en las emisiones.
- ⚠️ Aunque el modelo es consistente, el **IC amplio sugiere incertidumbre a largo plazo**. Zona preocupante por su peso global en emisiones.

---

### 🌊 **Oceanía**
- ⚖️ **Tendencia histórica**: Serie más errática, con variaciones y sin una tendencia clara.
- 🔮 **Pronóstico**: Ligero descenso, pero con **alta incertidumbre** (IC muy ancho).
- ⚠️ El modelo **tiene poca confianza en el futuro** de la serie debido a la falta de una tendencia fuerte.

---

### 📌 Conclusión general:
- **Europa y América** muestran trayectorias descendentes, lo que es positivo.
- **Asia y África** presentan **fuertes crecimientos proyectados**, lo cual representa un desafío urgente en términos de políticas climáticas.
- **Oceanía** tiene una proyección incierta debido a la **alta volatilidad histórica**.

""")



st.markdown("""
## 🔍 Preparación de series temporales para entrenamiento y validación

Para poder entrenar modelos de predicción y evaluar su desempeño, se dividieron las series de emisiones por continente en dos subconjuntos:

- **Serie completa (`y_full`)**: contiene todos los valores históricos desde 1961 (o el primer año disponible).
- **Entrenamiento (`y_train`)**: valores hasta el año **2018 inclusive**. Esta parte se utiliza para ajustar los modelos.
- **Validación (`y_test`)**: valores desde el año **2018 en adelante**, reservados para comparar con las predicciones y calcular el error (por ejemplo, MAPE).

Esta división es fundamental para asegurar que los modelos no estén viendo los datos del futuro durante el entrenamiento y así obtener una evaluación realista.

> La variable `series_dict` guarda todas las series por continente, lo que permite acceder fácilmente a sus versiones completas, de entrenamiento y de prueba.
""")



# Continentes a analizar
continents = ['Américas', 'África', 'Europa', 'Asia', 'Oceanía']
series_dict = {}

# Recorrer cada continente y construir series temporales
for cont in continents:
    df_serie = df_fao[
        (df_fao['Área'] == cont) &
        (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
        (df_fao['Código del producto'] == 6825)
    ].copy()

    # Crear serie temporal completa con índice anual
    y = df_serie.set_index(pd.PeriodIndex(df_serie['Año'], freq='Y'))['Valor']

    # División: entrenamiento vs validación
    y_train = y[:'2018']
    y_test  = y['2018':]

    series_dict[cont] = {
        'y_full': y,
        'y_train': y_train,
        'y_test': y_test
    }

# Ejemplo visualización rápida de una serie
st.markdown("### 📈 Últimos valores de la serie de entrenamiento (ejemplo: América)")
st.write(series_dict['Américas']['y_train'].tail())


st.markdown("""
### 📊 Análisis visual de las emisiones históricas por continente

El siguiente gráfico permite **explorar la evolución de las emisiones de gases de efecto invernadero (CO₂eq)** para cada región entre los años disponibles.

La serie se encuentra dividida en:

- 🟩 **Entrenamiento (verde)**: hasta 2018 inclusive. Se usa para ajustar modelos predictivos.
- 🔴 **Validación (rojo punteado)**: desde 2018 en adelante. Se compara contra los pronósticos.
- 🔵 **Serie completa (azul)**: toda la secuencia original.

Esto es clave para construir modelos confiables y medir su capacidad de predicción sin caer en sobreajuste.
""")


# Construcción de las series
for cont in continents:
    df_serie = df_fao[
        (df_fao['Área'] == cont) &
        (df_fao['Elemento'] == 'Emisiones (CO2eq) (AR5)') &
        (df_fao['Código del producto'] == 6825)
    ].copy()

    y = df_serie.set_index(pd.PeriodIndex(df_serie['Año'], freq='Y'))['Valor']

    y_train = y[:'2018']
    y_test  = y['2018':]

    series_dict[cont] = {
        'y_full': y,
        'y_train': y_train,
        'y_test': y_test
    }

# Visualización con selección de continente
st.markdown("## 🌍 Visualización de series históricas por continente")

continente_seleccionado = st.selectbox("Seleccioná un continente:", continents)

serie = series_dict[continente_seleccionado]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=serie['y_full'].index.to_timestamp(),
    y=serie['y_full'].values,
    mode='lines+markers',
    name='Serie completa',
    line=dict(color='steelblue')
))

fig.add_trace(go.Scatter(
    x=serie['y_train'].index.to_timestamp(),
    y=serie['y_train'].values,
    mode='lines',
    name='Entrenamiento',
    line=dict(color='green')
))

fig.add_trace(go.Scatter(
    x=serie['y_test'].index.to_timestamp(),
    y=serie['y_test'].values,
    mode='lines',
    name='Validación',
    line=dict(color='firebrick', dash='dash')
))

fig.update_layout(
    title=f"Serie temporal de emisiones – {continente_seleccionado}",
    xaxis_title="Año",
    yaxis_title="Emisiones CO₂eq (kt)",
    legend=dict(x=0, y=1),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### ℹ️ Evaluación de Modelos ARIMA con AIC

Para cada región se evaluaron múltiples combinaciones de modelos ARIMA (valores de p, d, q entre 0 y 2).

#### 📌 ¿Qué es el AIC?

El **Akaike Information Criterion (AIC)** mide la calidad del modelo ajustado penalizando la complejidad.  
Un valor de **AIC más bajo** implica un mejor equilibrio entre **ajuste a los datos** y **simplicidad del modelo**.

#### ⚙️ Estrategia aplicada:

- Se ajustaron **27 modelos distintos por región**.
- Se usó solo componente ARIMA simple (sin estacionalidad).
- Para cada modelo, se calculó el AIC.
- Se seleccionó el modelo con menor AIC como el mejor candidato.

A continuación, se muestran los modelos óptimos encontrados:
""")


# Parámetros a evaluar
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(0, 0, 0, 0)]  # sin estacionalidad

# Diccionario de mejores modelos por continente
best_arima_results = {}

st.markdown("### 🔍 Búsqueda de Mejor Modelo ARIMA por Continente")

for cont in series_dict:
    st.markdown(f"#### 🌍 {cont}")

    y_train = series_dict[cont]['y_train']
    best_score = np.inf
    best_params = None
    best_seasonal_params = None
    best_model = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    y_train,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = mod.fit(disp=False)

                if results.aic < best_score:
                    best_score = results.aic
                    best_params = param
                    best_seasonal_params = param_seasonal
                    best_model = results
            except:
                continue

    best_arima_results[cont] = {
        'best_model': best_model,
        'best_aic': best_score,
        'best_order': best_params,
        'best_seasonal_order': best_seasonal_params
    }

    st.markdown(f"""
    ✅ **Mejor modelo ARIMA**: {best_params}  
    📉 **AIC**: {best_score:.2f}  
    """)

st.markdown("""
### ℹ️ Análisis estadístico de modelos ARIMA

Una vez identificados los mejores modelos ARIMA para cada región según el criterio AIC, se realiza un análisis más profundo sobre los **coeficientes estimados** del modelo:

- Se evalúa si cada coeficiente es **estadísticamente significativo**, utilizando el **p-valor** del test z.
- Un coeficiente se considera significativo si su **p-valor < 0.05**.
- Si todos los coeficientes son significativos, el modelo es estadísticamente sólido.
- En caso contrario, puede indicar que ciertos términos del modelo (AR, MA, etc.) no aportan valor y podrían eliminarse o ajustarse.

A continuación, se presenta la tabla de coeficientes por continente y una advertencia si se detectan coeficientes no significativos.
""")


st.markdown("## 📊 Análisis estadístico de modelos ARIMA ajustados")

for cont in series_dict:
    st.markdown(f"### 🌍 {cont}")

    y_train = series_dict[cont]['y_train']
    order = best_arima_results[cont]['best_order']
    sorder = best_arima_results[cont]['best_seasonal_order']

    try:
        mod = sm.tsa.statespace.SARIMAX(
            y_train,
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = mod.fit(disp=False)

        # Extraer coeficientes y p-valores directamente
        coef = results.params
        pvals = results.pvalues
        df_stats = pd.DataFrame({'Coeficiente': coef, 'p-valor': pvals})

        st.dataframe(df_stats)

        # Evaluar significancia
        not_significant = df_stats[df_stats['p-valor'] > 0.05]
        if not not_significant.empty:
            st.warning("⚠️ Coeficientes no significativos detectados (p > 0.05):")
            st.dataframe(not_significant)
        else:
            st.success("✅ Todos los coeficientes son estadísticamente significativos (p < 0.05).")

    except Exception as e:
        st.error(f"❌ Error al ajustar modelo para {cont}: {e}")


st.markdown("""
## 📊 Análisis estadístico de modelos ARIMA ajustados

Se presentan los resultados del ajuste de modelos SARIMAX sobre las series de emisiones por continente.  
Se interpreta la significancia estadística de cada coeficiente usando un umbral clásico de **p < 0.05**.

---

### 🌎 Américas

- **ma.L1** = -0.7796 | p = 0.000 → ✅ Significativo. Fuerte componente MA(1).
- **ma.L2** = 0.2189 | p = 0.066 → ⚠️ No significativo. No se justifica el segundo rezago.
- **sigma2** = 1.14e+11 | p = 0.000 → 🔧 Varianza del error, significativa.

**Conclusión:** Modelo parcialmente adecuado. Se puede simplificar a `SARIMAX(1,0,0)` con tendencia constante.

---

### 🌍 África

- **ar.L1** = -0.7363 | p = 0.000 → ✅ Muy significativo. Fuerte efecto autoregresivo.
- **ma.L1** = 0.6954 | p = 0.000 → ✅ Significativo. Impacto del error rezagado.
- **ma.L2** = -0.6116 | p = 0.000 → ✅ Significativo. Mejora del ajuste.
- **sigma2** = 5.15e+09 | p = 0.000 → 🔧 Varianza del error.

**Conclusión:** Todos los coeficientes son significativos. Modelo bien ajustado. Se sugiere `SARIMAX(1,0,2, trend='t')`.

---

### 🌍 Europa

- **ar.L1** = -0.7569 | p = 0.001 → ✅ Significativo. Dependencia temporal presente.
- **ma.L1** = 0.0329 | p = 0.865 → ❌ No significativo. Podría eliminarse.
- **ma.L2** = -0.3311 | p = 0.088 → ⚠️ Límite. Marginalmente relevante.
- **sigma2** = 6.37e+10 | p = 0.000 → 🔧 Varianza del error.

**Conclusión:** Modelo con buen componente AR, pero los MA no aportan. Se recomienda `SARIMAX(1,0,0, trend='c')`.

---

### 🌏 Asia

- **ma.L1** = -0.8318 | p = 0.000 → ✅ Muy significativo. Alta dependencia con el error.
- **ma.L2** = 0.2530 | p = 0.002 → ✅ Significativo. Aporta al modelo.
- **sigma2** = 2.83e+11 | p = 0.000 → 🔧 Varianza del error.

**Conclusión:** Modelo bien especificado. Ambos MA son relevantes. Se sugiere `SARIMAX(0,0,2, trend='t')`.

---

### 🌐 Oceanía

- **ar.L1** = -0.4601 | p = 0.411 → ❌ No significativo.
- **ma.L1** = 0.4128 | p = 0.490 → ❌ No significativo.
- **ma.L2** = -0.0221 | p = 0.958 → ❌ Irrelevante.
- **sigma2** = 2.31e+09 | p = 0.000 → 🔧 Varianza del error.

**Conclusión:** Ningún coeficiente significativo. Se sugiere ETS sin tendencia o revisar calidad de datos.

---
""")


warnings.filterwarnings("ignore")

st.title("📊 MAPE por región (últimos 5 años)")
st.markdown("Comparación de modelos SARIMAX/ETS ajustados por continente")

# ────── Parámetros de entrada ──────
validation_years = 5
continents = ['Américas', 'África', 'Europa', 'Asia', 'Oceanía']
gas = 'Emisiones (CO2eq) (AR5)'
prod_code = 6825

# ────── Filtrado de DataFrame ──────
mask = (
    (df_fao['Área'].isin(continents)) &
    (df_fao['Elemento'] == gas) &
    (df_fao['Código del producto'] == prod_code) &
    (df_fao['Año'].between(1990, 2022))
)
df_ts = (
    df_fao[mask]
    .assign(Valor_Mt = df_fao['Valor'] / 1000)
    .pivot_table(index='Año', columns='Área', values='Valor_Mt')
    .sort_index()
)
df_ts.index = pd.PeriodIndex(df_ts.index, freq='Y')

# ────── Configuración de modelos ──────
model_config = {
    'Asia':     ('sarimax_trend',     (0, 0, 2), 't'),
    'África':   ('sarimax_trend',     (1, 0, 2), 't'),
    'Europa':   ('sarimax_constant',  (1, 0, 1), 'n'),
    'Américas': ('sarimax_constant',  (1, 0, 1), 'n'),
    'Oceanía':  ('ets', None, None)
}

# ────── Cálculo de MAPE ──────
mape_scores = {}

for cont in continents:
    try:
        y = df_ts[cont].dropna()

        if len(y) <= validation_years + 2:
            st.warning(f"⚠️ {cont}: no hay suficientes datos para validación.")
            continue

        # Separar datos en entrenamiento y validación
        y_train = y.iloc[:-validation_years]
        y_test = y.iloc[-validation_years:]

        model_type, order, trend = model_config[cont]

        if model_type in ['sarimax_trend', 'sarimax_constant']:
            model = SARIMAX(y_train, order=order, seasonal_order=(0, 0, 0, 0),
                            trend=trend, enforce_stationarity=False,
                            enforce_invertibility=False)
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=validation_years)
            y_pred = forecast.predicted_mean

        elif model_type == 'ets':
            model = ExponentialSmoothing(y_train, trend='add', seasonal=None,
                                          initialization_method='estimated')
            fitted_model = model.fit()
            y_pred = fitted_model.forecast(validation_years)

        # Calcular MAPE
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mape_scores[cont] = round(mape * 100, 2)

    except Exception as e:
        st.error(f"❌ Error en {cont}: {e}")

# ────── Mostrar resultados ──────
if mape_scores:
    st.markdown("### ✅ Resultados:")
    df_resultado = pd.DataFrame(list(mape_scores.items()), columns=["Región", "MAPE (%)"])
    st.dataframe(df_resultado.set_index("Región").sort_values(by="MAPE (%)"))
else:
    st.warning("No se pudo calcular el MAPE para ninguna región.")


# ─────────────────────────────────────────────────────────────
# 3 · Explicación de MAPE en Markdown
# ─────────────────────────────────────────────────────────────
st.markdown("""
### 🧮 ¿Qué es el MAPE y para qué sirve?

El **MAPE (Mean Absolute Percentage Error)** se utiliza para evaluar la **capacidad predictiva** de un modelo.  
Pero **no se mide sobre el futuro real**, sino sobre un período conocido que se finge no haber visto.

---

#### 🧪 ¿Cómo se calcula?

1. Se toma una ventana de años recientes (por ejemplo, los últimos 5 años).
2. Se ajusta el modelo solo con los datos anteriores.
3. Se predicen esos últimos 5 años.
4. Se comparan las predicciones con los valores reales y se mide el error porcentual promedio.

---

#### 🧠 ¿Por qué es importante?

- Permite saber **qué tan bien hubiera predicho el modelo** en condiciones similares a las futuras.
- Es útil para **comparar modelos** entre sí:
    - Si un modelo A tiene MAPE = 3% y otro modelo B tiene 8%, el A es claramente superior.
- Ayuda a elegir el modelo que **menos se espera que se equivoque en el futuro**.

---

> 🔍 **Cuanto menor sea el MAPE, mejor es el modelo** para predecir.

Estos valores de MAPE muestran un modelo muy sólido en general

| Región   | MAPE  | Evaluación    |
| -------- | ----- | ------------- |
| Américas | 2.57% | **Excelente** |
| África   | 3.64% | **Excelente** |
| Europa   | 3.17% | **Excelente** |
| Asia     | 3.93% | **Excelente** |
| Oceanía  | 4.76% | **Muy buena** |
""")



# ─────────────────────────────────────────────────────────────
# 1 · Filtro base: series anuales 1990‑2022
# ─────────────────────────────────────────────────────────────
continents = ['Américas', 'África', 'Europa', 'Asia', 'Oceanía']
gas = 'Emisiones (CO2eq) (AR5)'
prod_code = 6825  # «Emisiones totales incluyendo LULUCF»

mask = (
    (df_fao['Área'].isin(continents)) &
    (df_fao['Elemento'] == gas) &
    (df_fao['Código del producto'] == prod_code) &
    (df_fao['Año'].between(1990, 2022))
)

df_ts = (df_fao[mask]
         .assign(Valor_Mt = df_fao['Valor'] / 1000)
         .pivot_table(index='Año', columns='Área', values='Valor_Mt')
         .sort_index())
df_ts.index = pd.PeriodIndex(df_ts.index, freq='Y')

# ─────────────────────────────────────────────────────────────
# 2 · Modelos por región (ajuste individualizado)
# ─────────────────────────────────────────────────────────────
model_config = {
    'Asia':     ('sarimax_trend',     (0, 0, 2), 't'),
    'África':   ('sarimax_trend',     (1, 0, 2), 't'),
    'Europa':   ('sarimax_constant',  (1, 0, 1), 'n'),
    'Américas': ('sarimax_constant',  (1, 0, 1), 'n'),
    'Oceanía':  ('ets', None, None)
}

forecast_horizon = 5
results = {}

for cont in continents:
    y = df_ts[cont].dropna()
    model_type, order, trend = model_config[cont]

    if model_type in ['sarimax_trend', 'sarimax_constant']:
        model = SARIMAX(y, order=order, seasonal_order=(0, 0, 0, 0),
                        trend=trend, enforce_stationarity=False,
                        enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        forecast = fitted_model.get_forecast(steps=forecast_horizon)
        f_mean = forecast.predicted_mean
        f_ci = forecast.conf_int()

    elif model_type == 'ets':
        model = ExponentialSmoothing(y, trend='add', seasonal=None,
                                      initialization_method='estimated')
        fitted_model = model.fit()
        f_mean = fitted_model.forecast(forecast_horizon)
        std_resid = np.std(y.diff().dropna())
        ci_margin = 1.96 * std_resid
        f_ci = pd.DataFrame({
            'lower y': f_mean - ci_margin,
            'upper y': f_mean + ci_margin
        }, index=f_mean.index)

    results[cont] = {
        'model': fitted_model,
        'fc_mean': f_mean,
        'fc_ci': f_ci
    }

# ─────────────────────────────────────────────────────────────
# 3 · Visualización comparativa
# ─────────────────────────────────────────────────────────────
sns.set_style('whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
axes = axes.flatten()

for i, cont in enumerate(continents):
    ax = axes[i]
    y = df_ts[cont]
    fc_mean = results[cont]['fc_mean']
    fc_ci = results[cont]['fc_ci']

    # Convertir índices si es necesario
    hist_index = y.index.to_timestamp()
    pred_index = pd.to_datetime(fc_mean.index.to_timestamp() if hasattr(fc_mean.index, 'to_timestamp') else fc_mean.index)

    # Gráficos
    ax.plot(hist_index, y, label='Histórico', color='steelblue')
    ax.plot(pred_index, fc_mean, label='Pronóstico', color='firebrick')
    ax.fill_between(pred_index,
                    fc_ci.iloc[:, 0].astype(float).values,
                    fc_ci.iloc[:, 1].astype(float).values,
                    color='firebrick', alpha=0.25, label='IC 95%')

    ax.set_title(cont)
    ax.set_ylabel('Mt CO₂‑eq')
    ax.legend()

# Eliminar subgráficos vacíos si sobran
if len(continents) < len(axes):
    for j in range(len(continents), len(axes)):
        fig.delaxes(axes[j])

fig.suptitle('Modelos por región – Pronóstico CO₂eq (AR5)', y=1.02)
plt.tight_layout()
st.pyplot(fig)


st.markdown("""Análisis de resultados:


Américas

  * Modelo aplicado: SARIMAX(1,0,1) con constante.


  * Serie histórica: Presenta una clara suba desde 1990 hasta 2007-2008, seguida de una tendencia a la baja y una caída más marcada post-2020.


  * Proyección: El modelo mantiene la tendencia de los últimos años y proyecta una estabilización levemente decreciente.


  * IC 95%: Bastante amplio → refleja alta incertidumbre (probablemente por la inestabilidad reciente).


  * Interpretación: La caída post-2019 parece ser influyente. El modelo no fuerza reversión al alza, lo que es razonable dado el comportamiento reciente.



África
  * Modelo aplicado: SARIMAX(1,0,2) con tendencia.


  * Serie histórica: Tendencia fuertemente creciente y estable en el tiempo.


  * Proyección: Continua el patrón ascendente de forma bastante lineal.


  * IC 95%: Aumenta de forma moderada → confianza razonable en la proyección.


  * Interpretación: Proyección muy coherente con el patrón de crecimiento constante, impulsado por el aumento poblacional y la industrialización creciente en la región.



Europa
  * Modelo aplicado: SARIMAX(1,0,1) con constante.


  * Serie histórica: Disminución continua desde 1990, con cierta estabilización en los últimos 10 años.


  * Proyección: El modelo proyecta una caída más lenta, tendiendo a una meseta.


  * IC 95%: Amplio pero simétrico, refleja cierta variabilidad pero no grandes shocks esperados.


  * Interpretación: La política climática europea parece estar reflejada en esta tendencia. El modelo mantiene esa línea, sin anticipar un rebote.



Asia
  * Modelo aplicado: SARIMAX(0,0,2) con tendencia.


  * Serie histórica: Ascenso constante y muy pronunciado.


  * Proyección: Continúa con un crecimiento sostenido, acelerado.


  * IC 95%: Ligeramente divergente hacia arriba, consistente con la variabilidad creciente.


  * Interpretación: Alta dependencia de Asia en combustibles fósiles y crecimiento económico explica este patrón. La proyección es creíble pero también preocupante.



Oceanía
  * Modelo aplicado: Exponential Smoothing (ETS additive).


  * Serie histórica: Alta volatilidad, sin tendencia clara a largo plazo.


  * Proyección: Leve crecimiento, pero con bandas de confianza muy anchas.


  * IC 95%: Muy amplio, incertidumbre altísima (lógica, dada la volatilidad y tamaño reducido).


  * Interpretación: La ETS captura el comportamiento errático sin imponer una tendencia clara. Es una buena elección para esta serie poco predecible.

""")


st.markdown("""
## 🌍 Pronóstico a Largo Plazo (2023–2042)

Para obtener predicciones más robustas a largo plazo (20 años), se ajustaron modelos **SARIMA estacionales** a las series históricas de emisiones totales de gases de efecto invernadero (CO₂eq) en cada continente.

#### 📌 Parámetros del modelo
- **Modelo utilizado:** SARIMA(1,1,1)(1,1,1,10)
- **Horizonte de pronóstico:** 20 años
- **Frecuencia:** Anual
- **Intervalo de confianza:** 95 %

#### 🔍 Justificación
Este modelo combina:
- Un componente autorregresivo (AR)
- Un componente de media móvil (MA)
- Diferenciación regular y estacional
- Componente estacional con periodicidad 10 (ajuste empírico)

#### 🧪 Resultados visuales
En los gráficos:
- 📈 La línea azul representa la serie histórica.
- 🔴 La línea roja representa la media del pronóstico.
- 🔴 La banda rosa muestra el intervalo de confianza al 95 %.

#### 🧠 Interpretación
- El modelo proyecta la **tendencia futura esperada** junto con su nivel de incertidumbre.
- Si las bandas de confianza son **muy amplias**, implica una mayor incertidumbre en la predicción.
- Este enfoque es útil para **planificación estratégica a largo plazo**, aunque debe revisarse periódicamente con nuevos datos.

""")


# ────────────────────────────────
# Cargar y preparar los datos
# ────────────────────────────────
continents = ['Américas', 'África', 'Europa', 'Asia', 'Oceanía']
gas = 'Emisiones (CO2eq) (AR5)'
prod_code = 6825  # Emisiones totales incluyendo LULUCF

# Asegurate que `df_fao` esté previamente cargado como DataFrame
df_fao['Valor_Mt'] = df_fao['Valor'] / 1000

mask = (
        df_fao['Área'].isin(continents) &
        (df_fao['Elemento'] == gas) &
        (df_fao['Código del producto'] == prod_code) &
        df_fao['Año'].between(1990, 2022)
)

df_ts = (
    df_fao[mask]
    .pivot_table(index='Año', columns='Área', values='Valor_Mt')
    .sort_index()
)

# ────────────────────────────────
# Configuración de modelo SARIMA
# ────────────────────────────────
forecast_horizon = 20  # 20 años
sarima_order = (1, 1, 1)
seasonal_order = (1, 1, 1, 10)


# ────────────────────────────────
# Parámetros
# ────────────────────────────────
df_ts.index = pd.to_datetime(df_ts.index, format='%Y')
continents = df_ts.columns.tolist()

# Define los parámetros SARIMA (pueden ajustarse según el caso)
sarima_order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # estacionalidad anual

st.markdown("## 📈 Pronóstico de emisiones CO₂eq por continente (2023–2042)")

# ───────────── Grilla 3x2 ─────────────
cols = st.columns(3)

for i, cont in enumerate(continents):
    y = df_ts[cont].dropna()
    index_hist = y.index

    # Modelo SARIMA
    model = SARIMAX(y,
                    order=sarima_order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)

    # Pronóstico
    forecast = results.get_forecast(steps=forecast_horizon)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Eje de fechas del pronóstico
    forecast_index = pd.date_range(start=index_hist[-1] + pd.DateOffset(years=1),
                                   periods=forecast_horizon, freq='Y')

    # Gráfico
    fig = go.Figure()

    # Serie histórica
    fig.add_trace(go.Scatter(
        x=index_hist,
        y=y.values,
        mode='lines',
        name='Histórico',
        line=dict(color='steelblue')
    ))

    # Serie pronosticada
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_mean,
        mode='lines',
        name='Pronóstico',
        line=dict(color='firebrick')
    ))

    # IC 95%
    fig.add_trace(go.Scatter(
        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
        y=list(forecast_ci.iloc[:, 0]) + list(forecast_ci.iloc[:, 1][::-1]),
        fill='toself',
        fillcolor='rgba(178,34,34,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='IC 95%'
    ))

    fig.update_layout(
        title=f"🌍 {cont} — SARIMA: Emisiones CO₂eq",
        xaxis_title="Año",
        yaxis_title="Mt CO₂eq",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=30, r=30, b=30),
        height=400
    )

    # Mostrar en columna correspondiente
    cols[i % 3].plotly_chart(fig, use_container_width=True)



st.markdown("""# Utilizando ETS""")



# ---------- Título ----------
st.markdown("## 📊 Pronóstico de emisiones CO₂eq usando modelo ETS (Holt-Winters)")

st.markdown("""
El modelo **ETS (Error, Tendencia, Estacionalidad)** es útil cuando se quiere capturar la dinámica del crecimiento o descenso de una serie temporal. 
En este caso, utilizamos el componente de tendencia aditiva y sin estacionalidad, adecuado para series anuales sin fluctuaciones estacionales.

🔍 *Esta predicción se realiza para las emisiones de CO₂eq del continente 'Américas' desde 2023 hasta 2042.*
""")



# ---------- Título ----------
st.markdown("## 📊 Modelo ETS – Pronóstico de emisiones CO₂eq (2023–2042)")
st.markdown("""
Se utiliza el modelo **ETS (Error, Trend, Seasonality)** para proyectar las emisiones de CO₂eq en cada continente.

🔧 Este modelo se ajusta automáticamente a la tendencia y proyecta 20 años hacia adelante.
""")

# ---------- Suposición: df_ts ya está cargado correctamente ----------
# df_ts debe tener PeriodIndex (Año) y columnas = continentes ('Américas', 'Europa', 'Asia', 'Oceanía', 'África')



# ────── Configuración general ──────
forecast_years = 20
df_ts.index = pd.to_datetime(df_ts.index, format='%Y')
continentes = df_ts.columns.tolist()

# ────── Mostrar en grilla 3x2 ──────
cols = st.columns(3)
for i, continente in enumerate(continentes):
    y = df_ts[continente].dropna()

    # Modelo ETS
    model = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method='estimated')
    fitted_model = model.fit()
    forecast = fitted_model.forecast(forecast_years)

    # IC 95%
    std_resid = np.std(y.diff().dropna())
    ci_margin = 1.96 * std_resid
    ci_lower = forecast - ci_margin
    ci_upper = forecast + ci_margin

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Histórico', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Pronóstico', line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=ci_upper, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast.index, y=ci_lower, fill='tonexty', name='IC 95%', fillcolor='rgba(255,140,0,0.2)', line=dict(width=0)))

    fig.update_layout(
        title=f"🌍 {continente} – ETS: Emisiones CO₂eq (2023–{forecast.index[-1].year})",
        xaxis_title='Año',
        yaxis_title='Mt CO₂eq',
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )

    # Mostrar en la columna correspondiente
    cols[i % 3].plotly_chart(fig, use_container_width=True)


st.markdown("""## 📈 Modelo Prophet – Predicción de emisiones 2023–2042

Se utiliza el modelo **Prophet** (desarrollado por Facebook) para proyectar las emisiones de CO₂eq en cada continente durante los próximos 20 años.

### ¿Por qué Prophet?
- Se adapta automáticamente a patrones de tendencia y estacionalidad (si los hay).
- Es robusto frente a valores atípicos y cambios estructurales.
- Ideal para datos anuales como los analizados aquí.

### ✔️ Metodología
- Para cada continente se entrenó un modelo Prophet con los datos históricos 1990–2022.
- Luego se proyectaron 20 años adicionales (2023–2042).
- Se grafican:
  - Los valores históricos (línea azul).
  - La predicción central (línea roja).
  - El intervalo de confianza del 95 % (franja roja clara).

---

### ✅ Interpretación esperada

- Si el modelo capta correctamente la tendencia (creciente o decreciente), la proyección es valedera.
- La amplitud de la franja roja indica la **incertidumbre**: cuanto más ancha, menos precisión tiene la predicción.
- En regiones con alta variabilidad (como Oceanía), se espera mayor amplitud del intervalo.

---

> 📌 **Advertencia**: Prophet no incluye componentes autorregresivos explícitos, por lo que puede no capturar relaciones finas entre observaciones anuales consecutivas. Sin embargo, su robustez lo hace un excelente modelo comparativo frente a SARIMAX o ETS.
""")


# ---------- Título principal ----------
st.markdown("## 📈 Predicción de emisiones por continente (modelo Prophet)")

# ---------- Descripción detallada ----------
st.markdown("""
Se utiliza el modelo **Prophet** (desarrollado por Facebook) para proyectar las emisiones de CO₂eq en cada continente durante los próximos 20 años.

### ¿Por qué Prophet?
- Se adapta automáticamente a patrones de tendencia y estacionalidad (si los hay).
- Es robusto frente a valores atípicos y cambios estructurales.
- Ideal para datos anuales como los analizados aquí.

### ✅ Metodología
- Para cada continente se entrenó un modelo Prophet con los datos históricos 1990–2022.
- Luego se proyectaron 20 años adicionales (2023–2042).
- Se grafican:
  - Los valores históricos (línea azul).
  - La predicción central (línea roja).
  - El intervalo de confianza del 95 % (franja roja clara).
""")

# ────── Silenciar warnings y logs ──────
warnings.filterwarnings("ignore")  # Silencia todos los warnings
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('prophet').propagate = False
logging.getLogger('cmdstanpy').propagate = False

# ────── Estilo de gráficos ──────
sns.set_style("whitegrid")

# ────── Título Streamlit ──────
st.title("📊 Forecast de emisiones CO₂eq (AR5) con Prophet")
st.markdown("## Predicción por continente (1990‑2022 + 20 años)")
st.write(" ")
st.write(" ")
st.markdown("""# ¿Por qué usar Prophet para modelar emisiones de CO₂?

## ¿Qué es Prophet?

**Prophet** es una herramienta de pronóstico de series temporales desarrollada por **Facebook (Meta)**. Está pensada para:

- Modelar **tendencias no lineales**.
- Capturar **cambios de régimen** o inflexiones en la evolución histórica.
- Incluir opcionalmente **estacionalidades** (diarias, semanales, anuales).
- Ser **fácil de usar** para analistas sin conocimientos avanzados en estadística.

---

## Prophet descompone la serie temporal de la siguiente forma:


    y(t) = g(t) + s(t) + h(t) + ε_t

Donde:

* g(t) → Tendencia (puede ser lineal o logística, con posibles "cambios de pendiente").

* s(t) → Estacionalidad (opcional, puede ser anual, semanal, diaria).

* h(t) → Efectos por fechas especiales (festivos, eventos).

* ε_t → Ruido aleatorio (residuo no explicado).


| Beneficio clave                       | ¿Por qué importa en tu dataset?                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
|  Captura **cambios de tendencia**    | Las emisiones no evolucionan linealmente. Prophet detecta **cambios de pendiente automáticamente**, lo que ARIMA no hace bien. |
|  No requiere **estacionariedad**     | Prophet **no exige diferenciar** ni transformar la serie. SARIMA sí, y esto puede distorsionar el significado del pronóstico.  |
|  Funciona bien con **datos anuales** | Las series son anuales. Prophet acepta fácilmente series con cualquier frecuencia sin reconfigurar nada.                       |
|  Maneja bien la **incertidumbre**    | Prophet devuelve automáticamente **intervalos de confianza del 95%**, facilitando la comunicación de riesgo/incertidumbre.     |
|  Automatizable por región            | Se puede aplicar el mismo modelo a cada continente sin tunear manualmente los parámetros. Ideal para **automatización**.          |
|  Interpretabilidad de componentes    | Prophet permite ver **la tendencia sola**, algo útil para análisis visual y argumentación.                                     |


 ### Conclusión:

Prophet es una excelente elección para tu estudio de emisiones por continente porque:

* Se tiene series de más de 30 años.

* Las tendencias varían marcadamente por región.

* Se necesita proyectar 20 años más con un enfoque claro e interpretable.

* Requiere poco ajuste manual.

* Los resultados son fácilmente graficables y presentables.""")

st.write(" ")
st.write(" ")

# Silenciar warnings y logs molestos
warnings.filterwarnings("ignore")
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('prophet').propagate = False
logging.getLogger('cmdstanpy').propagate = False

# Configurar estilo
sns.set_style("whitegrid")

# ────── Datos base ──────
continents = ['Américas', 'África', 'Europa', 'Asia', 'Oceanía']
gas = 'Emisiones (CO2eq) (AR5)'
prod_code = 6825  # Emisiones totales incluyendo LULUCF

# ────── Filtro base ──────
mask = (
    (df_fao['Área'].isin(continents)) &
    (df_fao['Elemento'] == gas) &
    (df_fao['Código del producto'] == prod_code) &
    (df_fao['Año'].between(1990, 2022))
)

df_filtrado = df_fao[mask].copy()
df_filtrado['Valor_Mt'] = df_filtrado['Valor'] / 1000
df_ts = df_filtrado.pivot_table(index='Año', columns='Área', values='Valor_Mt')
df_ts = df_ts.sort_index()

# ────── Forecast por continente ──────
results_prophet = {}
graficos = []

for cont in continents:
    try:
        serie = df_ts[cont].dropna().reset_index()
        serie.columns = ['ds', 'y']
        serie['ds'] = pd.to_datetime(serie['ds'], format='%Y')

        model = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.5)
        model.fit(serie)

        future = model.make_future_dataframe(periods=20, freq='Y')
        forecast = model.predict(future)

        results_prophet[cont] = forecast

        fig, ax = plt.subplots(figsize=(5, 3.5))  # Tamaño reducido y uniforme
        ax.plot(serie['ds'], serie['y'], label='Histórico', color='steelblue', linewidth=1.5)
        ax.plot(forecast['ds'], forecast['yhat'], label='Pronóstico', color='firebrick')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        color='firebrick', alpha=0.25, label='IC 95%')
        ax.set_title(cont)
        ax.set_ylabel("Mt CO₂‑eq")
        ax.set_xlabel("Año")
        ax.legend()
        ax.grid(True)

        graficos.append(fig)

    except Exception as e:
        graficos.append(f"❌ Error al procesar {cont}: {e}")

# ────── Mostrar en grilla 2x3 con un espacio vacío al final ──────
st.subheader("🔮 Pronósticos por continente (con Prophet)")

# Fila 1
cols1 = st.columns(3)
for i in range(3):
    with cols1[i]:
        st.pyplot(graficos[i])

# Fila 2
cols2 = st.columns(3)
for i in range(3):
    with cols2[i]:
        if i + 3 < len(graficos):
            st.pyplot(graficos[i + 3])
        else:
            st.empty()  # Celda vacía para que la grilla quede uniforme



# ---------- Interpretación final ----------
st.markdown("""
---

### ✅ Interpretación esperada

- Si el modelo capta correctamente la tendencia (creciente o decreciente), la proyección es válida.
- La **amplitud del intervalo** indica la **incertidumbre**: cuanto más ancho, menos precisa la predicción.
- En regiones con alta variabilidad (como Oceanía), se espera mayor amplitud del intervalo.

🔺 **Advertencia**: Prophet no incluye componentes autorregresivos explícitos, por lo que puede no capturar relaciones finas entre observaciones anuales consecutivas. Sin embargo, su robustez lo hace un excelente modelo comparativo frente a SARIMAX o ETS.
""")