
#Importamos las librerias necesarias
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

#to make the plotly graphs
import plotly.graph_objs as go
import plotly.express as px
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

import ipywidgets as widgets

#text mining
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from wordcloud import WordCloud

#Fotos
import requests
from PIL import Image
from io import BytesIO
import random

st.set_option('deprecation.showPyplotGlobalUse', False)

#para que no nos aparezcan ciertos mensajes de error
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

page_bg_img="""
<style>
[data-testid="stAppViewContainer"]  {
background-image: url("https://a.cdn-hotels.com/gdcs/production148/d757/6059b2a0-8f10-11e8-a0da-0242ac11004d.jpg?impolicy=fcrop&w=1600&h=1066&q=medium");
background-size: cover;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style>
"""

st.set_page_config(
    page_title="Datos Airbnb Barcelona",
    page_icon="https://d33byq9npfy6u9.cloudfront.net/4/2022/06/28100321/289840534_10160185317062458_71839744288889599_n.jpg",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(page_bg_img, unsafe_allow_html=True)

#Conseguimos los archivos de insideairbnb
# Creamos carpeta

# Importar librerías
import sys
import os
import wget

# Creamos un directorio
os.getcwd() # donde estamos
downloads = os.getcwd() # ruta original de descarga
folder = "/InsideAirbnb" # nombre folder donde vamos a descargar los datos
datasets = ["listings.csv.gz","calendar.csv.gz", "reviews.csv.gz"]

if not os.path.exists(downloads + folder): # si la ruta no existe
    os.makedirs(downloads + folder) # crea una carpeta
os.chdir(downloads + folder) # cambiamos el directorio a nuestra folder

urls=["http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/data/calendar.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/data/reviews.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/visualisations/neighbourhoods.csv",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/visualisations/neighbourhoods.geojson"]

for url in urls:
    filename = url.split("/")[-1] # obtener el nombre del archivo de la URL
    filepath = os.path.join(downloads + folder, filename) # construir la ruta del archivo
    if not os.path.exists(filepath): # si el archivo no existe en la carpeta, descargarlo
        wget.download(url, filepath)

os.chdir(downloads) # cambiamos directorio al original


#Transformamos los archivos .gz a archivos .csv con los que trabajaremos mejor

listings_processed_path = "C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/listings_processed.csv"
calendar_processed_path = "C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/calendar_processed.csv"
reviews_processed_path = "C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/reviews_processed.csv"

if not os.path.isfile(listings_processed_path):
    listings = pd.read_csv("C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/listings.csv.gz", compression='gzip', header=0, sep=',', quotechar='"')
    listingsp = listings.to_csv(listings_processed_path, index=False)

if not os.path.isfile(calendar_processed_path):
    calendar = pd.read_csv("C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/calendar.csv.gz", compression='gzip', header=0, sep=',', quotechar='"')
    calendarp = calendar.to_csv(calendar_processed_path, index=False)

if not os.path.isfile(reviews_processed_path):
    reviews = pd.read_csv("C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/reviews.csv.gz", compression='gzip', header=0, sep=',', quotechar='"')
    reviewsp = reviews.to_csv(reviews_processed_path, index=False)
#Lectura de los csv
listingsp=pd.read_csv("C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/listings_processed.csv")
calendarp=pd.read_csv("C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/calendar_processed.csv")
reviewsp=pd.read_csv("C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/reviews_processed.csv")
neighbourhoodsp=pd.read_csv("C:/Users/User/Desktop/samplerepo/Modulo_2/InsideAirbnb/InsideAirbnb/neighbourhoods.csv")

#Limpieza del dataframe
listingsp = listingsp.dropna(subset=['latitude', 'longitude'])

data = listingsp.join(calendarp, lsuffix='_listingsp', rsuffix='_calendarp')

data['date'] = pd.to_datetime(data['date'])

data['grouping'] = ''
data.loc[data['property_type'].str.contains('entire|home', case=False), 'grouping'] = 'Entire home/apt'
data.loc[data['property_type'].str.contains('private|room', case=False), 'grouping'] = 'Private room'
data.loc[data['property_type'].str.contains('shared', case=False), 'grouping'] = 'Shared room'
data.loc[data['grouping'] == '', 'grouping'] = 'Entire home/apt'

data["price_calendarp"] = data["price_calendarp"].str.replace("$", "")
data["price_calendarp"] = data["price_calendarp"].astype(float)
data["price_calendarp"] = data["price_calendarp"].astype(int)


# Definir el diccionario de secciones
sections = {
    "Introducción": "introduccion",
    "Dataframe": "dataframe",
    "EDA": "EDA",
    "Conclusión": "conclusion"
}

# Crear las columnas para simular las pestañas y aplicar el estilo CSS personalizado a los botones
col1, col2, col3, col4 = st.columns(4)
with col1:
    tab1 = st.button(
        "Introducción",
        key="tab1",
        )
with col2:
    tab2 = st.button(
        "Dataframe",
        key="tab2",
        )
with col3:
    tab3 = st.button(
        "EDA",
        key="tab3",
        )
with col4:
    tab4 = st.button(
        "Conclusión",
        key="tab4",
        )

# Usar los identificadores de secciones para mostrar el contenido de cada pestaña
if tab1:
    st.title('<span style="background-color:black;color:red;padding:10px;">Airbnb Nueva York: Las propiedades de la gran manzana</span>')
    st.subheader("Fecha: 23/03/2023")
    st.header("Autor: Eloi López Massana")
    st.markdown("##")
    st.header("Introducción")
    st.markdown('<span style="background-color:black;color:white;padding:10px;">¡Bienvenidos a mi trabajo de Airbnb sobre la ciudad de Nueva York. En éste trataremos con bastantes datos de diferentes propiedades de la cuidad que nunca duerme, para ver la mejor manera de volvernos ricos con la multipropiedad inmobiliaria. Importante destacar que ésta información solo se usa con fines educativos y su uso debe ser bajo el propio riesgo de cada uno y se recomienda que se actúe con integridad moral. Recuerda, la ética siempre está de moda, incluso cuando estás buscando convertirte en el próximo magnate inmobiliario en Airbnb. Ahora, ¡vamos a sumergirnos en esos datos y ver qué podemos encontrar!</span>', unsafe_allow_html=True)
    
if tab2:
    st.header("Dataframe")

if tab3:
    st.write("Contenido de la pestaña EDA")
if tab4:
    st.write("Contenido de la pestaña Conclusión")