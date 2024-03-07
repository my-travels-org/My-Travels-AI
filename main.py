from fastapi import FastAPI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from pydantic import BaseModel

app = FastAPI()

values = []

@app.get("/kmeans/getRecomendations/{id}")
async def main(id:int):

    values = await kmeans(id)

    return values


async def kmeans(id):
    ### IMPORTAR EL LOS DATOS Y PREPARAR LOS DATASET
    datasetOriginal = pd.read_csv("MyTravelsdata-aws-final.csv")
    dataset =pd.read_csv("MyTravelsdata-aws-final.csv")
    dataset =  dataset.loc[:, dataset.columns != "user_id"]
    dataset =  dataset.loc[:, dataset.columns != "review_id"]
    numeric_values =dataset[['rate','money']]

    #TRANSFORMAR LOS DATOS CATEGORICOS A NUMERICOS
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform = 'pandas')
    ohetransform = ohe.fit_transform(dataset[['purpose','place_type', 'climate', 'activities']])

    #preparar los datos numericos y concatenar los dataset
    from sklearn.preprocessing import MinMaxScaler
    scaleMinMax = MinMaxScaler(feature_range=(0,1))
    numeric_values = scaleMinMax.fit_transform(numeric_values)
    numeric_values = pd.DataFrame(numeric_values, columns=['rate', 'money'])
    dataset_transformed = pd.concat([numeric_values,ohetransform], axis=1)

    #ENTRENANDO EL MODELO
    x = dataset_transformed .iloc[:, :].values
    kmeans  = KMeans(n_clusters = 7, init = 'k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    ##AGREGAR LOS CLUSTERS AL DATASET
    datasetOriginal['cluster'] = y_kmeans

    #FILTRAR LOS DATOS

    #Obteniendo los clusters de los reviews del usuario
    user_reviews_cluster = datasetOriginal[datasetOriginal["user_id"] == id]
    user_reviews_cluster = user_reviews_cluster[user_reviews_cluster["rate"] >= 4]
    clusters = user_reviews_cluster['cluster'].values

    #OBTENIENDO LOS REVIEWS DE DICHO LOS CLUSTERS 
    reviews = datasetOriginal[datasetOriginal["cluster"].isin(clusters)]
    reviews = reviews[reviews["rate"] == 5]
    reviews = reviews[reviews["user_id"] != id]
    reviews_id = reviews['review_id'].values

    #retornando los datos
    
    if (reviews_id.size <= 10 ):
        return reviews_id.tolist()
    else: 
        return reviews_id[0:10].tolist()