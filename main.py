from fastapi import FastAPI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from pydantic import BaseModel
import csv 

app = FastAPI()

class Review(BaseModel):
    user_id: int
    review_id: int
    rate: int
    money: float
    purpose: str
    place_type: str
    climate: str
    activities: str
       

values = []


@app.get("/kmeans/getRecomendations/{id}")
async def main(id:int):

    values = await kmeans(id)

    return values

@app.post("/kmeans/postData")
def postData(review: Review):
    add_to_dataframe(review)

    #return{f"{review.climate}"}

@app.get("/kmeans/discover/{id}")
async def discover(id:int):

    values = await kmeans_discover(id)

    return values


async def kmeans(id):
    ### IMPORTAR LOS DATOS Y PREPARAR LOS DATASET
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
    
    if (reviews_id.size <= 30 ):
        return reviews_id.tolist()
    else: 
        return reviews_id[0:30].tolist()
    
async def kmeans_discover(id):
    ### IMPORTAR LOS DATOS Y PREPARAR LOS DATASET
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
    reviews = datasetOriginal[ ~datasetOriginal["cluster"].isin(clusters)]
    reviews = reviews[reviews["rate"] == 5]
    reviews = reviews[reviews["user_id"] != id]
    reviews_id = reviews['review_id'].values

    #retornando los datos
    
    if (reviews_id.size <= 30 ):
        return reviews_id.tolist()
    else: 
        return reviews_id[0:30].tolist()
def add_to_dataframe(data):
    id = getLastId() +1
    data_converted = [id,data.user_id,data.review_id,data.rate,data.money,data.purpose,data.place_type,data.climate,data.activities]
    ### IMPORTAR LOS DATOS 
    file = open("MyTravelsdata-aws-final.csv", 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(data_converted)
    file.close  

def searchForExistingValue(id):
    dataset =pd.read_csv("MyTravelsdata-aws-final.csv")

    #return dataset.user_id.isin([id])
    return id in dataset[['id']].values

def getLastId():
    dataset =pd.read_csv("MyTravelsdata-aws-final.csv")
    return dataset['id'].iloc[-1]

def getId(review_id):
    dataset =pd.read_csv("MyTravelsdata-aws-final_copy.csv")
    row = dataset.loc[dataset['review_id']== review_id].set_index('id')
    if len(row) == 0:
        return -1
    
    return row.index[0]

def editRow(data):
    data_converted = [data.rate,data.money,data.purpose,data.place_type,data.climate,data.activities]
    dataset =pd.read_csv("MyTravelsdata-aws-final.csv")
    dataset.loc[data.reviewId-1,['rate','money','purpose','place_type','climate','activities']] = [4,1500.0,'vacaciones','Pueblo','Soleado','Turisticas']
    dataset.to_csv("MyTravelsdata-aws-final.csv", index=False) 

def deleteRow(review_id):
    review_id = getId(review_id)
    print(" ---- ",int(review_id))
    if review_id != -1:
        dataset =pd.read_csv("MyTravelsdata-aws-final_copy.csv")
        dataset.set_index(['id'])
        dataset.drop(int(review_id),  inplace=True)
        dataset.to_csv("MyTravelsdata-aws-final_copy.csv", index=False)
    else:
        print("nada")


#function get id creada, usala para borrarfila y editarcolumna

