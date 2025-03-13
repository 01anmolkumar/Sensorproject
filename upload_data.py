from pymongo import MongoClient

import pandas as pd
import json

#url
url="mongodb+srv://anmolshrivastava50:12345@cluster0.y4lp4.mongodb.net/?retryWrites=true&w=majority"

#create a new client and connect to server
client=MongoClient(url)

#create database name collection name
DATABASE_NAME="pwskills"
COLLECTION_NAME="waferfault"

df =pd.read_csv(r"E:\OneDrive\Desktop\PROJECT\sensorproject\notebooks\wafer_23012020_041211.csv")

df=df.drop("Unnamed: 0",axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)