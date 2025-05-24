import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

image = Image.open('/home/akki/Data_Science_Learning_and_Practice/data_science_projects/Iris_Prediction_App.png')
st.image(image,use_container_width=True)

st.write("""
# Simple Iris Flower Prediction App
         
This app predicts the Iris flower type!
         
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('sepal_length',4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)


prediction = clf.predict(df)
prediction_probability = clf.predict_proba(df)


st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Iris flower types Encoded')
iris_type_with_index = {'0':'setosa',
        '1':'versicolor',
        '2':'verginica'}
info = pd.DataFrame(iris_type_with_index, index=[0])
st.write(info)

st.subheader('Prediction Probability')
st.write(prediction_probability)

