# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Image for the Penguin Classification App
image = Image.open('/home/akki/Data_Science_Learning_and_Practice/data_science_projects/Penguine_Classification/Penguin Classification App Logo.png')
st.image(image, use_container_width=True)

# Title of the App
st.write("""
# Penguin Classification App
    
This app predicts the **Palmer Penguin** species based on the input features.
         
Data obtained from the [Palmer Penguins library]

""")



st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV file](/home/akki/Data_Science_Learning_and_Practice/data_science_projects/Penguine_Classification/penguins_example.csv)
""")


# Collects user input features into dataframe

uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['CSV'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 200.0)
        body_mass_g = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguin dataset
# this will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters(shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)