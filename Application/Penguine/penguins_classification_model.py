# Import Libraries

import streamlit as st
import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')

df = penguins.copy()
# st.write(df) # just to look at the dataset to identify the target feature
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# separating X and Y
X = df.drop('species', axis=1)
Y = df['species']

# Build Random Forest Model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model so no need to build the model again & again each time we change the input
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))