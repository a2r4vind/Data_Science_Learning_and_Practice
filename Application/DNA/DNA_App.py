# Import Libraries

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

# Page Title

image = Image.open('/home/akki/Data_Science_Learning_and_Practice/Application/DNA/ DNA_Web_App_Logo.jpg')
st.image(image, use_container_width=True)

st.write(""""
# DNA Nucleotide Count Web App
         
This app counts the nucleotide composition of query DNA!
         
***
""")

# Input Text Box

# st.sidebar.header('Enter DNA sequence')
st.header('Enter DNA sequence')
sequence_input = ">DNA Query\nGAACACGTGGAGGCAAA"

#sequence = st.sidebar.text_area("Sequence input", sequence_input, height) 
sequence = st.text_area("Sequence input", sequence_input, height=250)
sequence = sequence.splitlines()
sequence = sequence[1:] # Skips the first sequence name (first line)
sequence = ''.join(sequence) # concatinate list to string

st.write("""
***
""")

# print the input DNA sequence
st.header("INPUT (DNA Query)")
sequence

# DNA nucleotide count
st.header('OUTPUT (DNA Nucleotide Count)')

## 1. Print dictionary
st.subheader('1. Print Dictionary')
def DNA_nucleotide_count(seq):
    d = dict([
        ('A',seq.count('A')),
        ('T',seq.count('T')),
        ('G',seq.count('G')),
        ('C',seq.count('C'))
    ])
    return d

X = DNA_nucleotide_count(sequence)

X_label = list(X)
X_values = list(X.values())

X

## 2. Print text
st.subheader('2. Print text')
st.write(f'There are {str(X['A'])} adenine (A)')
st.write(f'There are {str(X['T'])} thymine (T)')
st.write(f'There are {str(X['G'])} adenine guanine (G)')
st.write(f'There are {str(X['C'])} adenine cytosine (C)')

## 3. Display Dataframe
st.subheader('3. Display DataFrame')
df = pd.DataFrame.from_dict(X, orient="index")
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'nucleotide'})
st.write(df)

## 4. Display Bar Chart Using Altair
st.subheader('4. Display Bar chart')
p = alt.Chart(df).mark_bar().encode(
    x='nucleotide',
    y='count'
)
p = p.properties(
    width=alt.Step(80) # control width of Bar.
)
st.write(p)