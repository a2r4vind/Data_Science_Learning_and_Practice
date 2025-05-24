import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image


st.title('NBA Player Statistics Explorer')

image = Image.open('/home/akki/Data_Science_Learning_and_Practice/data_science_projects/basketball_app_logo.png')
st.image(image, use_container_width=True)

st.markdown("""
This dashboard performs simple web scraping of NBA player statistics data
* **Python Libraries Used:** base64, pandas, streamlit, matplotlib, seaborn
* **Data Source:** [Basketball Reference](https://www.basketball-reference.com/).

""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2020))))

@st.cache_data
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{str(year)}_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0) # fills NaN with 0
    player_stats = raw.drop(["Rk"], axis=1) 
    return player_stats
playerstats = load_data(selected_year)
# st.write(playerstats.columns) just to check the columns

# sidebar - Team selection
sorted_unique_team = sorted(playerstats.Team.astype(str).unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# sidebar - Position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data 
df_selected_team = playerstats[(playerstats.Team.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write(f'Data Dimension: {str(df_selected_team.shape[0])} rows and {str(df_selected_team.shape[1])} columns.')
st.dataframe(df_selected_team)

# Download NBA Player Stats data
# discussed on https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="nba_player_stats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap 
# Data Visualization

if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.select_dtypes(include=['number']).corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(fig)