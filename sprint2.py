# Streamlit Output for Sprint 2 - Group 4

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import warnings
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
scaler = MinMaxScaler() 
warnings.filterwarnings('ignore')

# Parameters
features = ["danceability", "energy", "valence", "tempo", "loudness", "acousticness", "speechiness", "instrumentalness"]
success = ['streams', 'popularity', 'position']
columns_to_view = ['artist_name', 'track_name'] + features

# Dfs and Cleaning Data
tracks_df = pd.read_csv('data/spotify_daily_charts_tracks.csv')

charts_df = pd.read_csv('data/spotify_daily_charts.csv')
charts_df['date'] = pd.to_datetime(charts_df['date'])

merged_df = charts_df.merge(tracks_df, on='track_id', how='left')
merged_df = merged_df.drop(columns='track_name_y')
merged_df = merged_df.rename(columns={'track_name_x': 'track_name'})
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df = merged_df.set_index('date')

artisttracks_df = merged_df.copy()

sb19_df = artisttracks_df[artisttracks_df['artist_name'] == 'SB19'].drop_duplicates()
sb19_df['tempo'] =  scaler.fit_transform(sb19_df[['tempo']])
sb19_df['loudness'] = scaler.fit_transform(sb19_df[['loudness']])

# EDA 'Sound' Key df
sb19df_features = sb19_df[columns_to_view].copy()
sb19df_features = sb19df_features.set_index(['track_name', 'artist_name'])
sb19df_features_stacked = pd.DataFrame({'value': sb19df_features.stack()})
sb19df_features_stacked = sb19df_features_stacked.reset_index()
sb19df_features_stacked = sb19df_features_stacked.rename(columns={'level_2': 'feature'})

# Actual Page
my_page = st.sidebar.radio('Page Navigation', ['About', 'EDA', 'Results', 'Playlists'])

# About Page 
if my_page == 'About':
    image = Image.open('SB19.jpeg')
    
    st.title("SB19's Next Move")
    st.write("A new wave of Filipino talent has been introduced to the Philippine pop music scene. Boy group SB19" \
                 " is the newest sensation in this movement as they cement themselves as one of the trailblazers in the current era of Original Pinoy Music (OPM).")
    
    st.image(image, caption='SB19')
    st.header("Spotify Daily Charts Tracks")
    if st.checkbox('Show data', value = True):
        st.subheader('Data')
        data_load_state = st.text('Loading data...')
        st.write(tracks_df.head(20))
        data_load_state.markdown('Loading data...**done!**')
        
# EDA Page
elif my_page == 'EDA':
    option = st.sidebar.selectbox('Options:', ['Due for Another Hit', 'Key Track: MAPA', 'Sound', 'SB19 Audio Features', 'KPOP', 'Pinoy Tastes'])
    
    st.title(f"{option}")
    
    if option == 'Due for Another Hit':
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot(111)
    
        BOUNDS_WINDOW = 14
        data = sb19_df[(sb19_df['artist_name']=='SB19')]['streams'].resample('M').sum().cumsum()
    
        plt.plot(data, label='SB19 Overall Trend', color = '#808080')
        plt.legend()
        plt.ylabel('cumulative sum streams')
        plt.title('SB19 Monthly Streams')
        
        st.pyplot(fig)
        st.write('SB19’s breakthrough streaming performance in July of 2021 should be capitalized on to sustain their increasing market value')
        
    elif option == 'Key Track: MAPA':
        fig = plt.figure(figsize=(14, 6))
        ax = plt.subplot(111)
        
        # df to be used
        sb19_mstreams = sb19_df.groupby('track_name')[['streams']].resample('M').sum()
        sb19_mstreams = sb19_mstreams.reset_index()
        sb19_mstreams['track_id'] = sb19_mstreams['track_name'].apply(lambda x: x.split('(')[0])\
        .apply(lambda x: x.split(' - ')[0])

        sb19_heat = sb19_mstreams.pivot_table(index='track_name', columns='date', values='streams')
        sb19_heat = sb19_heat/1000000
        sb19_heat.fillna(0, inplace=True)
        sb19_heat['total_streams'] = sb19_heat.sum(axis=1)
        sb19_heat = sb19_heat.sort_values('total_streams',ascending=False)
        
        moncols = sb19_heat.columns[:-1]
        yymm_cols = pd.Series(moncols.values).apply(lambda x: x.strftime('%Y-%m'))
        
        sns.heatmap(sb19_heat[moncols], ax=ax,
                    cmap='viridis',
                    cbar_kws={'label': 'milliin streams', 'ticks': np.arange(0, 10, .5)},
                    xticklabels=yymm_cols, yticklabels=True)
        
        plt.title('SB19 tracks - streaming trend')
        plt.ylabel('billboard tracks')
        plt.xlabel('dates')
        
        st.pyplot(fig)
        st.write('Was the only SB19 track to reach at least 50 million streams in a month and sustain listener’s interest')
        
    elif option == 'Sound':
        # Figure 
        fig = plt.figure(figsize=(15, 6))
        ax = plt.subplot(111)
    
        sns.boxplot(data=sb19df_features_stacked, x='feature', y='value', ax=ax, palette = ['grey'])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)
        
        # display graph
        st.pyplot(fig)
        st.write('An analysis of SB19’s tracks shows a wide variety of audio features implying that the group has been exploring and identifying their own'
                      ' personal “sound”')
        
    elif option == 'SB19 Audio Features':
        
        #df initialization
        kpop_df = pd.read_csv('data/k-pop_playlist_tracks_data.csv')
        kpop_df['tempo'] =  scaler.fit_transform(kpop_df[['tempo']])
        kpop_df['loudness'] = scaler.fit_transform(kpop_df[['loudness']])
        
        opm_df = pd.read_csv('data/opm_playlist_tracks_data.csv')
        opm_df['tempo'] =  scaler.fit_transform(opm_df[['tempo']])
        opm_df['loudness'] = scaler.fit_transform(opm_df[['loudness']])
        
        for col in ["valence", "danceability"]:    
  
            fig = plt.figure(figsize=(9, 3))
        
            sns.distplot(kpop_df[col], color='blue')
            sns.distplot(opm_df[col], color='yellow')
            
            plt.title('SB19 VS KPOP VS OPM: Histogram of ' + col)
            plt.ylabel('frequency')
            plt.xlabel(col)
            plt.show()
            
            st.pyplot(fig)
            
        st.write('Their tracks tend to have similarities with both KPOP and OPM audio features in terms of valence (positive vs negative vibes) and danceability')
        
    elif option == 'KPOP':
        
        # KPOP
        st.header('KPOPS ARE VIRAL')
        
        df1 = pd.read_csv('data/spotify_daily_charts.csv')
        df1['date'] = pd.to_datetime(df1['date'])
        df1 = df1.set_index('date')
        data = df1[(df1['artist']=='BTS') | (df1['artist']=='TWICE') | (df1['artist']=='BLACKPINK') 
              | (df1['artist']=='ENHYPEN') | (df1['artist']=='Sam Kim') | (df1['artist']=='SB19')].copy()
       
        top_streams = data.groupby(['artist'])[['streams']].resample('M').sum().reset_index()
        top_streams.head()
        fig= plt.figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        
        sns.lineplot(data = top_streams, x='date', y='streams', hue='artist')
        
        plt.ylabel('streams')
        plt.title('Top K-Pop Artist & SB19 Tracks Monthly Total Streams')
        
        st.pyplot(fig)
        st.write('KPOP artists have extremely erratic and volatile streaming volumes')
        
    elif option == 'Pinoy Tastes':
        image = Image.open('pinoy_taste.png')
            
        st.header('MAINLY PINOY PRIDE')
        st.image(image)
        st.write('OPM music is repeatedly played and constantly in high demand. Among mainstay tracks, OPM stay in the Top 100 for longer periods of time compared to KPOP songs')
        
# Results Page
elif my_page == 'Results':
    option = st.sidebar.selectbox('Options:', ['Next Move', 'Recommended Collaborations', 'KPOP Recommendation', 'OPM Recommendation', 'Final Insights'])
    
    st.title(f"{option}")
    
    if option == 'Next Move':
        st.header('What should SB19’S next move be?')
        st.write('Improve spotify performance by identifying best options to collaborate with KPOP & OPM artists')
        st.write("") # New Line
        st.header('Methodology')
        image = Image.open('methodology.png')
        st.image(image)
        
    elif option == 'Recommended Collaborations':
        st.header('Who should SB19 collaborate with?')
        
        st.subheader('OPM Recommendations')
        image = Image.open('opm.PNG')
        st.image(image)
        
        st.subheader('KPOP Recommendations')
        image2 = Image.open('kpop.png')
        st.image(image2)
        
    elif option == 'KPOP Recommendation':
        # Recommended kpop 
        data2 = merged_df[(merged_df['artist']=='BTS') | (merged_df['artist']=='ENHYPEN') | (merged_df['artist']=='TOMORROW X TOGETHER') 
                      | (merged_df['artist']=='BLACKPINK') | (merged_df['artist']=='TWICE') | (merged_df['artist']=='SB19')].copy()
        
        top_streams2 = data2.groupby(['artist'])[['streams']].sum().reset_index().sort_values(by = 'streams', ascending = False)
        
        fig= plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        
        sns.barplot(x = 'artist',
                    y = 'streams',
                    data = top_streams2,
                    ci = 0)

        plt.ylabel('streams (in millions)')
        plt.title('Total Spotify Streams (2017-2022)')
        
        st.header('Get viral with BTS & BLACKPINK')
        st.pyplot(fig)
        st.write('BTS and BLACKPINK are the most streamed male & female KPOP groups respectively')
        
    elif option == 'OPM Recommendation':
        # Recommended opm
        data3 = merged_df[(merged_df['artist']=='December Avenue') | (merged_df['artist']=='Zack Tabudio') | (merged_df['artist']=='Sarah Geronimo') 
              | (merged_df['artist']=='Skusta Clee') | (merged_df['artist']=='Shanti Dope') | (merged_df['artist']=='SB19')].copy()

        top_streams3 = data3.groupby(['artist'])[['streams']].sum().reset_index().sort_values(by = 'streams', ascending = False)
        
        fig= plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        
        sns.barplot(x = 'artist',
                    y = 'streams',
                    #hue = 'artist',
                    data = top_streams3,
                    ci = 0)
        
        plt.ylabel('streams (in millions)')
        plt.title('Total Spotify Streams (2017-2022)')
        
        st.header('Be a pop-rock mainstay with December Avenue')
        st.pyplot(fig)
        st.write('Collaborating with December Avenue, the most streamed OPM artist, would likely produce a long-standing pop/rock hit')

    elif option == 'Final Insights':
        image = Image.open('final_insights.png')
        st.image(image)
        
elif my_page == 'Playlists':
    
    st.title("Song Recommendations")
    
    st.header("Key Track: MAPA")
    
    st.video("https://www.youtube.com/watch?v=DDyr3DbTPtk")
    
    st.markdown('The most streamed SB19 track in Spotify is the single MAPA; hence, we selected it as seed track for the recommender engine. We looked into possible collaboration of SB19 with K-Pop and OPM artists and came up with two playlists.',unsafe_allow_html=False)
    
    st.header("Spotify Playlist: OPM Recommendations")
    playlist_uri = '2TTlW7hihIS1gmMCgzJ4M4'
    uri_link = 'https://open.spotify.com/embed/playlist/' + playlist_uri
    components.iframe(uri_link, height=300)
    
    st.header("Spotify Playlist: K-Pop Recommendations")
    playlist_uri = '54tF3K5EWEbnbyGlj9SmFs'
    uri_link = 'https://open.spotify.com/embed/playlist/' + playlist_uri
    components.iframe(uri_link, height=300)