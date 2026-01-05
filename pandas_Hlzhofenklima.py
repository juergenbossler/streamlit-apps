import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import glob

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title="Herlazhofen Klima 2007-2025", page_icon='GMD-Digital_icon.png', initial_sidebar_state='collapsed', layout="wide")

#row1_1, row1_2, row1_3 = st.columns([1,1,4])
row1_1, row1_3 = st.columns([1,4])

with row1_1:
    #img = Image.open('company logo_myonic_rgb.jpg')
    img = Image.open('2026-01-04_10h52_14.jpg')
       
    st.image(img)

with row1_3:
    st.title('Lufttemp. Herlazhofen 2007-2025')

#with row1_2:
#    img2 = Image.open('GMD-Digital.png')
#    st.image(img2)


years = [0,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
months = [0,1,2,3,4,5,6,7,8,9,10,11,12]
days = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

# SIDEBAR ------------------------------------------------------------------------

st.sidebar.subheader('Auswahl')

#st.sidebar.dataframe(years)#, 2500, 500)

option_von_jahr = st.sidebar.selectbox("Filtern Datum von Jahr", years)
option_bis_jahr = st.sidebar.selectbox("Filtern Datum bis Jahr", years)
option_von_monat = st.sidebar.selectbox("Filtern Datum von Monat", months)
option_bis_monat = st.sidebar.selectbox("Filtern Datum bis Monat", months)
option_von_tag = st.sidebar.selectbox("Filtern Datum von Tag", days)
option_bis_tag = st.sidebar.selectbox("Filtern Datum bis Tag", days)
option_von_stunde = st.sidebar.selectbox("Filtern Datum von Stunde", hours)
option_bis_stunde = st.sidebar.selectbox("Filtern Datum bis Stunde", hours)


clk1 = st.sidebar.button("Show", key=1)
while not clk1:
    st.stop()
    pass




#all_files = []
#for name in glob.glob('klima[1-8].txt'):#('../klima[1-8].txt'):
#    all_files.append(name)
#
#all_reads = []
#x = 0
#for n in all_files:
#    klima_week = pd.read_csv(str(all_files[x]), parse_dates=['Time'], header=0)
#    all_reads.append(klima_week)
#    x += 1
#
#klima_concat = pd.concat(all_reads)

#print(pd.read_csv('data_new1.csv').columns)

klima_hlzhofen = pd.read_csv('data_new2.csv', sep=',', parse_dates=['Zeitstempel'])


klima_hlzhofen.index = klima_hlzhofen['Zeitstempel']


#klima_hlzhofen = klima_hlzhofen[(klima_hlzhofen.index.weekday >= 5) & (klima_hlzhofen.index.month == 7) & (klima_hlzhofen.index.hour >= 4) & (klima_hlzhofen.index.hour <= 5)] 
#klima_hlzhofen = klima_hlzhofen[(klima_hlzhofen.index.weekday >= 5) & (klima_hlzhofen.index.month == 7)] 
#klima_hlzhofen = klima_hlzhofen[(klima_hlzhofen.index.day >= 27) & (klima_hlzhofen.index.day <= 31) & (klima_hlzhofen.index.month == 4)]
#klima_hlzhofen = klima_hlzhofen[(klima_hlzhofen.index.day >= 27) & (klima_hlzhofen.index.day <= 31) & (klima_hlzhofen.index.month == 4) & (klima_hlzhofen.index.hour >= 14) & (klima_hlzhofen.index.hour <= 18)] 
#klima_hlzhofen = klima_hlzhofen[(klima_hlzhofen.index.year == 2018) & (klima_hlzhofen.index.month == 4)] 
#headers = ["Produkt_Code","SDO_ID","Zeitstempel","Wert","Qualitaet_Byte","Qualitaet_Niveau"]
#klima_concat.columns = headers




#months = [1,2,3,4,5,6,7,8,9,10,11,12]


if option_von_jahr != 0 and option_von_monat != 0  and option_von_tag != 0:
    klima_hlzhofen = klima_hlzhofen[(klima_hlzhofen.index.year >= option_von_jahr) & (klima_hlzhofen.index.year <= option_bis_jahr) & (klima_hlzhofen.index.month >= option_von_monat) & (klima_hlzhofen.index.month <= option_bis_monat) & (klima_hlzhofen.index.day >= option_von_tag) & (klima_hlzhofen.index.day <= option_bis_tag)& (klima_hlzhofen.index.hour >= option_von_stunde) & (klima_hlzhofen.index.hour <= option_bis_stunde)]


mean = klima_hlzhofen['Wert']
date = klima_hlzhofen['Zeitstempel']

#humidity = klima_concat['Humidity']

row2_1, row2_2 = st.columns([1, 1])

with row2_1:
    st.dataframe(data=klima_hlzhofen, width=600, height=300)  # Same as st.write()

with row2_2:
    #date_text = str('Durchschnittstemperatur [°C] von ' + str(start_date) + ' bis ' + str(end_date))
    st.metric('Durchschnittstemperatur [°C]', round(mean.mean(), 2))
    #st.metric('Durchschnittliche Luftfeuchtigkeit [%]', round(humidity.mean(), 2))

fig = px.line(klima_hlzhofen, x='Zeitstempel', y=['Wert'], title='Time Series with Range Slider and Selectors')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(fig, use_container_width=True)

