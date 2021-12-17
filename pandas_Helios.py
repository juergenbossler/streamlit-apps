import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# SETTING PAGE CONFIG TO WIDE MODE -----------------------------------------------

st.set_page_config(page_title="Helios Data", page_icon='GMD-Digital_icon.png', layout="wide", initial_sidebar_state='collapsed')

row1_1, row1_2, row1_3 = st.columns([6,2,1])

with row1_1:
    st.title('Helios Data 2021')

with row1_2:
    img = Image.open('company logo_myonic_rgb.jpg')
    st.image(img)

with row1_3:
    img2 = Image.open('GMD-Digital.png')
    st.image(img2)

col1, col2 = st.columns([2,1])

# READ DATA IN CACHE -------------------------------------------------------------

@st.cache(allow_output_mutation=True)
def read_in_data():
    helios_raw = pd.read_excel("HELIOS_Rohdaten.xlsx", sheet_name="Tabelle1", dtype= {'Ralu': str, 'Materialnummer IR': str, 'Materialnummer OR': str, 'Bezeichnung IR': str, 'Bezeichnung OR': str})
    helios_raw1 = helios_raw.fillna(0)
    helios_raw3 = helios_raw1[helios_raw1["Stückzahl pro Nadel"] > 0]
    #counts_mat = helios_raw2['Materialnummer IR'].value_counts()
    # #print(counts_mat)
    helios_raw3.insert(70, 'selected Type', 0)
    helios_raw3.insert(71, 'Mittlere Gruppe IRe', 0.0)
    helios_raw3.insert(72, 'Standardabweichung IRe', 0.0)
    helios_raw3.insert(73, 'Mittlere Gruppe ARe', 0.0)
    helios_raw3.insert(74, 'Standardabweichung ARe', 0.0)
    #helios_raw4 = helios_raw3.convert_dtypes()
    #print(helios_raw3.dtypes)

    return helios_raw3

helios_raw2 = read_in_data()

# NUMBER OF LOTS PER BEARING TYPE ------------------------------------------------

loc_name = helios_raw2.columns.get_loc('Bezeichnung OR')
baureihen = []
for i in range(len(helios_raw2.index)):
    bez = helios_raw2.iloc[i, loc_name].split()
    # print(bez)
    for t in bez:
        # print(t)
        if 'X' in t:
            text = t
        else:
            continue
        baureihe = text.split('X')[0]
        baureihen.append(baureihe)

unique_baureihen = list(set(baureihen))

anz_dict = {}
for br in range(len(unique_baureihen)):
    b_reihe = unique_baureihen[br]
    anz_dict[b_reihe] = baureihen.count(b_reihe)

sorted_anz_dict = dict(sorted(anz_dict.items(),
                              key=lambda item: item[1],
                              reverse=True))

df_sorted_anz_dict = pd.DataFrame.from_dict(sorted_anz_dict, orient='index')
df_sorted_anz_dict.reset_index(level=0, inplace=True)
df_sorted_anz_dict.columns = ['bearing type', 'number of lots']


loc_name = helios_raw2.columns.get_loc('Bezeichnung OR')
baureihen = []
for i in range(len(helios_raw2.index)):
    bez = helios_raw2.iloc[i, loc_name].split()
    for t in bez:
        if 'X' in t:
            text = t
        else:
            continue
        baureihe = text.split('X')[0]
        baureihen.append(baureihe)

unique_baureihen = list(set(baureihen))


# SIDEBAR ------------------------------------------------------------------------

st.sidebar.subheader('summary lots')

st.sidebar.dataframe(df_sorted_anz_dict)#, 2500, 500)

option = st.sidebar.selectbox('filter bearing type', df_sorted_anz_dict)

clk1 = st.sidebar.button("calculate", key=1)
while not clk1:
    st.stop()
    pass

# FILTER DATAFRAME BY INPUT ------------------------------------------------------

for i in range(len(helios_raw2.index)):
    if option in helios_raw2.iloc[i, loc_name]:
        helios_raw2.iat[i, 70] = 1
    else:
        helios_raw2.iat[i, 70] = 0

helios_raw2 = helios_raw2[helios_raw2["selected Type"] > 0]

# CALCULATION OF MEAN GROUP AND STDEV --------------------------------------------

start_col_IR = (helios_raw2.columns.get_loc('I_2') - 2)
start_col_OR = (helios_raw2.columns.get_loc('O_2') - 2)
row_count = len(helios_raw2.index)

loc_m_IR = helios_raw2.columns.get_loc('Mittlere Gruppe IRe')
loc_m_OR = helios_raw2.columns.get_loc('Mittlere Gruppe ARe')
loc_stdev_IR = helios_raw2.columns.get_loc('Standardabweichung IRe')
loc_stdev_OR = helios_raw2.columns.get_loc('Standardabweichung ARe')

for i in range(len(helios_raw2.index)):
    groups_array = np.arange(2, 32, dtype=int)
    count_IR = np.array([])
    for col in groups_array:
        count_value = int(helios_raw2.iloc[i, (start_col_IR+col)])
        #print('count_value='+ str(count_value) + '  ' + 'col=' + str(col))
        if helios_raw2.iloc[i, (start_col_IR+col)] == 0:
            continue
        else:
            for count in range(count_value):
                count_IR = np.append(count_IR, col)

    m = count_IR.mean()
    std = count_IR.std()

    helios_raw2.iat[i,loc_m_IR] = m
    helios_raw2.iat[i,loc_stdev_IR] = std

for i in range(len(helios_raw2.index)):
    groups_array = np.arange(2, 32, dtype=int)
    count_OR = np.array([])
    for col in groups_array:
        count_value = int(helios_raw2.iloc[i, (start_col_OR+col)])
        #print('count_value='+ str(count_value) + '  ' + 'col=' + str(col))
        if helios_raw2.iloc[i, (start_col_OR+col)] == 0:
            continue
        else:
            for count in range(count_value):
                count_OR = np.append(count_OR, col)

    m = count_OR.mean()
    std = count_OR.std()

    helios_raw2.iat[i,loc_m_OR] = m
    helios_raw2.iat[i,loc_stdev_OR] = std


with col1:
    st.header('Selected bearing type' + '     ' + option)
    st.write(helios_raw2)

    st.metric('number of lots', row_count)

    # CALCULATE MEANS OF MEAN GROUP AND STDEV ------------------------------------

    st.write('IRs - mean group over all', helios_raw2['Mittlere Gruppe IRe'].mean())
    st.write('IRs - mean stdev over all', helios_raw2['Standardabweichung IRe'].mean())
    st.write('ORs - mean group over all', helios_raw2['Mittlere Gruppe ARe'].mean())
    st.write('ORs - mean stdev over all', helios_raw2['Standardabweichung ARe'].mean())







