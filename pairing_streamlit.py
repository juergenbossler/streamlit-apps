### "Pairing Simulation 4008 series ORs and IRs"
### myonic GmbH
### Jürgen Bossler - Digital Transformation Projekts
### 30.11.2021

import streamlit as st
#import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import collections
import math
from PIL import Image
import plotly.figure_factory as ff
#import plotly.express as px
import datetime

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title="Pairing Simulation", page_icon='GMD-Digital_icon.png', layout="wide", initial_sidebar_state='collapsed')

row1_1, row1_2, row1_3 = st.columns([1,1,4])

with row1_1:
    img = Image.open('company logo_myonic_rgb.jpg')
    st.image(img)

with row1_3:
    st.title('Pairing Simulation 4008 series IRs and ORs')

with row1_2:
    img2 = Image.open('GMD-Digital.png')
    st.image(img2)

col2, col3, col4 = st.columns([2,1,2])


# EINGABEN
st.sidebar.header('Input')

# Anzahl Auftragsgröße
n = st.sidebar.select_slider("lot size", options=[100,500,750,1000,1500,2000,2500,3000,3500,4000], value=750)

# Bereich Mittelwert Normalverteilung
st.sidebar.subheader("random values min mean / max mean")
medium_min = st.sidebar.slider("minimal mean (group number)", min_value=1, max_value=31, value=13)
medium_max = st.sidebar.slider("maximal mean (group number)", min_value=1, max_value=31, value=23)
if medium_min > medium_max:
    st.sidebar.error('minimal mean can not exceed maximal mean')
    st.stop()

st.sidebar.subheader("forced values min mean / max mean")
medium_AR_f = st.sidebar.number_input("mean OR", min_value=0, max_value=31, value=0, step=None, format=None, key=None, help="use 0 for generating random values", on_change=None, args=None, kwargs=None)
medium_IR_f = st.sidebar.number_input("mean IR", min_value=0, max_value=31, value=0, step=None, format=None, key=None, help="use 0 for generating random values", on_change=None, args=None, kwargs=None)

# Standardabweichung
st.sidebar.subheader("random values min stdev / max stdev")
stdev_min = st.sidebar.slider("minimal stdev (group number)", min_value=0.1, max_value=3.0, value=1.0)
stdev_max = st.sidebar.slider("maximal stdev (group number)", min_value=0.1, max_value=3.0, value=2.0)
if stdev_min >= stdev_max:
    st.sidebar.error('minimal stdev can not exceed maximal stdev')
    st.stop()

st.sidebar.subheader("forced values min stdev / max stdev")
stdev_AR_f = st.sidebar.number_input("stdev OR", min_value=0.0, max_value=3.0, value=0.0, step=None, format="%1.3f", key=None, help="use 0 for generating random values", on_change=None, args=None, kwargs=None)
stdev_IR_f = st.sidebar.number_input("stdev IR", min_value=0.0, max_value=3.0, value=0.0, step=None, format="%1.3f", key=None, help="use 0 for generating random values", on_change=None, args=None, kwargs=None)

# Ralu
st.sidebar.subheader("radial play")
ralu_min = st.sidebar.slider("radial play min", min_value=1, max_value=20, value=6)
ralu_max = st.sidebar.slider("radial play max", min_value=1, max_value=20, value=10)
if ralu_min >= ralu_max:
    st.sidebar.error('radial play min can not equal/exceed radial play max')
    st.stop()

ralu_diff = int((((ralu_max + ralu_min) / 2) - ralu_min) // 1)

# Puffergröße
st.sidebar.subheader("buffer size")
buffersize = st.sidebar.select_slider("buffer size", options=[10, 25, 50, 100, 150, 200, 500], value=100)

# Teile pro Stab
st.sidebar.subheader("pieces per needle")
pcsperneedle = st.sidebar.slider("pieces per needle", min_value=1, max_value=300, value=94)

AR_time = st.sidebar.number_input("sec/piece OR", min_value=0.0, max_value=15.0, value=10.0, step=None, format="%1.3f", key=None, on_change=None, args=None, kwargs=None)
IR_time = st.sidebar.number_input("sec/piece IR", min_value=0.0, max_value=15.0, value=10.0, step=None, format="%1.3f", key=None, on_change=None, args=None, kwargs=None)

clk1 = st.sidebar.button("calculate", key=1)
while not clk1:
    st.stop()
    pass


# Zufällige Parameter für Gauss-Verteilung ARe
# Zufallswert für Gruppenmittelwert im Bereich Mittelwert Normalverteilung (ganzzahlig)
if medium_AR_f == 0:
    medium_AR = random.randint(medium_min, medium_max)
else:
    medium_AR = medium_AR_f

# Zufallswert für Standardabweichung Normalverteilung (nicht ganzzahliger Anteil)
if stdev_AR_f == 0:
    stdev_AR = random.uniform(stdev_min, stdev_max)
else:
    stdev_AR = stdev_AR_f

# Zufallswert für Gruppenmittelwert im Bereich Mittelwert Normalverteilung (ganzzahlig)
if medium_IR_f == 0:
    medium_IR = random.randint(medium_min, medium_max)
else:
    medium_IR = medium_IR_f

# Zufallswert für Standardabweichung Normalverteilung (nicht ganzzahliger Anteil)
if stdev_IR_f == 0:
    stdev_IR = random.uniform(stdev_min, stdev_max)
else:
    stdev_IR = stdev_IR_f

# Zufallszahlen (Gruppen) ARe
# Leere Listen ARe
values_AR = []
values_AR_old = []

# Liste füllen mit Zufallszahlen (Gruppen) gemäß Normalverteilungsparameter
while len(values_AR) < n:
    value_AR = int(random.gauss(medium_AR, stdev_AR))
    values_AR.append(value_AR)

#mu, sigma = medium_AR, stdev_AR
#values_AR_np = np.random.normal(mu, sigma, n)

# Erstellt eine Kopie der Liste ARe
values_AR_old = values_AR_old + values_AR

# Zufallszahlen (Gruppen) IRe
# Leere Listen IRe
values_IR = []
values_IR_old = []

# Liste füllen mit Zufallszahlen (Gruppen) gemäß Normalverteilungsparameter
while len(values_IR) < n:
    value_IR = int(random.gauss(medium_IR, stdev_IR))
    values_IR.append(value_IR)

# Erstellt eine Kopie der Liste IRe
values_IR_old = values_IR_old + values_IR

# Zufallsauswahl von ARe (Puffergröße)
picknumber_list_AR = []
buffer_list_AR = []
picked_piece_list_AR = []

# Liste Puffergröße
picknumber_list_AR = list(range(n))

for i in range(buffersize):
    # Zufallsnummer aus Liste Puffergröße
    picknumber_AR = random.choice(picknumber_list_AR)
    # Liste ausgewählter Zufallsnummern
    picked_piece_list_AR.append(picknumber_AR)
    # Zufallsnummer aus Liste Puffergröße entfernen
    picknumber_list_AR.remove(picknumber_AR)
    # Auswahl Teil mit Zufallsmummer als Index
    picked_piece_AR = values_AR[picknumber_AR]
    # Liste (Gruppe) ausgeählter Teile
    buffer_list_AR.append(picked_piece_AR)

# Ausgewählte Teile mit Gruppe 0 ersetzen
for i in picked_piece_list_AR:
    values_AR[i] = 0

# Teile mit Gruppe 0 entfernen
for i in range(buffersize):
    values_AR.remove(0)


# Zufallsauswahl von IRe (Puffergröße)
picknumber_list_IR = []
buffer_list_IR = []
picked_piece_list_IR = []

# Liste Puffergröße
picknumber_list_IR = list(range(n))

for i in range(buffersize):
    # Zufallsnummer aus Liste Puffergröße
    picknumber_IR = random.choice(picknumber_list_IR)
    # Liste ausgewählter Zufallsnummern
    picked_piece_list_IR.append(picknumber_IR)
    # Zufallsnummer aus Liste Puffergröße entfernen
    picknumber_list_IR.remove(picknumber_IR)
    # Auswahl Teil mit Zufallsmummer als Index
    picked_piece_IR = values_IR[picknumber_IR]
    # Liste (Gruppe) ausgeählter Teile
    buffer_list_IR.append(picked_piece_IR)

# Ausgewählte Teile mit Gruppe 0 ersetzen
for i in picked_piece_list_IR:
    values_IR[i] = 0

# Teile mit Gruppe 0 entfernen
for i in range(buffersize):
    values_IR.remove(0)


# Histogrammme Gruppenverteilung ARe/IRe / Histogramme Gruppenverteilung Puffer ARe/IRe
with col2:
    with st.expander('Gaugeing'):

        st.header('Distributions')
        st.subheader('Random Numbers for mean/stdev')
        st.write("mean group number ORs: ", medium_AR)
        st.write("stdev ORs: ", round(stdev_AR, 3))
        st.write("mean group number IRs: ", medium_IR)
        st.write("stdev IRs: ", round(stdev_IR, 3))

        st.subheader('OR Plots')

        # Histogrammme Gruppenverteilung AR
        fig1 = plt.figure(figsize=(8, 4))
        plt.hist(values_AR_old, bins=20)#, color='g')
        plt.title("group number OR (lot size)")
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.axis([0, 31, 0, n / 3])
        x_scale = list(range(0, 32))
        plt.xticks(x_scale)
        st.pyplot(fig1)


        # Histogramme Gruppenverteilung Puffer ARe
        fig2 = plt.figure(figsize=(8, 4))
        plt.hist(buffer_list_AR, bins=20)#, color='c')
        plt.title("group number OR (buffer size)")
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.axis([0, 31, 0, n / 10])
        x_scale = list(range(0, 32))
        plt.xticks(x_scale)
        st.pyplot(fig2)

        st.subheader('IR Plots')

        # Histogrammme Gruppenverteilung IRe
        fig3 = plt.figure(figsize=(8, 4))
        plt.hist(values_IR_old, bins=20, color='tab:orange')
        plt.title("group number IR (lot size)")
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.axis([0, 31, 0, n / 3])
        x_scale = list(range(0, 32))
        plt.xticks(x_scale)
        st.pyplot(fig3)

        # Histogramme Gruppenverteilung Puffer IRe
        fig4 = plt.figure(figsize=(8, 4))
        plt.hist(buffer_list_IR, bins=20, color='tab:orange')
        plt.title("group number IR (buffer size)")
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.axis([0, 31, 0, n / 10])
        x_scale = list(range(0, 32))
        plt.xticks(x_scale)
        st.pyplot(fig4)

        st.subheader('IR/OR Plot')

        ## Histogramme Gruppenverteilung AR/IRe

        #fig4_1 = plt.figure(figsize=(8, 4))
        #plt.hist(values_AR_old, bins=20, alpha=0.7, label='OR', color='g')
        #plt.hist(values_IR_old, bins=20, alpha=0.7, label='IR', color='r')
        #plt.legend(loc='upper right')
        #plt.title("group number OR/IR (lot size)")
        #plt.xlabel("value")
        #plt.ylabel("frequency")
        #plt.axis([0, 31, 0, n / 2])
        #x_scale = list(range(0, 32))
        #plt.xticks(x_scale)
        #st.pyplot(fig4_1)

        # Histogramme Gruppenverteilung AR/IRe

         # Group data together
        hist_data = [values_AR_old, values_IR_old]

        group_labels = ['OR', 'IR']
        colors = ['rgb(0, 255, 0)', 'rgb(255, 0, 0)']

         # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size = [.5, .5], curve_type='normal', show_rug=False)#, colors=colors)

         # Plot!
        st.plotly_chart(fig, use_container_width=True)

with col3:

    with st.expander('Gaugeing time'):

        AR_time_all = datetime.timedelta(seconds=(AR_time*n))
        IR_time_all = datetime.timedelta(seconds=(IR_time*n))

        st.header('Total time required')
        #st.subheader('Needles OR')

        st.write('gaugeing time OR:', AR_time_all)
        st.write('gaugeing time IR:', IR_time_all)

    with st.expander('Sorting'):

        st.header('Needles (lot size)')
        st.subheader('Needles OR')

        # Auflistung Stabmengen Gesamtmenge
        st.write("(", pcsperneedle, " pcs/needle)")

        counter = collections.Counter(values_AR_old)
        for elem in sorted(counter.items()):
            needles = math.ceil(elem[1] / pcsperneedle)
            if needles > 1:
                needle_value = "needles"
            else:
                needle_value = "needle"
            st.write("group: ", elem[0], " "" --> ", needles, " ", needle_value)


        st.subheader('Needles IR')

        # Auflistung Stabmengen Gesamtmenge
        st.write("(", pcsperneedle, " pcs/needle)")

        counter = collections.Counter(values_IR_old)
        for elem in sorted(counter.items()):
            needles = math.ceil(elem[1] / pcsperneedle)
            if needles > 1:
                needle_value = "needles"
            else:
                needle_value = "needle"
            st.write("group: ", elem[0], " "" --> ", needles, " ", needle_value)



# Algorithmus Verpaaren (Ermittlung Kugelabmaße u. Berück. der RaLu-Toleranz)

# Leere Liste Kugelabmaße
diff = []

# Gruppen absteigend sortieren
values_AR_old.sort(reverse=True)
values_IR_old.sort(reverse=True)

len_AR = len(values_AR_old)
len_IR = len(values_IR_old)

for i in range(len(values_AR_old)):

    # Berechnung Gruppenunterschied AR/IR von kleinsten zu größten Gruppen
    diff_value = (values_AR_old[-1]) - (values_IR_old[-1])
    # Kugelabmaß 0 für Gruppenunterschied kleiner-gleich Ralu/2
    if diff_value in range((-ralu_diff), (ralu_diff + 1)):
        diff_value = 0
    # Kugelabmaß um Ralu/2 größer bei Gruppenunterschied größer Ralu/2 (negativ)
    elif diff_value <= ((-ralu_diff) - 1):
        diff_value = diff_value + ralu_diff
    # Kugelabmaß um Ralu/2 größer bei Gruppenunterschied größer Ralu/2 (positiv)
    elif diff_value >= (ralu_diff + 1):
        diff_value = diff_value - ralu_diff
    # Liste der berechneten Gruppenunterschiede (u. Berück. der RaLu-Toleranz)
    diff.append(diff_value)
    # Entfernen der verpaarten Gruppen aus Auftragsgröße
    values_AR_old.pop()
    values_IR_old.pop()


# Algorithmus Verpaaren (Ermittlung Kugelabmaße u. Berück. der RaLu-Toleranz) (Puffer)

# Leere Liste Kugelabmaße
diff_buff = []

# Gruppen absteigend sortieren
buffer_list_AR.sort(reverse=True)
buffer_list_IR.sort(reverse=True)

len_AR = len(buffer_list_AR)
len_IR = len(buffer_list_IR)

for i in range(len(buffer_list_AR)):

    # Berechnung Gruppenunterschied AR/IR von kleinsten zu größten Gruppen
    diff_value = (buffer_list_AR[-1]) - (buffer_list_IR[-1])
    # Kugelabmaß 0 für Gruppenunterschied kleiner-gleich Ralu/2
    if diff_value in range((-ralu_diff), (ralu_diff + 1)):
        diff_value = 0
    # Kugelabmaß um Ralu/2 größer bei Gruppenunterschied größer Ralu/2 (negativ)
    elif diff_value <= ((-ralu_diff) - 1):
        diff_value = diff_value + ralu_diff
    # Kugelabmaß um Ralu/2 größer bei Gruppenunterschied größer Ralu/2 (positiv)
    elif diff_value >= (ralu_diff + 1):
        diff_value = diff_value - ralu_diff
    # Liste der berechneten Gruppenunterschiede (u. Berück. der RaLu-Toleranz)
    diff_buff.append(diff_value)
    # Entfernen der verpaarten Gruppen aus Auftragsgröße
    buffer_list_AR.pop()
    buffer_list_IR.pop()



with col4:

    with st.expander('Pairing'):
       st.header("Pairing")
       st.subheader("Ball sizes plots")

       # Histogrammm Kugelabmaße
       fig5 = plt.figure(figsize=(8, 4))
       plt.hist(diff)
       plt.title("Ball sizes for pairing (lot size) [µm]")
       plt.xlabel("Nominal size + [µm]")
       plt.ylabel("Frequency")
       plt.axis([-10, 10, 0, n])
       x_scale = list(range(-10, 11))
       plt.xticks(x_scale)
       st.pyplot(fig5)

       # Auflistung benötigter Kugelabmaße

       # Auflistung Topf
       counter = collections.Counter(diff)
       # print("Benötigte Kugelabmaße: ",sorted(counter.keys()))
       # print("Zähler Kugelabmaße: ",counter)
       kugeldurchmesser = 0
       for elem in sorted(counter.items()):
           kugeldurchmesser += 1
       if kugeldurchmesser > 1:
           st.write(kugeldurchmesser, " ", "different ball sizes needed!")
       else:
           st.write(kugeldurchmesser, " ", "ball size needed!")

       # Histogrammm Kugelabmaße (Puffer)
       fig6 = plt.figure(figsize=(8, 4))
       plt.hist(diff_buff)
       plt.title("Ball sizes for pairing (buffer size) [µm]")
       plt.xlabel("Nominal size + [µm]")
       plt.ylabel("Frequency")
       plt.axis([-10, 10, 0, n / 5])
       x_scale = list(range(-10, 11))
       plt.xticks(x_scale)
       st.pyplot(fig6)

       # Auflistung Puffer
       kugeldurchmesser_buff = 0
       counter = collections.Counter(diff_buff)

       for elem in sorted(counter.items()):
           kugeldurchmesser_buff += 1

       if kugeldurchmesser_buff > 1:
           st.write(kugeldurchmesser_buff, " ", "different ball sizes needed!")
       else:
           st.write(kugeldurchmesser_buff, " ", "ball size needed!")


       # Zeitverlauf Kugelabmaße
       st.subheader("Ball size sequence plots (auto pairing)")
       fig = plt.figure(figsize=(30, 10))

       # Zeitverlauf Kugelabmaße
       fig7 = plt.figure(figsize=(8, 4))
       plt.plot(diff)
       plt.title("Sequence of ball sizes for pairing (lot size)")
       plt.ylabel("Nominal size + [µm]")
       plt.xlabel("Pairings")
       plt.axis([0, n, -10, 10])
       y_scale = list(range(-10, 11))
       plt.yticks(y_scale)
       st.pyplot(fig7)

       # Anzahl der Kugelwechsel
       kugelwechsel = 0
       item = 0
       for i in diff:
           if item == 0:
               old_val = i
               item += 1
           else:
               if i != old_val:
                   kugelwechsel += 1
                   old_val = i
               item += 1
       if kugelwechsel > 1:
           st.write("Changing ball size: ", kugelwechsel, " times!")
       else:
           st.write("Changing ball size: never!")


       # Zeitverlauf Kugelabmaße (Puffer)
       sub1 = fig.add_subplot(223)
       fig8 = plt.figure(figsize=(8, 4))
       plt.plot(diff_buff)
       plt.title("Sequence of ball sizes for pairing (buffer size)")
       plt.ylabel("Nominal size + [µm]")
       plt.xlabel("Pairings")
       plt.axis([0, buffersize, -10, 10])
       y_scale = list(range(-10, 11))
       plt.yticks(y_scale)
       st.pyplot(fig8)

       # Anzahl der Kugelwechsel Puffer
       kugelwechsel_buff = 0
       item = 0
       for i in diff_buff:
           if item == 0:
               old_val = i
               item += 1
           else:
               if i != old_val:
                   kugelwechsel_buff += 1
                   old_val = i
               item += 1

       if kugelwechsel_buff > 1:
           st.write("Changing ball size: ", kugelwechsel_buff, " times!")
       else:
           st.write("Changing ball size: never!")
