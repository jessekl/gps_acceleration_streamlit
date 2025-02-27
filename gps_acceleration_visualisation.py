import streamlit as st
import pandas as pd
import folium 
from streamlit_folium import st_folium
import numpy as np
from math import radians, cos, sin, asin, sqrt
from scipy.signal import butter, filtfilt

df_gps = pd.read_csv("https://raw.githubusercontent.com/jessekl/gps_acceleration_streamlit/refs/heads/main/Location.csv")
df_la = pd.read_csv("https://raw.githubusercontent.com/jessekl/gps_acceleration_streamlit/refs/heads/main/Linear%20Acceleration.csv")

def trim_time(df, start, end):
    max_time = df['Time (s)'].max()
    df = df[df["Time (s)"] >= start]
    df = df[df["Time (s)"] <= max_time - end]
    df.reset_index(drop=True, inplace=True)
    return df
time_from_start = 12
time_from_end = 4
df_gps = trim_time(df_gps, time_from_start, time_from_end)
df_la = trim_time(df_la, time_from_start, time_from_end)

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c*r*1000

df_gps['dist'] = np.zeros(len(df_gps))
df_gps['time_diff'] = np.zeros(len(df_gps))
for i in range(len(df_gps)-1):
    #Kahden peräkkäisen pisteen välimatka
    df_gps.loc[i, 'dist'] = haversine(
        df_gps['Longitude (°)'][i], 
        df_gps['Latitude (°)'][i], 
        df_gps['Longitude (°)'][i+1], 
        df_gps['Latitude (°)'][i+1]
    )
    df_gps.loc[i,'time_diff'] = df_gps['Time (s)'][i+1] - df_gps['Time (s)'][i]

df_gps['Speed (m/s)'] = df_gps['dist']/df_gps['time_diff']
df_gps['Distance (km)'] = np.cumsum(df_gps['dist']) / 1000

def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

# Acceleration data analysis

data = df_la['Linear Acceleration x (m/s^2)']
t = df_la['Time (s)'].max()
n = len(df_la['Time (s)'])
fs = n/t
nyq = fs/2 
order = 3
cutoff = 2
data_lowpass = butter_lowpass_filter(data, cutoff, fs, nyq, order)
filt_signal = butter_highpass_filter(data_lowpass, cutoff, fs, nyq, order)
df_la['Filtered acceleration x'] = filt_signal

peaks = 0
for i in range(n - 1):
    if filt_signal[i] > filt_signal[i - 1] and filt_signal[i] > filt_signal[i + 1]:
        peaks = peaks + 1

f = df_la['Linear Acceleration x (m/s^2)']
N = len(f)
dt = np.max(t)/N
fourier = np.fft.fft(f,N)
psd = fourier*np.conj(fourier)/N
freq = np.fft.fftfreq(N,dt)
L = np.arange(1, int(N/2)) # 
f_max = freq[L][psd[L] == np.max(psd[L])][0]
askeleen_aika = 1/f_max
askelmaara = np.max(t)*f_max

askelpituus = df_gps['Distance (km)'].max() / askelmaara * 1000 * 100

st.header('Kävelyn analyysi')
# Show stats
st.write(f"Askelmäärä suodatuksen avulla: {peaks} askelta")
st.write(f"Askelmäärä Fourier-analyysin avulla: {int(askelmaara)} askelta")
st.metric("Keskinopeus:" ,f"{df_gps['Speed (m/s)'].mean():.3f} m/s")
st.metric("Kokonaismatka:" ,f"{df_gps['Distance (km)'].max():.3f} km")
st.write(f"Askelpituus: {round(askelpituus)} cm")
# Draw filtered acceleration component
st.subheader('Suodatettu kiihtyvyyden x-komponentti')
st.line_chart(df_la, x = 'Time (s)', y = 'Filtered acceleration x' , y_label = 'Kiihtyvyys x-akselilla',x_label = 'Aika (s)')
# Draw Fourier analysis
st.subheader('Tehospektri')
# Dataframe for st.line_chart
df_fourier = pd.DataFrame({
    'freq (Hz)': freq[L],
    'PSD' : psd[L].real
})
df_fourier.set_index('freq (Hz)', inplace=True)
st.line_chart(df_fourier, y_label = 'Teho',x_label = 'Taajuus [Hz]')

# Draw map
start_lat = df_gps['Latitude (°)'].mean()
start_long = df_gps['Longitude (°)'].mean()
map = folium.Map(location = [start_lat,start_long], zoom_start = 17)
folium.PolyLine(df_gps[['Latitude (°)','Longitude (°)']], color = 'blue', weight = 3.5, opacity = 1).add_to(map)
st_map = st_folium(map, width=900, height=650)