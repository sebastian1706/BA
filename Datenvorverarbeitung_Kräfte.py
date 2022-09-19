# Datenvorverarbeitung Kräfte

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from pywt import wavedec
import pywt
from skimage.restoration import denoise_wavelet



def csv_einlesen_Kraft(path_to_import):
    df = pd.read_csv(path_to_import, delimiter=",", header=0, names= ["Zeitstempel", "Kraft_X", "Kraft_Y", "Kraft_Z"])
    NaN = df.isnull().sum().sum()
    if NaN == 0:
        return df
    else:
        print("Anzahl NaN:", NaN)




def Kraft_Seg(df):
    df = df.iloc[400000:1100000]
    return df






def fft(df):
    numpy_array = df.to_numpy()
    N = len(numpy_array)
    sampling_rate = 44100
    yf = rfft(numpy_array[:,2])
    xf = rfftfreq(N, 1/sampling_rate)
    
  
    plt.plot(xf, np.abs(yf))
    plt.show()
# fft(df)




def lowpassfilter(df, thresh = 1, wavelet="db4" ):
    thresh = thresh*np.nanmax(df)
    coeff = pywt.wavedec(df, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal


def lowpassfilter_1(df, thresh = 0.4, wavelet="db4"):
    thresh = thresh*np.nanmax(df)
    coeffs = pywt.wavedec(df, wavelet, mode="per" )
    df = pd.DataFrame(coeffs)
    df = df.transpose()
    return df










