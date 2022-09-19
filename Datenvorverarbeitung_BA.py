# Datenvorverarbeitung_BA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# csv Messdaten als Pandas Df einlesen
def csv_einlesen(path_to_import):
    df = pd.read_csv(path_to_import, delimiter=",", header=0, skiprows=3, names=["Index", "Time", "Speed",  "POSF_X", "IQ_X","IQ_Y",
                                                                                "ERR_X","ERR_Y", "DTRQ", "ID_S1" ], index_col="Index")
    df.drop(["ID_S1"], axis=1, inplace=True)
    NaN = df.isnull().sum().sum()
    if NaN == 0:
        return df
    else:
        print("Anzahl NaN:", NaN)



# Segmete als Pandas DF einlesen
def csv_einlesen_Seg(path_to_import):
    df = pd.read_csv(path_to_import, delimiter=",", header=0, skiprows=3, names=["Index", "Time", "Speed",  "POSF_X", "IQ_X","IQ_Y",
                                                                                "ERR_X","ERR_Y", "DTRQ"], index_col="Index")
    df.drop(["POSF_X"], axis=1, inplace=True)
    df.drop(["Time"], axis=1, inplace=True)
    NaN = df.isnull().sum().sum()
    if NaN == 0:
        return df
    else:
        print("Anzahl NaN:", NaN)


# Signalanteile ohne Verfahrbewegung abschneiden
def df_Seg(df):
    df_POSF_X = df["POSF_X"]
    i=len(df_POSF_X)-1
    while df_POSF_X[i] < 100:
        Schnitt_max = i
        i=i-1
    n = 0
    while df_POSF_X[n] < 0.001:
        Schnitt_min = n
        n = n+1
    df = df[n:i]
    return df



# Signalanteile segmemtieren
def df_Seg_oE(df, Schnitt_l, Schnitt_r):
    df = df.iloc[Schnitt_l:Schnitt_r]
    return df

# df = df_Seg_oE(df, 3194, 5195)
# df = df_Seg_oE(df, 5196, 15195)
# df = df_Seg_oE(df, 15196, 17197)

# Pandas df Segmente als csv exportieren 
def csv_export(df, path_to_export):
    df.to_csv(path_to_export, sep =",")



# Datenpipline
def DV():
    Versuch = ["\V1"]
    Wiederholung = [ "W2", "W3", "W4", "W5", "W6",  "W8", "W9", "W10", "W11"]
    Segmente = ["Seg1", "Seg2", "Seg3"]

    for i in range(0, len(Versuch)):
        for n in range(0, len(Wiederholung)):
            path_to_import = "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten" + Versuch[i] +"\V1_" + Wiederholung[n] + "_vw300_n1910_ae0.2.csv"
            print(path_to_import, i, n)
            df = csv_einlesen(path_to_import)
            df = df_Seg(df)
            df1, df2, df3 = df_Seg_oE(df, 3194, 5195), df_Seg_oE(df, 5196, 15195), df_Seg_oE(df, 15196, 17197)
            path_to_export =  "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten" + Versuch[i] + "\Segmentierte_Daten" + Versuch[i] + "_" + Wiederholung[n] 
            csv_export(df1, path_to_export + Segmente[0] + "_vw300_n1910_ae0.2.csv")
            csv_export(df2, path_to_export + Segmente[1] + "_vw300_n1910_ae0.2.csv")
            csv_export(df3, path_to_export + Segmente[2] + "_vw300_n1910_ae0.2.csv")



