import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Signalsegmentierung as Seg
import Datenvorverarbeitung_BA as dv
import Datenvorverarbeitung_Kräfte as dvk
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline





# Merkmalsvektorberechnung

def Merk_Vec(df_Seg1, df_Seg2, df_Seg3, df_Kraft_Z):
    
    # Standdardabweichung
    df_std1 = df_Seg1.std(axis=0)
    df_std2 = df_Seg2.std(axis=0) 
    df_std3 = df_Seg3.std(axis=0)
    df_stdZ = df_Kraft_Z.std(axis=0)

     # Maximalwert
    df_max1 = df_Seg1.max(axis=0)
    df_max2 = df_Seg2.max(axis=0)
    df_max3 = df_Seg3.max(axis=0)
    df_maxZ = df_Kraft_Z.max(axis=0)

    # Minimalwert
    df_min1 = df_Seg1.min(axis=0)
    df_min2 = df_Seg2.min(axis=0)
    df_min3 = df_Seg3.min(axis=0)
    df_minZ = df_Kraft_Z.min(axis=0)

    # Median
    df_med1 = df_Seg1.median(axis=0)
    df_med2 = df_Seg2.median(axis=0)
    df_med3 = df_Seg3.median(axis=0)
    df_medZ = df_Kraft_Z.median(axis=0)


    # Schiefe
    df_skew1 = df_Seg1.skew(axis=0)
    df_skew2 = df_Seg2.skew(axis=0)
    df_skew3 = df_Seg3.skew(axis=0)
    df_skewZ = df_Kraft_Z.skew(axis=0)

    # Wölbung
    df_kur1 = df_Seg1.kurtosis(axis=0)
    df_kur2 = df_Seg2.kurtosis(axis=0)
    df_kur3 = df_Seg3.kurtosis(axis=0)
    df_kurZ = df_Kraft_Z.kurtosis(axis=0)

    # Varianz
    df_var1 = df_Seg1.var(axis=0)
    df_var2 = df_Seg2.var(axis=0)
    df_var3 = df_Seg3.var(axis=0)
    df_varZ = df_Kraft_Z.var(axis=0)
    

    # verbinden
    df = pd.concat([df_std1, df_std2, df_std3, df_stdZ,
                    df_max1, df_max2, df_max3, df_maxZ,
                    df_min1, df_min2, df_min3, df_minZ,
                    df_med1, df_med2, df_med3, df_medZ,
                    df_skew1, df_skew2, df_skew3, df_skewZ,
                    df_kur1, df_kur2, df_kur3, df_kurZ, 
                    df_var1, df_var2, df_var3, df_varZ], axis=0, keys=["std1","std2","std3", "stdZ",
                                                            "max1","max2","max3", "maxZ",
                                                            "min1", "min2", "min3", "minZ",
                                                            "med1", "med2", "med3", "medZ",
                                                            "skew1", "skew2", "skew3", "skewZ",
                                                            "kur1", "kur2", "kur3", "kurZ",
                                                            "var1", "var2", "var3", "varZ"])
    
    return df

Versuch = ["\V1"]
Wiederholung = [ "W2", "W3", "W4", "W5", "W6",  "W8", "W9", "W10", "W11"]
Segmente = ["Seg1", "Seg2", "Seg3"]
path_to_import = "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten" + Versuch[0] + "\Segmentierte_Daten" + Versuch[0] +"_" + Wiederholung[1] 
df_Seg1, df_Seg2, df_Seg3 = dv.csv_einlesen_Seg(path_to_import + Segmente[0] + "_vw300_n1910_ae0.2.csv"), dv.csv_einlesen_Seg(path_to_import + Segmente[1] + "_vw300_n1910_ae0.2.csv"), dv.csv_einlesen_Seg(path_to_import + Segmente[2] + "_vw300_n1910_ae0.2.csv")
path_to_import_Kraft = "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten" + Versuch[0] + "\Kraftmessung" + Versuch[0] +"_" + Wiederholung[1] 
df_Kraft = dvk.csv_einlesen_Kraft(path_to_import_Kraft +  "_Kraft_vw300_n1910_ae0.2.csv" )
df_Kraft = dvk.Kraft_Seg(df_Kraft)
df_Kraft_Z = dvk.lowpassfilter_1(df_Kraft["Kraft_Z"])
df_Seg = Merk_Vec(df_Seg1, df_Seg2, df_Seg3, df_Kraft_Z)
print(df_Seg["std1"])
df_Seg = pd.DataFrame(df_Seg)
# df_Seg.columns=["Wert"]

df_Seg = df_Seg.to_numpy()

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values



num_pipeline = Pipeline([
    ("inputer", SimpleImputer(strategy="median")),
    ("min_max_scaler", MinMaxScaler()),
])

df_Seg = num_pipeline.fit_transform(df_Seg)
df_Seg = pd.DataFrame(df_Seg)
dv.csv_export(df_Seg, "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten"+ Versuch[0] + "\Merkmalsvektor" + Versuch[0] + "_" + Wiederholung[1] + "Seg_Mvec_Test_vw300_n1910_ae0.2.csv" )








def Merk_Vec_R():
    Versuch = ["\V1"]
    Wiederholung = [ "W2", "W3", "W4", "W5", "W6",  "W8", "W9", "W10", "W11"]
    Segmente = ["Seg1", "Seg2", "Seg3"]
    for i in range(0, len(Versuch)):
        for n in range(0, len(Wiederholung)):
            path_to_import = "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten" + Versuch[i] + "\Segmentierte_Daten" + Versuch[i] +"_" + Wiederholung[n] 
            df_Seg1, df_Seg2, df_Seg3 = dv.csv_einlesen_Seg(path_to_import + Segmente[0] + "_vw300_n1910_ae0.2.csv"), dv.csv_einlesen_Seg(path_to_import + Segmente[1] + "_vw300_n1910_ae0.2.csv"), dv.csv_einlesen_Seg(path_to_import + Segmente[2] + "_vw300_n1910_ae0.2.csv")
            path_to_import_Kraft = "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten" + Versuch[i] + "\Kraftmessung" + Versuch[i] +"_" + Wiederholung[n] 
            df_Kraft = dvk.csv_einlesen_Kraft(path_to_import_Kraft +  "_Kraft_vw300_n1910_ae0.2.csv" )
            df_Kraft = dvk.Kraft_Seg(df_Kraft)
            df_Kraft_Z = dvk.lowpassfilter_1(df_Kraft["Kraft_Z"])
            df_Seg = Merk_Vec(df_Seg1, df_Seg2, df_Seg3, df_Kraft_Z)
            
            dv.csv_export(df_Seg, "C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten"+ Versuch[i] + "\Merkmalsvektor" + Versuch[i] + "_" + Wiederholung[n] + "Seg_Mvec_vw300_n1910_ae0.2.csv" )



# df = pd.read_csv("C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten\V1\Merkmalsvektor\V1_W2Seg_Mvec_vw300_n1910_ae0.2.csv", delimiter=",", index_col=[0])
# df1 = pd.read_csv("C:\\Users\sebas\OneDrive\Dokumente\Bachelorarbeit\Messdaten\V1\Merkmalsvektor\V1_W11Seg_Mvec_vw300_n1910_ae0.2.csv", delimiter=",", index_col=[0])
def PCA_function(df):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal_component_1', 'principal_component_2'])
    return principalDf

# df = PCA_function(df)
# df1 = PCA_function(df1)
# # print(principalDf)
# fig, ax = plt.subplots()
# scatter = ax.scatter(df.principal_component_1, df.principal_component_2, cmap='gray', marker="o")
# scatter1 = ax.scatter(df1.principal_component_1, df1.principal_component_2, cmap="gray", marker = "x")    
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="upper right", title="Classes")
# ax.add_artist(legend1)
# plt.show()
# plt.legend()






# Frequenzbandsumme berechnen
def freq_band(np_array_kraft, freq_band_min, freq_band_max):
    Seg_Kraft  = Seg.Seg_Kraft(np_array_kraft)
    Seg_Kraft_fft = Seg.fft(Seg_Kraft)
    xf = Seg_Kraft_fft[0]
    yf = Seg_Kraft_fft[1]
    
    # Schleife für berechnen der Grenzen des Frequenzbandes
    n = len(xf)-1
    while xf[n] >= freq_band_min:  #Untergrenze
        f_min = n
        n = n-1
    
    i = 0
    while xf[i] <= freq_band_max: #Obergrenze
        f_max = i
        i = i+1
    
    fft_sum = sum(i for i in np.abs(yf[f_min:f_max]))
    return fft_sum 

    


