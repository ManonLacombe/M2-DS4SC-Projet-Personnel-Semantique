import networkx as nx
import pandas as pd
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from scipy.spatial.distance import squareform
from CED import ced_sm  
from Context_function import gaussian
from dis_and_sim import halkidi, mval_sim_ignore_null

# Chargement des ontologies
path_ontoWeather = "data/WeatherOntology-WeatherConditions.txt"
path_ontoTemporal = "data/WeatherOntology-TemporalFactors.txt"
path_ontoAtmospheric = "data/WeatherOntology-AtmosphericFactors.txt"

ontologyWeather = nx.read_adjlist(path_ontoWeather, create_using=nx.DiGraph)
ontologyTemporal = nx.read_adjlist(path_ontoTemporal, create_using=nx.DiGraph)
ontologyAtmospheric = nx.read_adjlist(path_ontoAtmospheric, create_using=nx.DiGraph)

ontos = [ontologyWeather, ontologyAtmospheric, ontologyTemporal]

# Chargement des données de météo
filename = 'data/MergeLinesComp.csv' #ou MergeLines.csv
dataset = pd.read_csv(filename)

# Suppression des colonnes inutiles
dataset_CED = dataset.drop(columns=['name', 'latitude', 'longitude', 'duration'])

# Définition des groupes de colonnes (dimensions)
colonnes_datetime = ['datetime']
colonnes_weather = ['Temperature', 'Visibility', 'Wind', 'Rain', 'Snow']
colonnes_atmospheric = ['CloudCover', 'Pressure', 'Humidity', 'DewPoint', 'UVIndex', 'Radiation']
colonnes_temporal = ['Season', 'Timeframe']

# Réorganisation des colonnes du df
dataset_CED = dataset_CED[colonnes_datetime + colonnes_weather + colonnes_atmospheric + colonnes_temporal]

# Conversion de la colonne datetime en format datetime
dataset_CED['datetime'] = pd.to_datetime(dataset_CED['datetime'])

# Dictionnaire pour stocker les séquences
s_dict = defaultdict(list)

# Construction des séquences
for _, row in dataset_CED.iterrows():
    date_key = row['datetime'].strftime('%Y-%m-%d')  # Regrouper par date
    structured_row = [
        [str(row[col]) for col in colonnes_weather],  # Groupe météo
        [str(row[col]) for col in colonnes_atmospheric],  # Groupe atmosphérique
        [str(row[col]) for col in colonnes_temporal]  # Groupe temporel
    ]
    s_dict[date_key].append(structured_row)

# Transformation en liste
s_list = [entries for _, entries in sorted(s_dict.items())]

# Définition de la fonction de similarité
def sim(x, y):
    return mval_sim_ignore_null(x, y, ontos)

# Conversion en format numpy
np_seqs = []
for seq in s_list:
    seqA = np.empty((len(seq),), dtype=object)
    for k in range(len(seq)):
        seqA[k] = seq[k]
    np_seqs.append(seqA)

del s_list  # Libérer la mémoire

#print("seq 0 : ",np_seqs[0])
#print("seq 1 : ", np_seqs[1])

# Fonction pour comparer spécifiquement deux séquences spécifiques
def distance_between_two_sequences(A, B):
    print(f"Computing MULTIDIMENSIONAL CED distance between sequence {A} and sequence {B} : ")
    distance = ced_sm(np_seqs[A], np_seqs[B], sim, sim, gaussian)
    print(f"Multidimensional CED({A}, {B}) = {distance}")
    return distance

if __name__ == '__main__':

    # multiprocessing
    mp.set_start_method('spawn', force=True)

    date_keys = sorted(s_dict.keys())
    
    A, B = None, None  # indices de séquences à comparer ou None pour comparer toutes les séquences du dataset
    if A is not None and B is not None :
        if A <= len(np_seqs) and B <= len(np_seqs):
            distance_between_two_sequences(A, B)
        else : 
            print(f"Attention, indice(s) supérieur(s) au nombre de séquences présent")
    else:
        # Calcul complet de la matrice de distance CED multidimensionnel
        print("Computing distance matrix - MULTIDIMENSIONAL CED")
        pool = mp.Pool(mp.cpu_count())
        result = pool.starmap(ced_sm, [(np_seqs[i], np_seqs[j], sim, sim, gaussian, 0.) for i in range(len(np_seqs)) for j in range(i + 1, len(np_seqs))])
        pool.close()
        pool.join()

        CED_matrix = squareform(np.array(result))

        # Arrondir les valeurs à trois chiffres après la virgule
        matrix = np.round(CED_matrix, 3)

        # Résultats dans df
        df = pd.DataFrame(matrix, index=date_keys, columns=date_keys)

        # Fichier CSV de sortie
        df.to_csv("data/dis_matrix_ced_multidim2014_2024.csv") # ou dis_matrix_ced_multidim2024.csv