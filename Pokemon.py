# -*- coding: utf-8 -*-
"""
ESSAI d'application streamlit - challenge Pokemon WCS
"""

import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

###import de la DB source
df_pokemon = pd.read_csv('https://raw.githubusercontent.com/AgnesLorec/pokemon/main/data_pokemon.csv')



###Traitement des données source

#création d'un DF sans les noms + génération d'un index
df_pokemon_c = df_pokemon.dropna(subset=['Name'])
df_pokemon_c.reset_index(inplace=True)

#Type 1 : transformer en 0 ou 1
df_type1 = df_pokemon_c['Type 1'].str.get_dummies()

#Type 2 : transformer en 0 ou 1 -> get dummies car plus de 2 références
df_type2 = df_pokemon_c['Type 2'].str.get_dummies()
columns_name = df_type2.columns
for name in columns_name :
  df_type2.rename(columns={name : 'T2_' + name},inplace=True)

#HP / Attack / Defense / Sp. Atk / Sp. Def / Speed -> standardisation des données
nb_car = ['HP',	'Attack',	'Defense',	'Sp. Atk', 'Sp. Def', 'Speed']
X_nb_car = df_pokemon_c[nb_car]

scaler = StandardScaler().fit(X_nb_car)
X_nb_car_scaled = scaler.transform(X_nb_car)
df_nb_car = pd.DataFrame(X_nb_car_scaled,columns=nb_car)

#Generation : transformer en 0 ou 1 -> get dummies car plus de 2 références
df_generation = df_pokemon_c['Generation'].astype('str').str.get_dummies()
columns_name = df_generation.columns
for name in columns_name :
  df_generation.rename(columns={name : 'Generation_' + name},inplace=True)

#Création du DF avec toutes les infos standardisées
df_pokemon_AL = pd.concat([df_pokemon_c['Name'],df_type1, df_type2, df_nb_car, df_generation, df_pokemon_c['Legendary']],axis=1)



###Création du modèle : KNN
X_list = list(df_pokemon_AL.columns)
X_list.remove('Name')
X_list.remove('Legendary')

X = df_pokemon_AL[X_list]
y = df_pokemon_AL['Name']

#je cherche le pokemon le plus proche, je ne cherche pas à classifier en tant que tel mon pokemon.
##j'ai donc seulement besoin du plus proche voisin
model_KNN = KNeighborsClassifier(n_neighbors=1,weights='distance')

model_KNN.fit(X, y)

###Création de la fonction de détermination du plus proche voisin
def substitut_pokemon(poke_source):
  X_poke_source = df_pokemon_AL[X_list][df_pokemon_AL['Name'] == poke_source]
  list_X_poke_source = list(X_poke_source.iloc[0])
  distance, classement = model_KNN.kneighbors([list_X_poke_source])
  classement_condition =  classement[0]
  df_classement = df_pokemon_AL.iloc[classement_condition]
  df_classement.reset_index(drop=True, inplace=True)
  df_classement['Classement'] = [1] #10,9,8,7,6,5,4,3, 2,
  pokemon_substitut = df_classement['Name'].iloc[0]
  return pokemon_substitut


#######AFFICHAGE DE L'APPLICATION

#Titre de l'application
st.title('POKEMON')
st.write('_Trouve ton **pokemon**_')
'\n'
'\n'

st.sidebar.selectbox(
    'Que voulez-vous faire ?',
    ('Explorer les pokemon', 'Home phone', 'Mobile phone')
)

st.subheader('Quel légendaire veux-tu remplacer ?')
poke_source = st.selectbox(label='Sélectionne un pokemon',options=['Mewtwo', 'Lugia', 'Rayquaza', 'Giratina Altered Forme', 'Giratina Origin Forme', 'Dialga', 'Palkia'])

st.subheader('Le pokemon le plus proche est :')
resultat = substitut_pokemon(poke_source)
st.write(resultat)



