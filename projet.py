# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:59:58 2024

@author: Thibault Grivel
@binome: Samuel Odoardi
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import arbre_dec_bin
from RNN import NeuralNet
#Importation du fichier
synthetique = pd.read_csv("synthetic.csv")
#Nombre de colonnes/Attributs en comptant y
print("Nombre d'attribut en comptant le label : ", synthetique.shape[1])
#Nombre de classes
print("Nombre de classes : ", len(synthetique['Class'].value_counts()))
#Nombre d'individus par classe
print("Nombre d'indivifus par classe :", synthetique['Class'].value_counts())

#Séparation des données de tests et d'entrainement
train_data, test_data = train_test_split(synthetique, test_size=0.2, random_state=42)

print("Calcul de l'entropie du jeu de données : ", arbre_dec_bin.entropie(train_data, "Class"))
print("Calcul du gain de l'attribut A : ", arbre_dec_bin.gain(train_data, "Class", "Attr_A")[0])
print("Calcul du gain de l'attribut B : ", arbre_dec_bin.gain(train_data, "Class", "Attr_B")[0])
print("Calcul du meilleur attribut pour partitionner : ", arbre_dec_bin.meilleur_attribut_pour_split(train_data, "Class")[0])
print("Calcul du meilleur partitionnement : ", arbre_dec_bin.meilleur_attribut_pour_split(train_data, "Class")[2:])


y = test_data["Class"]
#Code utilisé pour créer les arbres et générer des prédictions sur les données de test
for i in range(3, 9):
    arbre = arbre_dec_bin.construction_arbre(train_data, list(train_data.columns[:-1]),"Class", 0, i)
    y_pred = arbre_dec_bin.predictions(test_data, arbre)
    print("Erreur arbre de profondeur ", arbre_dec_bin.profondeur_arbre(arbre), " : ", arbre_dec_bin.erreur_classification(y, y_pred))
    exactitude = arbre_dec_bin.comparer_predictions(y, y_pred)
    print("Exactitude profondeur de ", arbre_dec_bin.profondeur_arbre(arbre), " : ", exactitude)
    # Créer un DataFrame pandas à partir des prédictions
    #df_predictions = pd.DataFrame(y_pred)
    
    # Exporter le DataFrame au format CSV
    #df_predictions.to_csv(f"Predictions/y_pred_DT_{i}.csv", index=False, header=False)

X_train = train_data.drop("Class", axis=1).values
# Label
y_train = train_data["Class"].values
y_train = pd.get_dummies(y_train).values
# Standardisation des données d'entraînement comme vu au chap 5
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_normalized = (X_train - mean) / std

# Suppression de 'Class' pour obtenir X
X_test = test_data.drop("Class", axis=1).values
# Label
y_test = test_data["Class"].values
y_test = pd.get_dummies(y_test)
# Standardisation des données d'entraînement comme vu au chap 5
mean = np.mean(X_test, axis=0)
std = np.std(X_test, axis=0)
X_test_normalized = (X_test - mean) / std
# Séparation des données d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_train_normalized, y_train, test_size=0.15, random_state=5)


nn = NeuralNet(X_train, y_train, X_test_normalized, y_test, X_val, y_val, hidden_layer_sizes=(10, 8, 4), activation='tanh', learning_rate=0.01, epochs=200)
y= np.array([y_train[0]])
instance = np.array([X_train[0]]).T
prediction, err = nn.forward_propagation(instance, y.T) #Réalisation d'une passe avant sur le modèle
print("Sortie prédite:", prediction)
print("Sortie attendue:", y)
print("Erreur calculée :", err)
delta = nn.backward_pass(instance, y) #Réalisation d'une rétropropagation sur la même instance
print("Delta :", delta)
prediction, err = nn.forward_propagation(instance, y.T) #Réalisation d'une nouvelle passe avant pour constater que les sortie sont bien mise à jour
print("Sortie prédite:", prediction)
print("Sortie attendue:", y)
print("Erreur calculée :", err)


#Script de création des différents NeuralNet

#Modèles relu : 
#Couches : (10, 8, 6)
nn = NeuralNet(X_train, y_train, X_test_normalized, y_test, X_val, y_val, hidden_layer_sizes=(10, 8, 6), activation='relu', learning_rate=0.01, diminution_rate=0.3, epochs=1000)
nn.fit()
y_pred = nn.predict(X_test_normalized)

#Couches : (10, 8, 4)
nn = NeuralNet(X_train, y_train, X_test_normalized, y_test, X_val, y_val, hidden_layer_sizes=(10, 8, 4), activation='relu', learning_rate=0.01, diminution_rate=0.3, epochs=1000)
nn.fit()
y_pred = nn.predict(X_test_normalized)

#Couches : (6, 4)
nn = NeuralNet(X_train, y_train, X_test_normalized, y_test, X_val, y_val, hidden_layer_sizes=(6, 4), activation='relu', learning_rate=0.01,diminution_rate=0.4,  epochs=1000)
nn.fit()
y_pred = nn.predict(X_test_normalized)

#Modèles tanh : 
#Couches : (6, 4)
nn = NeuralNet(X_train, y_train, X_test_normalized, y_test, X_val, y_val, hidden_layer_sizes=(6, 4), activation='tanh', learning_rate=0.01, diminution_rate=0.3, epochs=1000)
nn.fit()
y_pred = nn.predict(X_test_normalized)

#Couches : (10, 8,  4)
nn = NeuralNet(X_train, y_train, X_test_normalized, y_test, X_val, y_val, hidden_layer_sizes=(10, 8, 4), activation='tanh', learning_rate=0.01, diminution_rate=0.4, epochs=1000)
nn.fit()
y_pred = nn.predict(X_test_normalized)

#Couches : (10, 8, 6)
nn = NeuralNet(X_train, y_train, X_test_normalized, y_test, X_val, y_val, hidden_layer_sizes=(10, 8, 6), activation='tanh', learning_rate=0.01, diminution_rate=0.9, epochs=1000)
nn.fit()
y_pred = nn.predict(X_test_normalized)
