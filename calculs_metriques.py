# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 07:27:19 2024

@author: Thibault Grivel
@binome: Samuel Odoardi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def compute_metrics(y_pred, true_labels):
    for i in range(4):
        # Filtrer les prédictions pour la classe i
        y_pred_target = y_pred.copy()
        y_pred_target[y_pred_target != i] = -1
        y_pred_target[y_pred_target == i] = 1
        
        # Filtrer les vraies étiquettes pour la classe i
        true_labels_target = true_labels.copy()
        true_labels_target[true_labels_target != i] = -1
        true_labels_target[true_labels_target == i] = 1
        
        TP = np.sum((y_pred_target == 1) & (true_labels_target == 1))
        FP = np.sum((y_pred_target == 1) & (true_labels_target != 1))
        TN = np.sum((y_pred_target != 1) & (true_labels_target != 1))
        FN = np.sum((y_pred_target != 1) & (true_labels_target == 1))
        
        # Calcul de la précision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
        
        f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        print(f"Metriques for class {i} : Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1-score = {f1_score}")

def confusion_matrix(y_pred, true_labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = np.sum((y_pred == j) & (true_labels == i))
    return matrix


# Importation du fichier
synthetique = pd.read_csv("synthetic.csv")

# Séparation des données de test et d'entraînement
train_data, test_data = train_test_split(synthetique, test_size=0.2, random_state=42)
true_labels = test_data["Class"].values
predictions = []
nom_pred = []
tanh1086 = pd.read_csv("Predictions/y_pred_NN_tanh_10-8-6.csv", header=None)
predictions.append(tanh1086)
nom_pred.append("tanh10-8-6")
tanh1084 = pd.read_csv("Predictions/y_pred_NN_tanh_10-8-4.csv", header=None)
predictions.append(tanh1084)
nom_pred.append("tanh10-8-4")
tanh64 = pd.read_csv("Predictions/y_pred_NN_tanh_6-4.csv", header=None)
predictions.append(tanh64)
nom_pred.append("tanh6-4")
relu1086 = pd.read_csv("Predictions/y_pred_NN_relu_10-8-6.csv", header=None)
predictions.append(relu1086)
nom_pred.append("relu10-8-6")
relu1084 = pd.read_csv("Predictions/y_pred_NN_relu_10-8-4.csv", header=None)
predictions.append(relu1084)
nom_pred.append("relu10-8-4")
relu64 = pd.read_csv("Predictions/y_pred_NN_relu_6-4.csv", header=None)
predictions.append(relu64)
nom_pred.append("relu6-4")
DT3 = pd.read_csv("Predictions/y_pred_DT_3.csv", header=None)
predictions.append(DT3)
nom_pred.append("DT4")
DT4 = pd.read_csv("Predictions/y_pred_DT_4.csv", header=None)
predictions.append(DT4)
nom_pred.append("DT4")
DT5 = pd.read_csv("Predictions/y_pred_DT_5.csv", header=None)
predictions.append(DT5)
nom_pred.append("DT5")
DT6 = pd.read_csv("Predictions/y_pred_DT_6.csv", header=None)
predictions.append(DT6)
nom_pred.append("DT6")
DT7 = pd.read_csv("Predictions/y_pred_DT_7.csv", header=None)
predictions.append(DT7)
nom_pred.append("DT7")
DT8 = pd.read_csv("Predictions/y_pred_DT_8.csv", header=None)
predictions.append(DT8)
nom_pred.append("DT8")
for i in range(len(predictions)):
    print("Predictions pour ", nom_pred[i])
    indices_meilleure_classe = predictions[i]
    if i<6:
        indices_meilleure_classe = predictions[i].idxmax(axis=1)
    y_pred = indices_meilleure_classe.values.flatten() 
    compute_metrics(y_pred, true_labels)
    print(confusion_matrix(y_pred, true_labels, 4))