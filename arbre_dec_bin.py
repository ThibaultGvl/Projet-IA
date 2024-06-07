# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:59:58 2024

@author: Thibault Grivel
@binome: Samuel Odoardi
"""

import numpy as np
import pandas as pd 

#Calcul de l'entropie
def entropie(dataframe, attribut_cible):
    nb_lignes = dataframe.shape[0]
    series = dataframe[attribut_cible].value_counts()
    res = 0
    for i in series:
        res = res + i/nb_lignes*np.log2(i/nb_lignes)
    return -res


#Calcul du gain d'information pour un attribut 'a' dans un dataframe.
def gain(dataframe, attribut_cible, a):
    # Initialisation des variables pour stocker le meilleur gain et la meilleure division
    meilleur_gain = 0
    split_value = 0
    partitions = []

    # Tri des données en fonction de l'attribut 'a'
    sorted_data = dataframe.sort_values(by=a)
    
    # Calcul de l'entropie initiale
    E = entropie(sorted_data, attribut_cible)
    
    # Obtention des quartiles pour l'attribut 'a'
    quartiles = sorted_data[a].quantile([0.25, 0.5, 0.75])

    for quartile in quartiles:
        # Création des partitions en utilisant le quartile actuel comme point de split
        
        partition_gauche = sorted_data[sorted_data[a] < quartile]
        partition_droite = sorted_data[sorted_data[a] >= quartile]
        
        # Calcul du gain d'information
        gain_actuel = E - ((len(partition_gauche) / len(sorted_data)) * entropie(partition_gauche, attribut_cible) +
                           (len(partition_droite) / len(sorted_data)) * entropie(partition_droite, attribut_cible))
        # Mise à jour du meilleur gain et des meilleures partitions
        if gain_actuel > meilleur_gain:
            meilleur_gain = gain_actuel
            split_value = quartile
            partitions = [partition_gauche, partition_droite]
    return meilleur_gain, split_value, partitions

#Trouve le meilleur attribut pour diviser le dataframe.
def meilleur_attribut_pour_split(dataframe, attribut_cible):
    attributs = [col for col in dataframe.columns if col != attribut_cible]
    meilleur_gain = 0
    meilleur_attribut = None
    meilleur_split_value = None
    meilleures_partitions = None
    
    for attribut in attributs:
        # Calcul du gain d'information pour cet attribut
        gain_attribut, split_value, partitions = gain(dataframe, attribut_cible, attribut)
        
        # Mise à jour du meilleur attribut si le gain est meilleur
        if gain_attribut > meilleur_gain:
            meilleur_gain = gain_attribut
            meilleur_attribut = attribut
            meilleur_split_value = split_value
            meilleures_partitions = partitions
    
    # Vérification si meilleures_partitions n'est pas nul avant de retourner ses éléments
    if meilleures_partitions is not None:
        partition_gauche = meilleures_partitions[0]
        partition_droite = meilleures_partitions[1]
    else:
        partition_gauche = None
        partition_droite = None
    
    return meilleur_attribut, meilleur_split_value, partition_gauche, partition_droite

class Noeud:
    def __init__(self, attribut=None, split=None, branche_gauche=None, branche_droite=None, prediction=None, feuille=False):
        self.attribut = attribut  # Attribut pour la division
        self.split = split  # Valeur de split
        self.branche_gauche = branche_gauche  # Sous-arbre pour les valeurs inférieures ou égales au split
        self.branche_droite = branche_droite  # Sous-arbre pour les valeurs supérieures au split
        self.prediction = prediction  # Prédiction pour les feuilles
        self.feuille = feuille  # Indique si c'est une feuille de l'arbre
        
    def node_result(self, spacing=''):
        s = ''
        for v in range(len(self.prediction.values)):
            s +=  'Class' + str(self.prediction.index[v]) + ' Count: ' + str(self.prediction.values[v]) + '\n' + spacing
            return s
 
def print_tree(node, spacing=''):
    if node is None:
        return
    if node.feuille:
        print(spacing + node.node_result(spacing))
        return
    print('{}[Attribute: {} Split value: {}]'.format(spacing, node.attribut, node.split))
    print(spacing + '> True')
    print_tree(node.branche_gauche, spacing + '-')
    print(spacing + '> False')
    print_tree(node.branche_droite, spacing + '-')
    return

def construction_arbre(data, attributs_restants, cible, profondeur, seuil):
    # Vérification du seuil de profondeur
    if profondeur >= seuil:
        prediction = data[cible].value_counts().idxmax()
        return Noeud(prediction=prediction, feuille=True)

    attribut, split, partitions_gauche, partitions_droite = meilleur_attribut_pour_split(data, cible)

    # Initialisation de la variable prediction
    prediction = None

    # Vérification des partitions gauche et droite
    if partitions_gauche is None or partitions_droite is None:
        # Assigner la valeur de prédiction appropriée
        prediction = data[cible].value_counts().idxmax()
        return Noeud(prediction=prediction, feuille=True)

    # Calcul de la prédiction pour le nœud en fonction des données spécifiques au nœud
    classes_du_noeud = partitions_gauche[cible].unique()
    occurrences_par_classe = partitions_gauche[cible].value_counts()
    prediction = classes_du_noeud[occurrences_par_classe.argmax()]

    # Copie des attributs restants pour les branches gauche et droite
    attributs_restants_gauche = attributs_restants.copy()
    attributs_restants_droite = attributs_restants.copy()
    if attribut in attributs_restants_gauche:
        attributs_restants_gauche.remove(attribut)
    if attribut in attributs_restants_droite:
        attributs_restants_droite.remove(attribut)

    # Construction des branches gauche et droite de manière récursive
    branche_gauche = construction_arbre(partitions_gauche, attributs_restants_gauche, cible, profondeur+1, seuil)
    branche_droite = construction_arbre(partitions_droite, attributs_restants_droite, cible, profondeur+1, seuil)

    # Retourner le nœud avec les branches gauche et droite
    return Noeud(attribut=attribut, split=split, branche_gauche=branche_gauche, branche_droite=branche_droite, prediction=prediction)

def inference(instance, noeud):
    if noeud.feuille:
        return noeud.prediction
    else:
        valeur_attribut = instance[noeud.attribut]
        if valeur_attribut < noeud.split:
            return inference(instance, noeud.branche_gauche)
        else:
            return inference(instance, noeud.branche_droite)
        
def predictions(test_data, tree):
    predictions = []  # Liste pour stocker les prédictions

    # Parcourir chaque instance dans le jeu de données de test
    for index, instance in test_data.iterrows():
        # Appeler la fonction inference avec l'instance et la racine de l'arbre
        prediction = inference(instance, tree)
        # Ajouter la prédiction à la liste des prédictions
        predictions.append(prediction)

    return predictions

def comparer_predictions(y_true, y_pred):
    # Vérifier si les entrées sont sous forme de liste ou de vecteur pandas
    if isinstance(y_true, pd.Series):
        y_true = y_true.tolist()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.tolist()

    # Vérifier si les longueurs des deux listes sont égales
    if len(y_true) != len(y_pred):
        raise ValueError("Les longueurs de y_true et y_pred ne correspondent pas.")

    # Calculer le nombre de prédictions correctes
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # Calculer l'exactitude
    exactitude = correct_predictions / len(y_true)

    return exactitude

def erreur_classification(y_true, y_pred):
    erreur = sum(y_true != y_pred) / len(y_true)
    return erreur

def profondeur_arbre(noeud):
    if not noeud:
        return -1
    else:
        profondeur_gauche = profondeur_arbre(noeud.branche_gauche)
        profondeur_droite = profondeur_arbre(noeud.branche_droite)

        # Profondeur maximale entre la branche gauche et droite, plus 1 pour le noeud actuel
        return max(profondeur_gauche, profondeur_droite) + 1