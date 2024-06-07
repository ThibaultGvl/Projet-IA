# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:08:34 2024

@author: Thibault Grivel
@binome: Samuel Odoardi
"""

import numpy as np
import matplotlib.pyplot as plt
from utility import Utility
from sklearn.utils import shuffle


class NeuralNet:
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, val_X=None, val_y=None,
                 hidden_layer_sizes=(6, 4), activation='relu', learning_rate=0.01, diminution_rate = 0.4, epochs=50):
        # Initialisation des données d'entraînement et de test ainsi que des hyperparamètres
        self.X_train = X_train  # Données d'entraînement
        self.y_train = y_train  # Étiquettes d'entraînement
        self.X_test = X_test  # Données de test
        self.y_test = y_test  # Étiquettes de test
        self.val_X = val_X  # Données de validation
        self.val_y = val_y  # Étiquettes de validation
        self.hidden_layer_sizes = hidden_layer_sizes  # Tailles des couches cachées
        self.activation = activation  # Fonction d'activation
        self.learning_rate = learning_rate  # Taux d'apprentissage
        self.diminution_rate = diminution_rate
        self.epochs = epochs  # Nombre d'époques
        self.n_layers = len(hidden_layer_sizes) + 1  # Nombre total de couches (couches cachées + couche de sortie)

        # Initialisation des matrices de poids et de biais
        self.weights = [None] * self.n_layers  # Initialisation des poids
        self.biases = [None] * self.n_layers  # Initialisation des biais
        self.weights_initialization(X_train.shape[1], y_train.shape[1])  # Initialisation des poids
        self.activations = [None] * (self.n_layers + 1)  # Initialisation des activations
        # Initialisation des dérivées des fonctions d'activation
        self.df = [None] * (self.n_layers + 1)  # Initialisation des dérivées des fonctions d'activation

    #Fonction qui fonctionne sans soucis
    def weights_initialization(self, n_attributes, n_classes):
        # Initialisation des poids pour les couches cachées
        for i, layer_size in enumerate(self.hidden_layer_sizes):
            if i == 0:
                # Pour la première couche cachée
                self.weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(layer_size, n_attributes))
            else:
                # Pour les autres couches cachées
                self.weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(layer_size, self.hidden_layer_sizes[i-1]))
            self.biases[i] = np.zeros((layer_size, 1))  # Initialisation des biais
    
        # Initialisation des poids pour la couche de sortie
        self.weights[-1] = np.random.uniform(low=-1.0, high=1.0, size=(n_classes, self.hidden_layer_sizes[-1]))
        self.biases[-1] = np.zeros((n_classes, 1))

    def forward_propagation(self, X, y = None):
        # Liste pour stocker les activations et les entrées pondérées
        activations = [None] * self.n_layers  # Initialisation des activations
        weights = [None] * self.n_layers #
        error = 0
        # Propagation avant pour les couches cachées
        A_prev = X
        for l in range(self.n_layers - 1):
            Z = np.dot(self.weights[l], A_prev) + self.biases[l]  # Calcul des entrées pondérées
            A, df = Utility.tanh(Z) if self.activation == 'tanh' else Utility.relu(Z)  # Application de la fonction d'activation
            activations[l] = A  # Stockage des activations
            weights[l] = Z  # Stockage des entrées pondérées
            A_prev = A  # Mise à jour de l'activation précédente
            self.df[l] = df  # Stockage de la dérivée dans la liste des dérivées
        # Propagation avant pour la couche de sortie avec softmax
        Z_output = np.dot(self.weights[-1], A_prev) + self.biases[-1]  # Calcul de l'entrée pondérée de la couche de sortie
        A_output = Utility.softmax(Z_output)  # Application de la fonction softmax
        activations[-1] = A_output  # Stockage de l'activation de la couche de sortie
        weights[-1] = Z_output
        if y is not None:
            # Calcul de l'erreur avec la fonction cross entropy cost
            error = Utility.cross_entropy_cost(A_output, y)  # Calcul de l'erreur
        self.activations = activations  # Mise à jour des activations
        return A_output, error  # Retourne l'activation de la couche de sortie et l'erreur

    def backward_pass(self, X, y):
        # Initialisation des listes pour stocker les erreurs et les ajustements
        delta = [None] * self.n_layers  # Initialisation de delta
        dW = [None] * self.n_layers  # Initialisation de dW
        db = [None] * self.n_layers  # Initialisation de db
    
        # Calcul de l'erreur pour la couche de sortie
        delta[-1] = self.activations[-1] - y.T  # Calcul de l'erreur
        dW[-1] = np.dot(delta[-1], self.activations[-2].T)  # Calcul du gradient des poids
        db[-1] = delta[-1]  # Calcul du gradient des biais
        # Calcul de l'erreur pour les couches cachées
        for l in range(self.n_layers - 2, -1, -1):
            # Hadamard product: delta[l] = (W[l+1].T * delta[l+1]) * df[l]
            delta[l] = np.dot(self.weights[l + 1].T, delta[l + 1]) * self.df[l]
            if l == 0:
                dW[l] = np.dot(delta[l], X.T)  # Calcul du gradient des poids pour la première couche cachée
            else:
                dW[l] = np.dot(delta[l], self.activations[l-1].T)  # Calcul du gradient des poids
            db[l] = delta[l]  # Calcul du gradient des biais
        # Mise à jour des paramètres
        for l in range(0, self.n_layers):
            self.weights[l] -= self.learning_rate * dW[l]  # Mise à jour des poids
            self.biases[l] -= self.learning_rate * db[l]  # Mise à jour des biais
    
        return delta
    
    def fit(self):
        best_val_error = float('inf')  # Initialisation de la meilleure erreur de validation
        patience = 4  # Patience pour l'early stopping
        n_no_improve = 0  # Nombre d'itérations sans amélioration
    
        train_errors = [] # Histotique des erreurs d'entrainement
        val_errors = [] # Histotique des erreurs de validation
    
        old_val_error = float('inf')  # Initialisation de l'ancienne erreur de validation
    
        # Entraînement sur plusieurs époques
        for epoch in range(self.epochs):
    
            self.adjust_learning_rate(epoch) # Màj du learning rate
            
            train_error, val_error = self.epoch()  # Entraînement sur une époque
            
            # Évaluation sur les données de validation
            val_errors.append(val_error)
    
            # Mise à jour de la meilleure erreur de validation et compteur d'itérations sans amélioration
            if val_error < best_val_error:
                best_val_error = val_error
                n_no_improve = 0
            else:
                n_no_improve += 1
    
            # Vérification de l'early stopping
            if n_no_improve >= patience:
                print("Early stopping: Pas d'amélioration après {} iterations".format(patience))
                break
            
            # Enregistrement de l'erreur d'entraînement
            train_errors.append(train_error)
    
            # Affichage de l'évolution des erreurs
            print(f"Epoque {epoch+1}/{self.epochs} - Erreur d'entrainement': {train_error}, Erreur de Validation: {val_error}")
    
        # Affichage des courbes d'évolution des erreurs
        plt.figure()
        title = f"{self.activation.capitalize()} {self.hidden_layer_sizes}"
        plt.plot(train_errors, label='Erreur entrainement')
        plt.plot(val_errors, label='Erreur validation')
        plt.xlabel('Epoque')
        plt.ylabel('Erreur')
        plt.legend()
        plt.title(title)
        plt.show()


    def epoch(self):
        train_errors = []  # Liste pour stocker les erreurs d'entraînement
        val_errors = []    # Liste pour stocker les erreurs de validation
        
        # Mélange des données d'entraînement
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        
        # Entraînement par mini-batch de 4
        batch_size = 4
        # Nombre de sous groupe pour voir toutes les données en 1 époque 
        num_batches = len(self.X_train) // batch_size
        
        # Calcul de l'erreur d'entraînement pour chaque mini-batch
        for batch_index in range(num_batches):
            batch_X = self.X_train[batch_index * batch_size : (batch_index + 1) * batch_size]
            batch_y = self.y_train[batch_index * batch_size : (batch_index + 1) * batch_size]
    
            batch_errors = []  # Liste pour stocker les erreurs du mini-batch
            # Calcul de l'erreur pour chaque instance du mini-batch
            for i in range(len(batch_X)):
                instance = np.array([batch_X[i]]).T
                y= np.array([batch_y[i]]).T
                y_pred, err = self.forward_propagation(instance, y)
                self.backward_pass(instance, np.array([batch_y[i]]))
                # Stockage de l'erreur de cette instance
                batch_errors.append(err)
    
            # Stockage de l'erreur moyenne du mini-batch
            train_errors.append(np.mean(batch_errors))
    
        # Calcul de l'erreur moyenne d'entraînement sur tous les mini-batchs
        train_error = np.mean(train_errors)
        
        # On mélange les données de validation au cas où
        self.val_X, self.val_y = shuffle(self.val_X, self.val_y)
        
         # Calcul de l'erreur de validation pour chaque instance
        for j in range(len(self.val_X)):
            instance_val = np.array([self.val_X[j]]).T
            y_val = np.array([self.val_y[j]]).T
            y_pred_val, err = self.forward_propagation(instance_val, y_val)
            val_errors.append(err)
    
        # Calcul de l'erreur moyenne de validation
        val_error = np.mean(val_errors)
    
        return train_error, val_error
    
    # Fonction de màj du learning rate, la coef de diminution_rate est défini itérativement après observation des courbes
    def adjust_learning_rate(self, epoch):
        if epoch%10 == 0:
            self.learning_rate *= self.diminution_rate

    # Fonction de prédiction d'instances
    def predict(self, X):
        predictions = []
        predicted_class = []
        
        for i in range(len(X)):
            instance = np.array([X[i]]).T # Transposer l'instance
            prediction, _ = self.forward_propagation(instance)  # Appel de la propagation avant
            predicted_class.append(np.argmax(prediction)) 
            predictions.append(prediction)
        return np.array(predictions), np.array(predicted_class)