#Q2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

nbDonnee = 100
X = 3*np.random.rand(nbDonnee,1) # Génération de 100 valeurs aléatoires représentant 100 exemples d'entrées.
y = 2 + 4*pow(X,2) + np.random.randn(nbDonnee,1) # Génération des données de sorties qui sont une fonction des entrées plus un bruit.

plt.plot(X,y, "b.") # On trace un point bleu pour chaque point des données
plt.axis([0,3,0,15]) # On fixe les bornes de l'abscisse et de l'ordonnée du graphique
plt.show() # On affiche le tout

X_b = np.c_[np.ones((nbDonnee,1)),X] # Ajout de la constante '1' à chaque donnée pour calculer le biais theta0 (rappelons que l'équation d'une regression linéaire est y = theta0 + theta1*x1 + theta2*x2 + ... Ici on a juste une valeur par exemple (donc x1) et on ajoute le 1 pour calculer le theta0 
#Quand on va faire le calcul du model, on va avoir la dimension X et la dimension X², c'est a partir des deux qu'il va essayer de calculer Y.
#Avant on avait ax + b, maintenant on aura ax + b + cx². Mais il ne decomposra que en droite. On reste sur des droites, mais on a un plan entre x et x².
#Dans la Q1, elle contient 1 et X. Maintenant elle contiendra 1, x, et x². A ca on va appliquer la recherche des coef theta, et on espere que le fait d'avoir introduit x² dans les entrées, on aura un Y qui sera autre chose qu'une droite.
#Si on met que pow(X,2), on va rajouter des infos redondantes, et voir si ca l'aide.
X_b = np.c_[X_b, pow(X,2)] # Q2: Ajout de ces valeurs au carré, concatenation de X_b qui contenant les 1 et les X, et on y ajoute les X au carré
#On doit avoir des paquets de 3 valeurs (1, x aléatoir, X²)
#print(X_b)

before = time.time() #permettra d'obtenir le temps de calcul de l'algorithme
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #  Numpy est doté de nombreux algorithmes d'algèbre linéaire. Ici, il permet de calculer l'inverse d'une matrice. (rappelons qu'il existe une solution analytique permettant de calculer le theta qui minimise la MSE et qui est [1/(Xt.X)].Xt.y
print("Theta : ", theta_best)
# Nous allons maintenant tracer la droite représentant le theta_best ainsi que les points utilisés pour l'apprendre

#Choix de deux valeur de X pour lequelles on calcul de y en utilisant theta x. On applique pour les deux y = ax + b et a partir des deux points, on fait la droite, puisqu'on a que deux points. Maintenant, comme les donnees ont ete crees
#a partir de X ET de X². Le y, a partir de X n'est plus une droite. On ne peut plus l'utiliser que de deux points. On doit maintenant faire une série de X (0, 0.2, 0.4, 0,6...) et on affiche des points et non plus des droites.
#La droite rouge est faite a partir des points, il faut donc faire un array non pas de [0][3] mais un array de range entre 0 et 3 avec un pas de 0.2.
X_new= np.array(np.arange(start=0, stop=3, step=0.2)) # Decoupage de X en segments de taille 0.2 entre 0 et 3.
X_new_b = np.c_[np.ones((15,1)),X_new] # On ajoute le '1' à chaque donnée pour les 15 valeurs
#il faut faire le meme traitement pour les x² que ce qu'on a fait pour les 1 pour en faire les nouvelles entrées.
#X c'est toutes nos valeurs donc 100 valeurs, X_new, c'est de quoi faire ax, et x_new_b, c'est X plus un vecteur de 1, dont on a besoin pour calculer theta. X_new_b, c'est les X auquelq on a ajouté le 1 pour ajouter le biais dans les droites
X_new_square = np.c_[pow(X_new,2),X_new_b]

y_predict = X_new_square.dot(theta_best) # on calcul le y à partir de theta_best
#Fait pour afficher une droite, il faut modifier car le X_new n'utilisait que 2 points, 0 et 3. Ca créait une droite avec 2 points. Maintenant, chaque y dépend de X et de X². Il faut génerer une liste de X_new. Faire un range de 0 à 3 avec un interval
# de 0.2, et calculer ces Y a partir de beaucoup plus de X_new. On trace donc un point a partir des points bleus au lieu d'une droite.
plt.plot(X_new, y_predict, "r-") # On trace une droite rouge entre les 2 points prédits
plt.plot(X,y, "b.") #On trace un point bleu pour chaque point des données utilisées pour apprendre theta
plt.axis([0,3,0,15]) # On fixe les bornes de l'abscisse et de l'ordonnée du graphique
plt.show() # On affiche la courbe









#Q3
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

nbDonnee = 5

#On veut que le model prédise Y a partir de X, pas x etc... Nous on veut donner un X, et avoir un Y. Comme on fait de la polynomialisation, on le fait a l'apprentissage et au test.
#Reprendre le code de calcul theta_best et replacer par ce qui sort de x_poly

# on va pousser jusque la puissance 3 et on ajoute la colonne de 1 pour le Theta0
poly_features = PolynomialFeatures(degree=3, include_bias=True) 
X = 3*np.random.rand(nbDonnee,1) # Generation de trois valeurs pouvant représenter 3 exemples d'entrées
X_poly=poly_features.fit_transform(X) # Cette fonction fait la polynomialisation pour nous ;-)
y = 2 + 4*pow(X_poly,2) + np.random.randn(nbDonnee,1) # Génération des données de sorties qui sont une fonction des entrées.

#Découpage en 2/3 apprentissage et 1/3 test
print("Nombre de données : ", nbDonnee)
print("Taille set d'apprentissage : ", round(2*nbDonnee/3) )
print("Taille set de test : ", round(nbDonnee/3) )
base_apprentissage = X_poly[round(2*nbDonnee/3):]
base_test = X_poly[:round(nbDonnee/3)]

print ("X : " , X)
print ("X_poly : ", X_poly)
print("Y : " , y)