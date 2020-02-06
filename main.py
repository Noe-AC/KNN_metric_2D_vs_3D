"""
2020-02-03

Entraînement supervisé
Target y discret

Le but ici est de faire un modèle KNN pour prédire le continent d'un point.
"""

# On importe les données dans un array numpy
import pandas as pd
import numpy as np
Xy = (pd.read_csv("cities.csv",sep = ',',skipinitialspace=False)).to_numpy()
# On découpe les données en features et target
X = Xy[:,3:] # features : position = (x,y,z) cartésien
#X = Xy[:,1:3] # features : position = (phi,theta) sphérique
y = Xy[:,0]  # le target : continent = {0,1,2,3,4}

# On découpe les données en train + test
from sklearn.model_selection import train_test_split
test_size=0.25
random_state = 27
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
print(X.shape,X_train.shape,X_test.shape,y.shape,y_train.shape,y_test.shape) # donne : (1661, 5) (1245, 5) (416, 5) (1661,) (1245,) (416,)

# Paramètres d'entraînement
n_neighbors = 1
kernel = 'rbf'
SVC_gamma = 0.001
#SVC_C = 4.9
SVC_C = 100.

# On met les erreurs à OFF
import warnings
warnings.simplefilter("ignore")


#####################################
############### KNN #################

# Apprentissage : KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("KNN :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

########################################
######## Bayésien naïf gaussien ########

# Apprentissage : bayésien naïf gaussien
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("BNG :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

########################################
######## Bayésien naïf Bernoulli ########

# Apprentissage : bayésien naïf Bernoulli
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("BNB :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

########################################
######## Bayésien naïf Multinomial ########
"""
# Apprentissage : bayésien naïf multinomial
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("BNM :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

J'ai enlevé BNM car ça donnait une erreur :

Traceback (most recent call last):
  File "main.py", line 73, in <module>
    clf.fit(X_train,y_train)
  File "/usr/local/lib/python3.7/site-packages/sklearn/naive_bayes.py", line 613, in fit
    self._count(X, Y)
  File "/usr/local/lib/python3.7/site-packages/sklearn/naive_bayes.py", line 720, in _count
    raise ValueError("Input X must be non-negative")
ValueError: Input X must be non-negative

"""
########################################
################## SVM #################

# Apprentissage : SVM
from sklearn import svm
clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C)
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("SVM :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##############################################
####### Régression logistique lbfgs ##########

# Apprentissage : régression logistique avec solveur 'lbfgs'
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=100) # il faut mettre 4000 pour que ça converge, mais c'est long à calculer
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("lbf :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
####### Régression logistique liblinear ##########

# Apprentissage : régression logistique avec solveur 'liblinear'
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear',multi_class='auto',max_iter=100)
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("lib :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
########### Random Forest Classifier #############

# Apprentissage : Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("RFC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
################# Perceptron #####################

# Apprentissage : Perceptron
from sklearn.linear_model import Perceptron
clf = Perceptron()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("Per :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
################ SGDClassifier (SGDC) ###################

# Apprentissage : SGDClassifier, descente de gradient stochastique, version classificateur
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("SGD :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
####### Decision Tree Classifier (DTC) ###########

# Apprentissage : DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("DTC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))


####################################################################################################
####################################################################################################

"""

Prédictions en coordonnées cartésiennes (X,Y,Z) :

KNN :	 train : 100.0%	 test : 99.0%
BNG :	 train : 96.2%	 test : 96.4%
BNB :	 train : 74.4%	 test : 71.2%
SVM :	 train : 95.7%	 test : 95.0%
lbf :	 train : 96.5%	 test : 95.7%
lib :	 train : 94.1%	 test : 95.0%
RFC :	 train : 100.0%	 test : 98.8%
Per :	 train : 90.9%	 test : 88.5%
SGD :	 train : 96.7%	 test : 94.7%
DTC :	 train : 100.0%	 test : 97.4%

Prédictions selon (phi,theta), avec mauvaise métrique "cartésienne" sur le plan (phi,theta) :

KNN :	 train : 100.0%	 test : 98.8%
BNG :	 train : 97.2%	 test : 96.4%
BNB :	 train : 71.7%	 test : 67.8%
SVM :	 train : 95.3%	 test : 94.7%
lbf :	 train : 95.7%	 test : 95.2%
lib :	 train : 92.5%	 test : 93.0%
RFC :	 train : 100.0%	 test : 98.3%
Per :	 train : 80.1%	 test : 79.8%
SGD :	 train : 94.5%	 test : 92.1%
DTC :	 train : 100.0%	 test : 97.1%

"""


####################################################################################################
####################################################################################################

# Ici je vais faire une matrice de confusion pour KNN, k=1.
# Colonne est la réalité, ligne est la prédiction

# Apprentissage : KNN
#from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train,y_train)
y_validation = clf.predict(X_test)
#print("KNN :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

from sklearn.metrics import confusion_matrix
y_true = y_test
y_pred = y_validation
matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
print(matrix)

print("\nNombre de prédictions : ",len(y_true))

"""
Donne :

[[101   0   0   0   0]
 [  0  80   0   0   0]
 [  0   1  47   0   0]
 [  0   3   0 181   0]
 [  0   0   0   0   3]]

0 : Amérique
1 : Europe
2 : Afrique
3 : Asie
4 : Océanie

En général c'est bon.
Il y a 4 erreurs :
1 ville d'Afrique est prédite être en Europe
3 villes d'Asie sont prédites être en Europe

"""

####################################################################################################
####################################################################################################

# Ici je trouve quelles villes sont mal prédites

# Il me faut les coordonnées longitude, latitude en degrés :

import math
for i in range(len(y_true)):
	if y_true[i]!=y_pred[i]:
		for j in range(len(X)):
			if Xy[j,3]==X_test[i,0]:
				#print("yes!")
				print("i : ",i, "\ty_true : ",y_true[i],"\ty_pred : ",y_pred[i], "\t(longitude,latitude) : ",180*Xy[j,1]/math.pi,180*Xy[j,2]/math.pi)

"""
i :  78 	y_true :  3.0 	y_pred :  1.0 	(longitude,latitude) :  57.181389 50.29805600000001
i :  139 	y_true :  2.0 	y_pred :  1.0 	(longitude,latitude) :  -0.6416667 35.69111109999999
i :  219 	y_true :  3.0 	y_pred :  1.0 	(longitude,latitude) :  63.58333299999978 53.16666699999998
i :  260 	y_true :  3.0 	y_pred :  1.0 	(longitude,latitude) :  76.95000000000019 52.29999999999998

Pour avoir la ville en SQL :

57.181389 50.29805600000001
SELECT * FROM cities_full WHERE longitude>57.1813 AND longitude<57.1814 AND latitude>50.29805 AND latitude<50.299; 

 kz      | aqtobe        | Aqtöbe        | 04     |   262471.0 | 50.298056 | 57.181389


-0.6416667 35.69111109999999
SELECT * FROM cities_full WHERE longitude>-0.642 AND longitude<-0.64 AND latitude>35.691 AND latitude<35.69112; 

 dz      | wahran  | Wahran     | 09     |   646025.0 | 35.6911111 | -0.6416666999999999

63.58333299999978 53.16666699999998
SELECT * FROM cities_full WHERE longitude>63.58333 AND longitude<63.58334 AND latitude>53.16666 AND latitude<53.1667; 

 kz      | qostanay   | Qostanay   | 13     |   230282.0 | 53.166667000000004 | 63.583332999999996


76.95000000000019 52.29999999999998
SELECT * FROM cities_full WHERE longitude>76.94 AND longitude<76.96 AND latitude>52.2 AND latitude<52.4; 

 kz      | pavlodar | Pavlodar   | 11     |   329014.0 |     52.3 |     76.95


SELECT * FROM cities_full WHERE longitude>... AND longitude<... AND latitude>... AND latitude<...; 


Les villes :
i=78 : Aqtöbe (kz), (Kazakhstan)
i=139 : Wahran (dz) (Algérie)
i=219 : Qostanay (kz), (Kazakhstan)
i=260 : Pavlodar (kz), (Kazakhstan)

Bref :
La ville africaine prédite en Europe : Wahran (Oran) en Algérie
Les 3 villes asiatiques prédites en Europe : Aqtöbe, Qostanay et Pavlodar au Kazakhstan
"""


####################################################################################################
####################################################################################################

# On peut faire des plot et scatter de points

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(Xy)):
	color=''
	label=''
	continent = Xy[i][0]
	if continent==0: # Amérique
		color='#a70000'
	if continent==1: # Europe
		color='#005ce6'
	if continent==2: # Afrique
		color='#86592d'
	if continent==3: # Asie
		color='#00cc00'
	if continent==4: # Océanie
		color='#ff8000'
	ax.scatter(X[i][0], X[i][1], X[i][2],c=color,label=label)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.legend() # non ça fait plein de lignes
plt.show()
"""



"""
# Isomap : Isometric mapping
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=5, n_components=2)
proj = iso.fit_transform(X)
plt.scatter(proj[:,0],proj[:,1],c=y)
plt.colorbar()
plt.show()
"""


"""
# De retour à KNN :
# Pour afficher les frontières de décisions
# Ici c'est du ML avec coord cartésiennes (x,y,z)
# https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X,y)
h = .01  # step size in the mesh
# Create color maps
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_bold  = ListedColormap(['#a70000', '#005ce6', '#86592d','#00cc00','#ff8000'])
cmap_light = ListedColormap(['#ff0000', '#80b3ff', '#d9b38c','#99ff99','#ffbf80'])
# On plot la frontière de décision
import math
#phi_min,   phi_max   = min(Xy[:,1]), max(Xy[:,1])
#theta_min, theta_max = min(Xy[:,2]), max(Xy[:,2])
phi_min,   phi_max   = -math.pi, math.pi
theta_min, theta_max = -math.pi/2, math.pi/2
range_phi   = list(np.arange(phi_min,   phi_max,   h))
range_theta = list(np.arange(theta_min, theta_max, h))
X_grid = []
for j in range(len(range_theta)):
	for i in range(len(range_phi)):
		phi   = range_phi[i]
		theta = range_theta[j]
		x = math.cos(theta)*math.cos(phi)
		y = math.cos(theta)*math.sin(phi)
		z = math.sin(theta)
		X_grid.append([x,y,z])
phiphi, thetatheta = np.meshgrid( range_phi , range_theta )
Z = clf.predict(X_grid)
# On affiche la grille prédite
Z = Z.reshape(thetatheta.shape)
print(phiphi.shape,thetatheta.shape) # (20,59)
plt.figure()
plt.pcolormesh(phiphi,thetatheta, Z, cmap=cmap_light)
# Plot also the training points
plt.scatter(Xy[:,1], Xy[:,2], c=Xy[:,0], cmap=cmap_bold)
plt.xlim(phiphi.min(),     phiphi.max())
plt.ylim(thetatheta.min(), thetatheta.max())
plt.xlabel("Longitude phi (radians)")
plt.ylabel("Latitude theta (radians)")
plt.show()
"""


"""
# De retour à KNN :
# Pour afficher les frontières de décitions
# Ici c'est du ML avec coord sphériques (phi,theta)
# https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X,y)
h = .01  # step size in the mesh
# Create color maps
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_bold  = ListedColormap(['#a70000', '#005ce6', '#86592d','#00cc00','#ff8000'])
cmap_light = ListedColormap(['#ff0000', '#80b3ff', '#d9b38c','#99ff99','#ffbf80'])
# On plot la frontière de décision
import math
#phi_min,   phi_max   = min(Xy[:,1]), max(Xy[:,1])
#theta_min, theta_max = min(Xy[:,2]), max(Xy[:,2])
phi_min,   phi_max   = -math.pi, math.pi
theta_min, theta_max = -math.pi/2, math.pi/2
range_phi   = list(np.arange(phi_min,   phi_max,   h))
range_theta = list(np.arange(theta_min, theta_max, h))
X_grid = []
for j in range(len(range_theta)):
	for i in range(len(range_phi)):
		phi   = range_phi[i]
		theta = range_theta[j]
		X_grid.append([phi,theta])
phiphi, thetatheta = np.meshgrid( range_phi , range_theta )
Z = clf.predict(X_grid)
# On affiche la grille prédite
Z = Z.reshape(thetatheta.shape)
print(phiphi.shape,thetatheta.shape) # (20,59)
plt.figure()
plt.pcolormesh(phiphi,thetatheta, Z, cmap=cmap_light)
# Plot also the training points
plt.scatter(Xy[:,1], Xy[:,2], c=Xy[:,0], cmap=cmap_bold)
plt.xlim(phiphi.min(),     phiphi.max())
plt.ylim(thetatheta.min(), thetatheta.max())
plt.xlabel("Longitude phi (radians)")
plt.ylabel("Latitude theta (radians)")
plt.show()
"""


"""
# Plot also the training points
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_bold  = ListedColormap(['#a70000', '#005ce6', '#86592d','#00cc00','#ff8000'])
plt.scatter(Xy[:,1], Xy[:,2], c=Xy[:,0], cmap=cmap_bold)
#phi_min,   phi_max   = min(Xy[:,1]), max(Xy[:,1])
#theta_min, theta_max = min(Xy[:,2]), max(Xy[:,2])
phi_min,   phi_max   = -math.pi, math.pi
theta_min, theta_max = -math.pi/2, math.pi/2
plt.xlim(phi_min,phi_max)
plt.ylim(theta_min,theta_max)
plt.xlabel("Longitude phi (radians)")
plt.ylabel("Latitude theta (radians)")
plt.show()
"""

