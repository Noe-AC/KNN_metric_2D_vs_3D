2020-02-05

		ASPECTS LOCAUX ET GLOBAUX DE L'APPRENTISSAGE MACHINE KNN

######################################################################
######################################################################

		DESCRIPTION

L'apprentissage machine KNN (k plus proches voisins) est un algorithme de prise de décision basé sur la distance entre les points.

Cet algorithme est implémenté dans la librairie Python scikit-learn.
La distance y est la distance euclidienne.

En géométrie et en topologie, il existe plusieurs métriques différentes.
Dans un contexte d'apprentissage machine, le choix d'une métrique particulière a un impact sur la prise de décisions.

La surface de la Terre est une sphère S²⊂ℝ³.
Sur cet espace on peut considérer trois métriques :
• la métrique euclidienne sur l'espace (longitude ϕ, latitude θ),
• la métrique euclidienne sur l'espace ℝ³,
• la métrique "longueur d'arc de cercles" sur S².
Les deux dernières métriques sont physiquement plus réalistes et indépendantes d'un choix de coordonnées particulières sur la sphère S².
On s'attend donc à ce que les résultats d'apprentissages soient meilleurs pour ces deux métriques que pour la première.

Pour illustrer ce phénomène on peut considérer les deux tables suivantes :
	réf. [1] : pays et continent
	réf. [2] : villes, positions (ϕ,θ), pays
En prenant la jointure sur le pays on obtient une nouvelle table contenant :
	position (ϕ,θ) et continent
L'apprentissage machine sur cette dernière table nous permet de prédire le continent d'un point géographique situé en un (ϕ,θ) donné.
On peut alors considérer l'impact du choix de la métrique sur le score d'apprentissage.
C'est ce que j'ai fait en réf. [3].

Ici, sur GitHub, je joins les deux fichiers suivants :
• cities.sql
• main.py

Le fichier cities.sql sert à prendre les deux tables [1] et [2], les importer dans une base de donnée Postgres, les joindre sur le pays puis les exporter dans la table (position, continent).

Le fichier contient le code Python 3 pour l'apprentissage machine.

Sont aussi jointes quatre images produites par matplotlib :
• scatter_R2.png : continents des villes (>200K habitants) en coord. (ϕ,θ).
• scatter_R3.png : continents des villes (>200K habitants) en coord. (x,y,z).
• metrique_phi_theta.png : frontières décisionnelles pour la métrique (ϕ,θ).
• metrique_R3.png : frontières décisionnelles pour la métrique sur ℝ³.

######################################################################
######################################################################

		RÉFÉRENCES

• [1] : Chaitanya Gokhale, Kaggle, country to continent, https://www.kaggle.com/statchaitya/country-to-continent.
• [2] : Max Mind, Kaggle, world cities database, https://www.kaggle.com/max-mind/world-cities-database.
• [3] : https://dms.umontreal.ca/~aubincn/Documents/Beamer/2020-02-05_Beamer_BNC.pdf

######################################################################
######################################################################

Noé Aubin-Cadot, 2020.