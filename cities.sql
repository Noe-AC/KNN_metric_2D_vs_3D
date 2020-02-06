/*
2020-02-03

But :
Créer une table contenant (position,continent)

Plan :
1. importer le fichier "worldcitiespop.csv" dans une table Postgres
2. importer le fichier "countryContinent.csv" dans une table Postgres
3. joindre les tables, choisir les lignes de population>200K, mettre coord. cartésiennes 3D.
4. exporter ça dans un fichier cities.csv
*/


-- On efface la table cities_full (optionnel) :
DROP TABLE cities_full;

-- On crée une cities_full
CREATE TABLE cities_full
(
--  Id           SERIAL PRIMARY KEY,
  Country      CHAR(2), -- CHAR(2)
  City         TEXT,
  AccentCity   TEXT,
  Region       CHAR(2), -- pas INT car peut être "BD" pour "boundary"...
  Population   DECIMAL, -- DECIMAL
  Latitude     DECIMAL, -- DECIMAL
  Longitude    DECIMAL -- DECIMAL
);

-- On importe le fichier worldcitiespop.csv dans Postgres :
\copy cities_full FROM '//Users/NAC/Desktop/cities/world-cities-database/worldcitiespop.csv' DELIMITER ',' CSV HEADER;

-- Pour exporter les colonnes (region, latitude, longitude) de la table "cities_full" dans le fichier "cities.csv" :
-- \copy (SELECT region,latitude,longitude FROM cities_full) to '//Users/NAC/Desktop/cities/Python/cities.csv' CSV HEADER;
-- ok, mais je ne veux pas juste ça.

------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- Pour sélectionner le nombre de régions distinctes :
SELECT COUNT(DISTINCT region) FROM cities_full; -- 396
-- Pour sélectionner le nombre de lignes :
SELECT COUNT(region) FROM cities_full; -- 3173950, approx. 3 millions de lignes
-- Pour avoir le nombre de villes de plus d'un million d'habitants :
SELECT COUNT(region) FROM cities_full WHERE population > 1000000; -- 279
-- Pour avoir le nombre de villes de plus de 300,000 habitants :
SELECT COUNT(region) FROM cities_full WHERE population > 300000; -- 1098
-- Pour avoir le nombre de villes de plus de 200,000 habitants :
SELECT COUNT(region) FROM cities_full WHERE population > 200000; -- 1661
-- Pour avoir le nombre de villes de plus de 100,000 habitants :
SELECT COUNT(region) FROM cities_full WHERE population > 100000; -- 3527
/*
Avec 1661 lignes ça semble assez pour faire du ML.
Je vais prendre ça.
*/

-- Pour avoir le nombre de villes de plus de 100,000 habitants au Canada :
SELECT COUNT(region) FROM cities_full WHERE (population>100000 AND country LIKE 'ca'); -- 26 au Canada
-- Les villes canadiennes de plus de 100,000 habitants sont :
SELECT city,region,population FROM cities_full WHERE (population>100000 AND country LIKE 'ca') ORDER BY population DESC; -- 26 au Canada
/*
      city      | region | population 
----------------+--------+------------
 toronto        | 08     |  4612187.0
 montreal       | 10     |  3268513.0
 vancouver      | 02     |  1837970.0
 calgary        | 01     |   968475.0
 ottawa         | 08     |   874433.0
 edmonton       | 01     |   822319.0
 hamilton       | 08     |   653637.0
 quebec         | 10     |   645623.0
 winnipeg       | 03     |   632069.0
 kitchener      | 08     |   409111.0
 london         | 08     |   346774.0
 victoria       | 02     |   289625.0
 windsor        | 08     |   278013.0
 halifax        | 07     |   266012.0
 oshawa         | 08     |   247989.0
 saskatoon      | 11     |   198957.0
 barrie         | 08     |   182070.0
 regina         | 11     |   176182.0
 abbotsford     | 02     |   151685.0
 sherbrooke     | 10     |   129447.0
 kelowna        | 02     |   125110.0
 trois-rivieres | 10     |   119693.0
 guelph         | 08     |   115763.0
 kingston       | 08     |   114243.0
 waterloo       | 08     |   110800.0
 sudbury        | 08     |   109724.0
*/
/*
Ah ok !
La région c'est la province / état, etc.
*/
-- Pour avoir les villes du Québec :
SELECT city,region,population FROM cities_full WHERE (country LIKE 'ca' AND region LIKE '10' AND population>1) ORDER BY population DESC; 
SELECT count(city) FROM cities_full WHERE (country LIKE 'ca' AND region LIKE '10' AND population>1); -- 165 villes québécoises dont on sait la population
SELECT count(city) FROM cities_full WHERE (country LIKE 'ca' AND region LIKE '10'); -- 1049 villes québécoises
/*
Ok, assez de fun.
Maintenant je dois avoir les continents.
*/

------------------------------------------------------------------------------
------------------------------------------------------------------------------

/*
Maintenant j'importe le fichier CSV "countryContinent.csv" pour relier pays et continent.
*/
-- On crée une coutry_continent
CREATE TABLE country_continent
(
--  Id           SERIAL PRIMARY KEY,
  country          TEXT,
  code_2           TEXT,
  code_3           TEXT,
  country_code     INT,
  iso_3166_2       TEXT,
  continent        TEXT,
  sub_region       TEXT,
  region_code      INT,
  sub_region_code  TEXT
);

-- On importe le fichier worldcitiespop.csv dans Postgres :
\copy country_continent FROM '//Users/NAC/Desktop/cities/countryContinent/countryContinent.csv' DELIMITER ',' CSV HEADER ENCODING 'WIN1252';

-- Pour voir la table :
SELECT * FROM country_continent;
-- Pour voir les pays dont le continent n'est pas nul :
SELECT * FROM country_continent WHERE continent NOT LIKE ''; 
-- Pour voir le nombre de continents :
SELECT COUNT(DISTINCT(continent)) FROM country_continent WHERE continent NOT LIKE ''; -- 5
-- Pour voir les continents :
SELECT DISTINCT(continent) FROM country_continent WHERE continent NOT LIKE ''; 
/*
 Europe
 Oceania
 Americas
 Africa
 Asia
*/
-- Pour voir juste le pays, le code_2 et le continent :
SELECT country, code_2, continent FROM country_continent WHERE continent NOT LIKE '';
/*
Là le problème c'est que dans une table le pays est en 2 lettres minuscules et dans l'autre c'est en 2 lettres majuscules
*/
SELECT COUNT(DISTINCT(continent)) FROM country_continent WHERE continent NOT LIKE ''; 
/*
Pour changer le code en minuscules je peux faire ceci :
*/
SELECT country, LOWER(code_2), continent FROM country_continent WHERE continent NOT LIKE '';
/*
Ok ça marche.
Maintenant je dois faire un inner join pour fusionner les deux tables en une seule table.
*/

------------------------------------------------------------------------------
------------------------------------------------------------------------------

/*
Là on met les deux tables ensemble avec un INNER JOIN sur le country code.
*/
-- Les deux tables qui m'intéressent sont :
SELECT country, LOWER(code_2), continent FROM country_continent WHERE continent NOT LIKE '';
SELECT city,accentcity,country,population,latitude,longitude FROM cities_full WHERE population > 200000; -- 1661

-- Ok ce qui m'intéresse c'est ceci :
SELECT
	t1.country AS country_name,
	t1.continent AS continent,
	t2.city AS city,
	t2.accentcity AS accentcity,
	t2.country AS country_code,
	t2.population AS population,
	t2.latitude AS latitude,
	t2.longitude AS longitude
FROM (
SELECT
	country AS country,
	continent AS continent,
	LOWER(code_2) AS code_2
FROM country_continent
WHERE continent NOT LIKE '') AS t1
INNER JOIN (
SELECT
	*
FROM cities_full
WHERE population>200000) AS t2
ON t1.code_2=t2.country
ORDER BY population DESC
LIMIT 15; -- enlever le LIMIT 15 et le ORDER BY quand j'aurai ce que je veux 
/*
       country_name       | continent |   city    | accentcity | country_code | population |      latitude       |      longitude      
--------------------------+-----------+-----------+------------+--------------+------------+---------------------+---------------------
 Japan                    | Asia      | tokyo     | Tokyo      | jp           | 31480498.0 |              35.685 |          139.751389
 China                    | Asia      | shanghai  | Shanghai   | cn           | 14608512.0 |           31.045556 |          121.399722
 India                    | Asia      | bombay    | Bombay     | in           | 12692717.0 |              18.975 |   72.82583299999999
 Pakistan                 | Asia      | karachi   | Karachi    | pk           | 11627378.0 |             24.9056 |             67.0822
 India                    | Asia      | delhi     | Delhi      | in           | 10928270.0 |  28.666666999999997 |           77.216667
 India                    | Asia      | new delhi | New Delhi  | in           | 10928270.0 |                28.6 |                77.2
 Philippines              | Asia      | manila    | Manila     | ph           | 10443877.0 |             14.6042 |            120.9822
 Russian Federation       | Europe    | moscow    | Moscow     | ru           | 10381288.0 |  55.752221999999996 |           37.615556
 Korea (Republic of)      | Asia      | seoul     | Seoul      | kr           | 10323448.0 |             37.5985 |            126.9783
 Brazil                   | Americas  | sao paulo | São Paulo  | br           | 10021437.0 | -23.473292999999998 | -46.665803000000004
 Turkey                   | Asia      | istanbul  | Istanbul   | tr           |  9797536.0 |           41.018611 |           28.964722
 Nigeria                  | Africa    | lagos     | Lagos      | ng           |  8789133.0 |            6.453056 |            3.395833
 Mexico                   | Americas  | mexico    | Mexico     | mx           |  8720916.0 |  19.434167000000002 |          -99.138611
 Indonesia                | Asia      | jakarta   | Jakarta    | id           |  8540306.0 |           -6.174444 |          106.829444
 United States of America | Americas  | new york  | New York   | us           |  8107916.0 |          40.7141667 |         -74.0063889
*/
/*
Ok.
Maintenant ce qui m'intéresse c'est d'avoir juste le continent et latitude et longitude
*/
SELECT
	t1.continent AS continent,
	t2.latitude AS latitude,
	t2.longitude AS longitude
FROM (
SELECT
	country AS country,
	continent AS continent,
	LOWER(code_2) AS code_2
FROM country_continent
WHERE continent NOT LIKE '') AS t1
INNER JOIN (
SELECT
	*
FROM cities_full
WHERE population>200000) AS t2
ON t1.code_2=t2.country
ORDER BY continent DESC
LIMIT 15;
/*
 continent |      latitude       |     longitude      
-----------+---------------------+--------------------
 Oceania   |          -37.813938 |         144.963425
 Oceania   |          -36.866667 | 174.76666699999998
 Oceania   |          -32.927792 | 151.78448500000002
 Oceania   |             -34.424 |         150.893448
 Oceania   | -28.000290000000003 |         153.430878
 Oceania   | -31.952240000000003 |         115.861397
 Oceania   |          -42.883209 |         147.331665
 Oceania   |               -36.8 |             174.75
 Oceania   |  -9.464722199999999 |           147.1925
 Oceania   |          -34.928661 |         138.598633
 Oceania   |           -35.27603 |          149.13435
 Oceania   |          -43.533333 |         172.633333
 Oceania   |          -33.861481 |         151.205475
 Oceania   | -27.471009999999996 |         153.024292
 Europe    |           51.266667 |           7.183333
*/
/*
Maintenant je veux convertir le continent en code numérique
*/
SELECT
	*
FROM
(SELECT
	t1.continent AS continent,
	t2.latitude AS latitude,
	t2.longitude AS longitude
FROM (
SELECT
	country AS country,
	continent AS continent,
	LOWER(code_2) AS code_2
FROM country_continent
WHERE continent NOT LIKE '') AS t1
INNER JOIN (
SELECT
	*
FROM cities_full
WHERE population>200000) AS t2
ON t1.code_2=t2.country) AS t3;
/*
 continent |      latitude       |      longitude      
-----------+---------------------+---------------------
 Americas  |           -7.116667 |          -34.866667
 Americas  |               -26.3 |          -48.833333
 Americas  |                -7.2 |          -39.333333
 Americas  |          -21.751667 |          -43.352778
 Americas  | -23.183332999999998 |          -46.866667
 Americas  |          -22.561667 | -47.402778000000005
 Americas  |               -23.3 |              -51.15
 Americas  |            0.033333 |              -51.05
 Americas  |           -9.666667 |          -35.716667
 Americas  |          -22.655556 |          -43.015278
 Americas  | -3.1133330000000004 |          -60.025278
 Americas  |          -22.216667 | -49.933333000000005
 Americas  |          -23.416667 |          -51.916667
 Americas  |          -23.666667 |              -46.45
 Americas  |          -23.516667 | -46.183333000000005
*/
/*
Au lieu d'avoir le string "continent" je dois avec un entier pour l'apprentissage machine.
Je vais utiliser ce code :

1 : Americas
2 : Europe
3 : Africa
4 : Asia
5 : Oceania

En SQL je dois alors utiliser quelque chose comme :
*/
CASE
	WHEN (continent LIKE 'Americas')
		THEN 0
	WHEN (continent LIKE 'Europe')
		THEN 1
	WHEN (continent LIKE 'Africa')
		THEN 2
	WHEN (continent LIKE 'Asia')
		THEN 3
	WHEN (continent LIKE 'Oceania')
		THEN 4
END
/*
Par exemple ceci :
*/
SELECT
	country,
	LOWER(code_2),
	continent
FROM country_continent
WHERE continent NOT LIKE '';
/*
devient ceci :
*/
SELECT
	country,
	LOWER(code_2),
	continent,
	CASE
		WHEN (continent LIKE 'Americas')
			THEN 0
		WHEN (continent LIKE 'Europe')
			THEN 1
		WHEN (continent LIKE 'Africa')
			THEN 2
		WHEN (continent LIKE 'Asia')
			THEN 3
		WHEN (continent LIKE 'Oceania')
			THEN 4
	END
FROM country_continent
WHERE continent NOT LIKE '';
/*
Le tableau qui m'intéresse dès lors est celui-ci :
*/
SELECT
	CASE
		WHEN (continent LIKE 'Americas')
			THEN 0
		WHEN (continent LIKE 'Europe')
			THEN 1
		WHEN (continent LIKE 'Africa')
			THEN 2
		WHEN (continent LIKE 'Asia')
			THEN 3
		WHEN (continent LIKE 'Oceania')
			THEN 4
	END,
	phi,
	theta,
	cos(theta)*cos(phi) AS x,
	cos(theta)*sin(phi) AS y,
	sin(theta) AS z
FROM
(SELECT
	t1.continent AS continent,
	t2.latitude AS latitude,
	t2.longitude AS longitude,
	pi()*(t2.latitude)/180 AS theta, -- latitude en radians
	pi()*(t2.longitude)/180 AS phi -- longitude en radians
FROM (
SELECT
	country AS country,
	continent AS continent,
	LOWER(code_2) AS code_2
FROM country_continent
WHERE continent NOT LIKE '') AS t1
INNER JOIN (
SELECT
	*
FROM cities_full
WHERE population>200000) AS t2
ON t1.code_2=t2.country) AS t3;
/*
Voilà, j'ai la table que je voulais.
Je peux maintenant l'exporter dans un fichier CSV
*/

------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- On exporte le tableau vers le fichier CSV
\copy (SELECT CASE WHEN (continent LIKE 'Americas') THEN 0 WHEN (continent LIKE 'Europe') THEN 1 WHEN (continent LIKE 'Africa') THEN 2 WHEN (continent LIKE 'Asia') THEN 3 WHEN (continent LIKE 'Oceania') THEN 4 END, phi, theta, cos(theta)*cos(phi) AS x, cos(theta)*sin(phi) AS y, sin(theta) AS z FROM (SELECT t1.continent AS continent, t2.latitude AS latitude, t2.longitude AS longitude, pi()*(t2.latitude)/180 AS theta, pi()*(t2.longitude)/180 AS phi FROM (SELECT country AS country, continent AS continent, LOWER(code_2) AS code_2 FROM country_continent WHERE continent NOT LIKE '') AS t1 INNER JOIN (SELECT * FROM cities_full WHERE population>200000) AS t2 ON t1.code_2=t2.country) AS t3) to '//Users/NAC/Desktop/cities/Python/cities.csv' CSV HEADER;















