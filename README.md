# Projet de Recommender System - Rapport

## Choix pour la méthode de base et pour le traitement des données

### Méthode de base : Collaborative Filtering

Pour mon système de recommandation, j'ai choisi d'utiliser le collaborative filtering comme méthode de base, plus précisément celle présentée dans le cours 4. D'après ce même cours, voici les avantages de cette méthode :

* **Pas de connaissances de domaine nécessaires** : Le collaborative filtering ne nécessite pas de connaissances spécifiques sur les films ou les utilisateurs pour fonctionner efficacement.
* **Découverte de nouveaux intérêts** : Le modèle peut aider les utilisateurs à découvrir de nouveaux films. Même si le système ne sait pas que l'utilisateur est intéressé par un certain film, il peut le recommander parce que des utilisateurs similaires l'ont apprécié.
* **Simplicité des données** : Le système a seulement besoin de la matrice de notation pour entraîner un modèle de collaborative filtering. En particulier, il n'a pas besoin de caractéristiques contextuelles.

Cette méthode semble donc être un bon cas de départ. C'est aussi une méthode assez simple à implémenter et qui, d'après les exemples vus en cours, obtient facilement des résultats acceptables.

### Choix et Traitement des données

Avant de pouvoir implémenter et entraîner cet algorithme de recommandation, j'ai besoin de préparer toutes les données dont je vais avoir besoin. Voici ce qui a été choisi ou effectué :

#### IMDb

Au sein de IMDb, je n'ai conservé que les datasets:
- title.basics (pour avoir accès aux informations sur les films)
- title.crew (pour avoir accès aux directeurs (je n'ai pas pris celui associé avec les noms puisque utiliser leur vrai nom ou leur id ne change pas grand-chose))
- title.ratings (pour obtenir la note moyenne)

Pour lier ces datasets entre eux, j'ai simplement eu à utiliser la colonne correspondant à l'id des films.

Le système doit simplement proposer des films. Le dataset IMDb contenant bien plus que ça (films, séries, jeux), un premier nettoyage a eu lieu pour supprimer tout ce qui n'était pas un film (différent du type movie).

#### Merge des deux

Pour lier movieLens et le dataset de IMDb, j'ai décidé d'utiliser deux colonnes: celles du titre et celle de la date de début du film. Cependant cela a nécessité un nettoyage de la colonne titre de movieLens : celle-ci était au format "titre (année)" ce qui pose un problème puisque IMDb contient seulement le titre dans cette colonne.
Un premier nettoyage a donc eu lieu pour séparer la colonne titre de movieLens en deux colonnes: title et year.

Une fois cela fait, il a suffi de lier les deux. Le merge utilisé fait en sorte de conserver tous les films de movieLens et de les lier aux films correspondants dans IMDb. Ce choix est fait pour éviter tous soucis avec les id de la matrice d'interactions user-movie (cela évite des soucis de films qui sont en trop ou qui ont disparu).

D'autres nettoyages du dataframe résultant de ce merge ont ensuite été nécessaires :
- remplacement des nan par d'autres valeurs (0, moyenne ou string vide)
- regroupement des deux colonnes de genres (en ne gardant ainsi qu'une seule colonne de genres et en enlevant les duplicatas)
- suppression des colonnes qui ne sont pas importantes
- suppression des duplicatas dans les lignes dûes à plusieurs versions d'un film (version longue et courte par exemple)

#### Matrice d'intéractions user-movie

Le dataframe ratings de movieLens sera utilisé par la suite pour créer la matrice Y qui contient les notes de chaque utilisateur pour chaque film et la matrice R correspondant aux actions des utilisateurs (s'ils ont déjà ou non noté un film).

Avant de créer ces matrices cependant, de nouvelles lignes y sont ajoutées pour permettre par la suite une manipulation permettant d'utiliser les features de chaque film (genres, directors etc). Ces lignes correspondent aux films de movieLens qui n'ont jamais été notés par aucun utilisateur.

### Feature Engineering

Tout a déjà été expliqué dans le code mais remettons-le aussi ici.

Les caractéristiques suivantes se sont révélées intéressantes à conserver pour aider notre modèle à faire de meilleurs choix :
- genres
- directors
- isAdult
- averageRating

Et voici les raisons de ce choix :
- Genres et directors sont choisis car les gens ont souvent des préférences pour des genres ou des préférences pour certains directeurs de films.
- isAdult est choisi car certaines personnes peuvent préférer des films considérés comme pour les adultes plutôt que des films plus enfantins (type dessins animés ou film d'animation).
- averageRating est choisi car c'est une donnée que les gens peuvent regarder pour choisir un film (si un film est bien noté en moyenne, on aura tendance à le préférer). Cela peut également aider si un film n'est pas beaucoup noté ou n'est pas du tout noté par les utilisateurs.


Une fois ces caractéristiques choisies, il a fallu les modifier pour pouvoir les utiliser dans l'algorithme de recommendation. Voici les transformations effectuées :
- transformations des string genres et directors en liste de string
- encodage des genres et des directors en matrice binaire pour être utilisés comme features
- utilisation de isAdult et de averageRating (remis sur 5) comme features
- scaling de la matrice de features pour donner le même poids à chacune d'entre elles et pour lui permettre d'être utilisée plus tard dans l'entraînement

## Expériences et résultats correspondants

A partir de maintenant, la matrice d'intéractions sera découpée en 3 parties : train, validation et test.
Aucun film n'est noté dans plusieurs parties : si un film possède une note dans train, il sera à 0 dans validation et test.
La matrice R s'adaptera également en conséquence.

Le modèle s'entrainera donc avec train, vérifiera ses métriques intermédiaires avec validation (lors de l'utilisation de optuna par exemple) et donnera ses métriques finales à la toute fin avec test.

### 1. Utilisation de la matrice d'interaction user-movie avec des hyperparamètres aléatoires

Dans cette première expérience, j'ai utilisé uniquement la matrice d'interaction user-movie au sein de l'algorithme de collaborative filtering, avec un choix d'hyperparamètres aléatoire (j'ai simplement conservé ceux donnés dans le cours).

Pour cette expérience, le choix de métrique était l'écart absolu moyen entre les notes prédies et les véritables notes. Cette métrique permet de voir si le modèle a tendance à prédire ou non une note proche de la réalité (plus il est élevé, plus les notes prédies ont tendance à être loin de la réalité).

Les résultats étaient passables pour cette première expérience mais pouvaient être largement améliorés: l'erreur absolue moyenne était de 3 sur le set de test (ce qui est assez élevé pour une note sur 5).

### 2. Système hybride avec ajout de biais supplémentaires

Pour améliorer les résultats, j'ai décidé de passer à un système hybride. J'ai donc conservé l'approche de collaborative filtering, mais ai ajouté des biais supplémentaires liés aux caractéristiques propres des films, comme les genres, les directeurs, s'il s'agit d'un film pour adulte... 

Ce biais est calculé grâce à une variable supplémentaire qui se met à jour à chaque itération du programme et qui contient les poids de chaque features pour un film. La matrice contenant les caractéristiques est également utilisée (celle décrite dans features engineering).

Bien que le choix des hyperparamètres soit resté aléatoire (toujours ceux du cours), les résultats obtenus étaient déjà meilleurs, avec une erreur absolue moyenne comprise entre 2 et 1.8 sur le set de test.

### 3. Optimisation des hyperparamètres avec Optuna et système hybride + changement de métriques

Dans cette dernière expérience, j'ai utilisé l'algorithme Optuna pour trouver de meilleurs hyperparamètres tout en conservant l'algorithme hybride décrit précédemment.

Un changement de métrique prend également place durant cette troisième expérience. Pour le choix des hyperparamètres avec optuna et pour vérifier la qualité du modèle, nous utilisons à présent la racine carrée de l'erreur quadratique moyenne plutôt que l'erreur absolue moyenne. Cela permet de donner plus de poids aux grands écarts de notes puisque c'est ce que nous voulons éviter et donc que nous devons prendre en compte.

Avec optuna et ces nouveaux hyperparamètres, j'arrive à un modèle avec comme métrique 1.3 sur le set de validation (1.29) et le set de test (1.32). Si l'ancienne métrique avait été gardée, le modèle serait aux environs de 1 pour les deux (donc une nette amélioration par rapport aux versions précédentes).

Une nouvelle métrique est également introduite pour vérifier la qualité du modèle. Savoir que les notes sont bien devinées est une chose. Mais la pertinence des suggestions est également importante, surtout dans notre cas où nous voulons suggérer un film qui plairait à notre couple.La précision du modèle est donc également calculée (sachant que les notes >= 4 sont considérées comme des bons choix). Avec cette troisième version, la précision obtenue est de 0.72 (donc 72%) sur le set de validation et 0.76 (76%) sur le set de test. Ce n'est pas un score extrêmement élevé mais cela reste un résultat correct.

## Choix d'un film

Pour des questions de mémoire, j'ai décidé de calculer les notes de chaque utilisateur pour les films puis de faire la moyenne de ces notes pour un couple d'utilisateurs. Cela permet d'avoir une taille de matrice plus agréable à gérer que si nous devions faire tourner l'algorithme avec une matrice contenant chaque combinaison de couple possible (surtout que ce genre de matrice grandirait fortement à chaque ajout d'un seul utilisateur !).

Pour choisir un film, nous avons donc la marche à suivre suivante :
- récupérer les notes pour chaque film pour un couple d'utilisateurs (récupérer les notes de chacun, calculer la moyenne pour chaque film et utiliser ces moyennes)
- calculer un score de couple :
  - si le film a été vu par les deux personnes, alors il est ignoré (sauf si bien évidemment ils n'ont aucun nouveau film à voir mais je doute que ce cas arrive souvent)
  - si le film a été vu par seulement l'un des deux membres du couple, son score de couple est sa note
  - si le film n'a été vu par aucun des deux, son score de couple est sa note + 1
  
Ce système permet de faire en sorte de donner un avantage aux films qui sont nouveaux pour les deux personnes sans pour autant prendre le risque d'avoir un film qu'aucun des d'eux n'aimerait prendre le pas sur les autres.

Une fois ce score calculé, il suffit de prendre le film avec le meilleur score (ou l'un des films avec le meilleur score) et le renvoyer: il s'agit donc du film proposé pour ce couple.

## Conclusion

Pour conclure nous avons donc un modèle de recommendation qui:
- utilise le collaborative filtering avec la matrice d'interactions des users-movies
- utilise également les caractéristiques des films pour prendre ses décisions (genres, isAdult, directors, averageRating) pour ajouter un biais supplémentaire
- se sert de la racine carrée de son erreur quadratique moyenne pour trouver les bons hyperparamètres
- se sert de la racine carrée de son erreur quadratique et de sa précision pour vérifier qu'il s'agit d'un bon modèle

Les notes prédites sont ensuite utilisées pour choisir un film correspondant à un couple (avec la moyenne de leurs notes respectives et d'un léger biais).

Il est sûrement possible d'améliorer encore les performances du modèle:
- en utilisant d'autres caractéristiques pour les films
- en trouvant des hyperparamètres encore plus adaptés
- en utilisant un autre algorithme de recommendation plus adapté

Mais les performances obtenues restent correctes.