# PROJET_STATAPP_MILLIMAN
## Projet de génération des données synthétiques avec du LLM en assurance

###  Résumé du projet

Ce projet explore l’utilisation des **modèles de langage (LLM)** open source pour **générer des données synthétiques réalistes** dans le secteur de l’assurance.  
L’objectif est de démontrer comment ces modèles peuvent pallier les limites des données réelles(sensibilité, déséquilibre, rareté) tout en améliorant la **détection de fraude**, la **segmentation des risques** et le **ciblage client**.  
Ce travail s’inscrit dans une démarche d’**innovation responsable**, où l’intelligence artificielle générative devient un outil au service de la **confidentialité**, de la **qualité des modèles** et du **développement durable de la donnée**.

---

### Description du sujet

Dans le secteur de l’assurance, la donnée constitue un **actif stratégique** pour l’évaluation du risque, la prévention de la fraude et la personnalisation des produits.  
Cependant, plusieurs défis persistent dans l’exploitation de ces données :

- **Sensibilité des données** : les informations personnelles, médicales et financières nécessitent une protection stricte.  
- **Déséquilibre des classes** : les événements rares, comme les fraudes, sont sous-représentés, rendant les modèles moins performants.  
- **Rareté de scénarios extrêmes** : peu de données existent pour les cas atypiques (catastrophes naturelles, sinistres massifs, nouvelles pathologies, etc.).

La **génération de données synthétiques** se présente comme une solution innovante à ces enjeux :  elle permet de **créer des données artificielles mais réalistes**, préservant la confidentialité tout en enrichissant la diversité et la représentativité des jeux de données.

---

### Objectifs du projet
L’objectif principal est de **mettre en œuvre et d’évaluer des techniques de génération de données synthétiques** à l’aide de modèles de langage (LLM) open source.  

---

### Plan du projet 
Le projet contient 3 dossiers dont un dossier annexe. 
   
   1. Analyse descriptive "Fraud Detection Dataset"
   Cette partie est une analyse descriptive du jeu de données utilisé. Elle permet une première familiarisation avec le dataset utilisé. Elle comprend notamment des statistiques descriptives, une mise en évidence du déséquilibre des classes du jeu de données ainsi qu'une étude sur la distribution et la nature des valeurs manquantes présentes. 

   2. Génération de données synthétiques
   Ce dossier contient les deux parties de génération synthétiques du projet. 
   D'une part la partie "Modèles Statistiques" et d'autre part la partie "LLMs".

   La partie "Modèles Statistiques" s'intéresse à la génération de données synthétiques grâce à des méthodes statistiques. 
   L'algorithme utilisé est le SMOTE qui permet de rééquilibrer le jeu de données. 

   La partie "LLMs" quant à elle s'intéresse à la génération de données synthétiques directement grâce à des LLMs.



