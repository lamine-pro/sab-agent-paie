# 🏛️ Modern Data Stack Lakehouse (Trino + Iceberg + MinIO)

Ce projet met en place une architecture **Data Lakehouse** moderne, locale et hautement scalable, basée sur le format de table ouvert **Apache Iceberg**. L'infrastructure est entièrement conteneurisée avec Docker pour un déploiement et une réplication rapides.

Le stockage des données est géré par **MinIO** (compatible S3), le catalogue de métadonnées est centralisé via **Iceberg REST Catalog** (adossé à PostgreSQL), et le moteur de requêtage distribué **Trino** permet d'interroger le tout en SQL à la vitesse de l'éclair.

---

## 🏗️ Architecture du Projet

Le Lakehouse est découpé en 3 couches distinctes :
1. **Moteur de calcul (Compute)** : Trino (v440) se charge de planifier et d'exécuter les requêtes SQL de manière distribuée.
2. **Catalogue & Métadonnées (Catalog)** : Un serveur Iceberg REST centralise la gestion des tables et stocke les pointeurs de métadonnées dans une base PostgreSQL (`metastore`).
3. **Stockage Objet (Storage)** : MinIO fait office de Data Lake en stockant les fichiers physiques au format optimisé Apache Parquet.

---

## 🚀 Démarrage Rapide

### Prerequis
* Docker et Docker Compose installés sur votre machine.
* Git.

### 1. Cloner le projet
```bash
git clone [https://github.com/ton-pseudo/mon-projet-lakehouse.git](https://github.com/ton-pseudo/mon-projet-lakehouse.git)
cd mon-projet-lakehouse
