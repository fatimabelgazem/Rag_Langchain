# 🔍 RAG Chatbot avec Flask, ZenML, Chroma et Ollama

## 📌 Description

Ce projet implémente un chatbot basé sur le paradigme Retrieval-Augmented Generation (RAG) en utilisant Flask, ZenML, ChromaDB et Ollama pour répondre aux questions des utilisateurs sur la base de documents PDF chargés dynamiquement.

## 🏗️ Architecture du Projet

Flask : Fournit une API pour interagir avec le chatbot.

ZenML : Gère le pipeline de traitement des documents et des requêtes.

ChromaDB : Base de données vectorielle pour la recherche des passages pertinents.

Ollama : Modèle LLM pour générer des réponses basées sur les passages récupérés.

## 📦 Installation

Assurez-vous d'avoir Python installé, puis exécutez les commandes suivantes :

 1- Cloner le projet
```bash 
git clone <lien_du_repo>
cd <nom_du_repo>

```

 2- Installer les dépendances
 
```bash

pip install flask zenml langchain langchain-community langchain-chroma chromadb ollama opencv-python PyMuPDF

```
##🚀 Utilisation

Démarrer l'API Flask :

```bash
python app.py
```

Accéder à l'interface :

Ouvrez http://127.0.0.1:5000/ dans votre navigateur.

Envoyer une requête à l'API :

curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "Quelle est la capitale de la France ?"}'

## 📚 Fonctionnalités du Pipeline RAG

Le pipeline suit les étapes suivantes :

Chargement des documents (fichiers PDF).

Segmentation des documents en petits morceaux.

Stockage des embeddings dans ChromaDB.

Recherche de passages pertinents via Similarity Search.

Génération de réponse avec le modèle Ollama.

## 🧠 Modèles Ollama

Le projet utilise les modèles suivants :

nomic-embed-text : Génère des embeddings pour la recherche vectorielle.

llama3 : Génère des réponses basées sur les passages trouvés.

Vous pouvez tester d'autres modèles disponibles avec :

from langchain_community.llms.ollama import Ollama
model = Ollama(model="mistral")

## 🛠️ Personnalisation

Modifier DATA_PATH pour charger d'autres documents.

Ajuster chunk_size et chunk_overlap dans RecursiveCharacterTextSplitter pour un meilleur découpage.

Changer le modèle Ollama en mettant à jour model="llama3" dans Ollama(model=...).
