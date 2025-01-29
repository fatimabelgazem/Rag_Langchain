from flask import Flask, render_template, request, jsonify
from zenml.pipelines import pipeline
from zenml.steps import step
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importation des étapes du pipeline ZenML
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
import os
from flask import Flask, render_template, request, jsonify, session

# Définir les constantes pour Chroma et les documents
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Initialiser l'application Flask

app = Flask(__name__)
app.secret_key = "124"
# Charger les documents (étape ZenML)
@step(enable_cache=False)
def load_documents_step() -> list[Document]:
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

# Fonction pour obtenir l'embedding
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
@step
def split_documents_step(documents: list[Document]) -> list[Document]:
    """Découper les documents en morceaux plus petits."""
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
@step
def add_to_chroma_step(chunks: list[Document]) -> str:
    """Ajouter les documents découpés à la base de données Chroma et retourner son chemin."""
  
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    
    # Calculer les IDs des pages
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Ajouter ou mettre à jour les documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Nombre de documents existants dans la DB : {len(existing_ids)}")
    
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Ajout de nouveaux documents : {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ Aucun document à ajouter")
    
    # Retourner le chemin de la base de données
    return CHROMA_PATH
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks
@step
def clear_database_step() -> None:
    """Effacer la base de données Chroma si nécessaire."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
# Étape pour effectuer une recherche dans la base de données Chroma

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
@step(enable_cache=False)
def query_rag_step(chroma_path: str, query_text: str) -> str:
    """Recherche dans la base de données Chroma et génère la réponse avec Ollama."""
 
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Recherche dans la base de données
    results = db.similarity_search_with_score(query_text, k=5)

    # Extraire le contexte pour le prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Génération de la réponse avec Ollama
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Extraire les sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    print(formatted_response)
    with open("response.txt", "w") as f:
        f.write(response_text)
    
    return response_text

# Définir le pipeline ZenML
@pipeline
def rag_pipeline(query_text: str = "") ->str:
    """Pipeline ZenML pour le processus RAG avec recherche de requêtes."""
    # Charger les documents
    documents = load_documents_step()
 
    # Découper les documents
    chunks = split_documents_step(documents)
  
    # Ajouter les documents découpés à Chroma et obtenir le chemin
    chroma_path = add_to_chroma_step(chunks)
    
    # Rechercher et répondre à la requête
    c=query_rag_step(chroma_path=chroma_path, query_text=query_text)
    return c

# Création de l'API Flask
@app.route('/')
def home():
    return render_template('index.html')  # Page HTML avec le formulaire

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question', '')
    
    if question:
        # Charger l'historique de la session
        if 'conversation' not in session:
            session['conversation'] = []
        
        # Ajouter la question à la session
        session['conversation'].append({'role': 'user', 'text': question})
        
        # Exécuter le pipeline pour obtenir une réponse
        first_pip = rag_pipeline(query_text=question)
        with open("response.txt", "r") as f:
            response_text = f.read()
        
        # Ajouter la réponse à la session
        session['conversation'].append({'role': 'bot', 'text': response_text})
        
        return jsonify({'question': question, 'response': response_text, 'conversation': session['conversation']})
    
    return jsonify({'error': 'No question provided'}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)