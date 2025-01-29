from setuptools import setup, find_packages

setup(
    name="rag_flask_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'zenml',
        'langchain',
        'langchain_community',
        'langchain_chroma',
        'ollama',
    ],
)
