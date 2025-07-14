from collections import Counter
from dataclasses import dataclass

import nltk
import re
from nltk import word_tokenize
from nltk.corpus.reader import documents
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import csv

ALFABETO = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"
    , "N", "O", "P", "Q", "R", "s", "T", "U", "V", "W", "X", "Y", "Z", ""]


def load_stopwords(stopwords_file: str) -> list[str]:
    with open(stopwords_file, "r") as file:
        stopwords = []
        # Clean stopwords
        for word in file:
            w = word.strip()
            if w not in ALFABETO:
                stopwords.append(w.lower())
        return stopwords


def load_documents(documents_path: str) -> list[str]:
    with open(documents_path, "r") as file:
        text = file.read()

    documents = []
    fragments = text.split("*TEXT")[1:]

    for id, fragment in enumerate(fragments, start=1):

        lines = fragment.strip().split("\n")
        if lines:
            content = " ".join(lines[1:])
            documents.append(content)

    return documents


def load_queries(queries_path: str) -> list[str]:
    with open(queries_path, "r") as file:
        text = file.read()

    queries = []

    fragments = text.split("*FIND")[1:]

    for id, fragment in enumerate(fragments, start=1):
        lines = fragment.strip().split("\n", maxsplit=1)
        if lines:
            queries.append(lines[1])

    return queries


def preprocess_doc(text: str, stopwords: list[str]) -> list:
    # clean text
    text = text.replace("\n", " ")
    text = text.strip()
    text = re.sub(r'[^\w]', ' ', text)

    # normalisation, case-folding
    text = text.lower()

    # tokenise
    word_tokens = word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords)
    filtered_word_tokens = [
        w for w in word_tokens if w not in stop_words]

    # retrieve stem from words
    ps = PorterStemmer()
    stemming = []

    for w in filtered_word_tokens:
        stem = ps.stem(w)
        stemming.append(stem)

    # retrieve lemma from words
    wordnet_lemmatizer = WordNetLemmatizer()

    lemmatization = []
    for w in stemming:
        lemma = wordnet_lemmatizer.lemmatize(w)
        lemmatization.append(lemma)

    return lemmatization

def matriz_de_ocurrencias_de_documentos(path, stopwords):
    documentos_preprocesados = load_documents(path+".ALL")
    documentos = [preprocess_doc(documento, stopwords) for documento in documentos_preprocesados]
    cabecera_tmp = []
    for documento in documentos:
        cabecera_tmp.extend(documento)
    cabecera = list(set(cabecera_tmp))
    cabecera.insert(0, "No.")
    matriz = [[0]*len(cabecera) for _ in range(len(documentos))]
    for id, documento in enumerate(documentos):
        matriz[id][0] = f"Documento {id+1}"
        ocurrencias = Counter(documento)
        for palabra, cantidad in ocurrencias.items():
            indice_cabecera = cabecera.index(palabra)
            matriz[id][indice_cabecera] = cantidad
    return matriz, cabecera


def matriz_de_occurrencias_queries(path, stopwords, cabecera):
    queries_prepocesadas = load_queries(path+".QUE")
    queries = [preprocess_doc(query, stopwords) for query in queries_prepocesadas]

    matriz_queries = [[0] * len(cabecera) for _ in range(len(queries))]
    for id, query in enumerate(queries):
        matriz_queries[id][0] = f"Consulta {id+1}"
        ocurrencias = Counter(query)
        for palabra, cantidad in ocurrencias.items():
            try:
                indice_cabecera = cabecera.index(palabra)
                matriz_queries[id][indice_cabecera] = cantidad
            except ValueError:
                continue
    return matriz_queries

def escribir_csv(cabecera, matriz, nombre):
    matriz.insert(0,cabecera)
    with open(nombre, 'w', newline='', encoding='utf-8') as archivo:
        ocurrencia = csv.writer(archivo)
        for fila in matriz:
            ocurrencia.writerow(fila)

path = "./Datos/TIME"
stopwords = load_stopwords(path+".STP")

matriz, cabecera = matriz_de_ocurrencias_de_documentos(path,stopwords)
matriz_de_queries = matriz_de_occurrencias_queries(path,stopwords,cabecera)

escribir_csv(cabecera,matriz_de_queries, "Matriz_de_representación_de_queries.csv")
escribir_csv(cabecera,matriz,"Matriz_de_frecuencias.csv")