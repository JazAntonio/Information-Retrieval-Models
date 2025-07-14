import numpy as np
import pandas as pd
from collections import Counter
from math import log

import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


def distancia_coseno(a, b):
    # Calcular el producto punto de los vectores
    producto_punto = np.dot(a, b)

    # Calcular las normas de los vectores
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)

    # Calcular la distancia del coseno
    distancia_coseno = producto_punto / (norma_a * norma_b)

    return distancia_coseno


def load_stopwords(stopwords_file):
    with open(stopwords_file, "r") as file:
        stopwords = []
        # Clean stopwords
        for word in file:
            w = word.strip()
            if w not in [""]:
                stopwords.append(w.lower())
        return stopwords


def load_documents(documents_path):
    with open(documents_path, "r") as file:
        text = file.read()

    documents = []
    fragments = text.split("*TEXT")[1:]

    for id, fragment in enumerate(fragments, start=1):

        lines = fragment.strip().split("\n")
        if lines:
            content = " ".join(lines[1:])
            content = content.strip()
            documents.append(content)

    return documents


def load_queries(queries_path):
    with open(queries_path, "r") as file:
        text = file.read()

    queries = []

    fragments = text.split("*FIND")[1:]

    for id, fragment in enumerate(fragments, start=1):
        lines = fragment.strip().split(maxsplit=1)
        if lines:
            queries.append(lines[1])
    return queries


def load_rel(rel_path):
    with open(rel_path, "r") as file:
        text = file.read()

    # Dividir el texto en líneas y filtrar las líneas vacías
    lineas = [linea.strip() for linea in text.split('\n') if linea.strip()]

    # Crear listas solo para líneas con elementos después del primer término
    listas_enteros = []
    for linea in lineas:
        elementos = linea.split()[1:]
        if elementos:
            listas_enteros.append([int(num) for num in elementos])

    return listas_enteros


def preprocess_doc(text, stopwords):

    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub(r'[^\D]', ' ', text)
    # text = re.sub(r'[^0-9]', ' ', text)

    text = text.lower()

    word_tokens = word_tokenize(text)

    stop_words = set(stopwords)
    filtered_word_tokens = [w for w in word_tokens if w not in stop_words]

    # retrieve stem from words
    ps = PorterStemmer()
    stemming = []

    for w in filtered_word_tokens:
        stem = ps.stem(w)
        stemming.append(stem)

    wordnet_lemmatizer = WordNetLemmatizer()

    lemmatization = []
    for w in stemming:
        lemma = wordnet_lemmatizer.lemmatize(w)
        lemmatization.append(lemma)

    return lemmatization


def documentos_tf_idf(path, stopwords):
    documentos = load_documents(path + ".ALL")
    docs_preprocesados = [preprocess_doc(documento, stopwords) for documento in documentos]

    cabecera_tmp = np.concatenate(docs_preprocesados)
    cabecera = np.unique(cabecera_tmp)
    cabecera = np.insert(cabecera, 0, "No.de documento")

    # Representación tf
    matriz_ocurrencias = np.zeros((len(docs_preprocesados), len(cabecera)), dtype=object)
    for id, documento in enumerate(docs_preprocesados):
        matriz_ocurrencias[id][0] = id+1
        ocurrencias = Counter(documento)
        for palabra, cantidad in ocurrencias.items():
            indice_cabecera = np.where(cabecera == palabra)[0][0]
            matriz_ocurrencias[id][indice_cabecera] = cantidad
    matriz_ocurrencias = np.insert(matriz_ocurrencias, 0, cabecera, axis=0)

    bolsa_conteo = pd.DataFrame(matriz_ocurrencias[1:, 1:], columns=matriz_ocurrencias[0, 1:], index=matriz_ocurrencias[1:, 0])
    bolsa_tf = bolsa_conteo.copy()

    i=0
    for documento in bolsa_tf.index:
        bolsa_tf.loc[documento] = bolsa_tf.loc[documento]/len(docs_preprocesados[i])
        i += 1

    # Representación binaria
    matriz_binaria = np.zeros((len(docs_preprocesados), len(cabecera)), dtype=object)
    for id, documento in enumerate(docs_preprocesados):
        matriz_binaria[id][0] = id+1
        ocurrencias = Counter(documento)
        for palabra, cantidad in ocurrencias.items():
            indice_cabecera = np.where(cabecera == palabra)[0][0]
            matriz_binaria[id][indice_cabecera] = 1
    matriz_binaria = np.insert(matriz_binaria, 0, cabecera, axis=0)

    bolsa_binaria = pd.DataFrame(matriz_binaria[1:, 1:], columns=matriz_binaria[0, 1:], index=matriz_binaria[1:, 0])
    docs_por_palabra = bolsa_binaria.sum()
    docs_por_palabra = docs_por_palabra.to_frame().T

    # Uso de la representación binaria para la representación idf
    matriz_idf = np.zeros((len(docs_preprocesados), len(cabecera)), dtype=object)
    for id, documento in enumerate(docs_preprocesados):
        matriz_idf[id][0] = id+1
        ocurrencias = Counter(documento)
        for palabra, cantidad in ocurrencias.items():
            indice_cabecera = np.where(cabecera == palabra)[0][0]
            matriz_idf[id][indice_cabecera] = log(423/(1+docs_por_palabra[palabra][0]))
    matriz_idf = np.insert(matriz_idf, 0, cabecera, axis=0)
    bolsa_idf = pd.DataFrame(matriz_idf[1:, 1:], columns=matriz_idf[0, 1:], index=matriz_idf[1:, 0])

    return bolsa_conteo, bolsa_tf, bolsa_idf, cabecera


def consultas_binario(path, stopwords, cabecera):
    queries_prepocesadas = load_queries(path + ".QUE")
    queries = [preprocess_doc(query, stopwords) for query in queries_prepocesadas]

    matriz_binaria = np.zeros((len(queries), len(cabecera)), dtype=object)
    for id, query in enumerate(queries):
        matriz_binaria[id][0] = f"Consulta {id+1}"
        ocurrencias = Counter(query)
        for palabra, cantidad in ocurrencias.items():
            try:
                indice_cabecera = np.where(cabecera == palabra)[0][0]
                matriz_binaria[id][indice_cabecera] = 1
            except IndexError:
                continue
    matriz_binaria = np.insert(matriz_binaria, 0, cabecera, axis=0)
    bolsa_binaria = pd.DataFrame(matriz_binaria[1:, 1:], columns=matriz_binaria[0, 1:], index=matriz_binaria[1:, 0])

    return bolsa_binaria


def ap(documentos_recuperados, documentos_relevantes):
    N = len(documentos_recuperados)
    suma = 0
    aciertos = 0

    for i in range(N):
        if documentos_recuperados[i] in documentos_relevantes:
            aciertos += 1
            rel_i = 1
        else:
            rel_i = 0
        P_i = aciertos / (i + 1)
        suma += (P_i * rel_i)
    AP = suma / N
    return AP


def modelo_rocchio(consulta, docs_maximos, documentos_relevantes, a, b, c = 0.1):
    documentos_recuperados = {}

    for documento in bolsa_tf_idf.index:
        documentos_recuperados[documento] = distancia_coseno(consultas_binario.loc[consulta],
                                                                bolsa_tf_idf.loc[documento])
    diccionario_ordenado_por_valor = dict(sorted(documentos_recuperados.items(), key=lambda x: x[1], reverse = True))

    docs_recuperados = {}
    contador = 1
    for clave in diccionario_ordenado_por_valor:
        if diccionario_ordenado_por_valor[clave] != 0.0 and contador <= docs_maximos:
            docs_recuperados[clave] = diccionario_ordenado_por_valor[clave]
            contador = contador + 1
        else:
            break

    claves = list(docs_recuperados.keys())
    # valores = list(docs_recuperados.values())

    AP = ap(claves, documentos_relevantes)
    print("Valor de AP sin Rocchio: ", "{:.5f}".format(AP))

    # Creamos una lista para los ids de los documentos relevantes recuperados
    D_r = []

    # Con las siguiente lineas comentadas podemos seleccionar a los documentos relevantes y a los irrelevantes,
    # si así lo deseamos, con un enfoque supervisado

    # Creamos una lista para los ids de los documentos irrelevantes recuperados
    # D_ir = []
    # for clave in claves:
    #     if clave in documentos_relevantes:
    #         D_r.append(clave)
    #     else:
    #         D_ir.append(clave)

    for i in range(len(claves)):
        if i<3:
            D_r.append(claves[i])       #Elegimos a los tres primeros documentos recuperados como los relevantes
        else:
            # D_ir.append(claves[i])    #Podemos rellenar D_ir con todos los documentos restantes
            continue                    #En esta tarea la lista D_ir es una lista vacía

    longitud_vector = len(consultas_binario.loc[consulta])

    # Calculamos el primer sumando de la fórmula de Rocchio
    sumando1 = a * consultas_binario.loc[consulta]

    # Calculamos el segundo sumando de la fórmula de Rocchio
    sumando2 = np.zeros(longitud_vector)
    for id_doc in D_r:
        sumando2 += bolsa_tf_idf.loc[id_doc]
    sumando2 = b * (1 / len(D_r)) * sumando2

    #En caso de ser usado, aquí calculamos el sumando referente a los documentos irrelevantes
    #Calculamos el tercer sumando de la fórmula de Rocchio
    # sumando3 = np.zeros(longitud_vector)
    # for id_doc in D_ir:
    #     sumando3 += bolsa_tf_idf.loc[id_doc]
    # sumando3 = c * (1 / len(D_ir)) * sumando3

    # consultas_binario.loc[consulta] = sumando1 + sumando2 - sumando3
    consultas_binario.loc[consulta] = sumando1 + sumando2

    #Segunda recupearación
    documentos_recuperados = {}

    for documento in bolsa_tf_idf.index:
        documentos_recuperados[documento] = distancia_coseno(consultas_binario.loc[consulta],
                                                             bolsa_tf_idf.loc[documento])
    diccionario_ordenado_por_valor = dict(sorted(documentos_recuperados.items(), key=lambda x: x[1], reverse=True))

    docs_recuperados = {}
    contador = 1
    for clave in diccionario_ordenado_por_valor:
        if diccionario_ordenado_por_valor[clave] != 0.0 and contador <= docs_maximos:
            docs_recuperados[clave] = diccionario_ordenado_por_valor[clave]
            contador = contador + 1
        else:
            break

    claves = list(docs_recuperados.keys())
    # valores = list(docs_recuperados.values())

    AP = ap(claves, documentos_relevantes)
    print("Valor de AP con Rocchio: ", "{:.5f}".format(AP))


path = "./Datos/TIME"
stopwords = load_stopwords(path+".STP")
consultas = load_queries(path+".QUE")
documentos_relevantes = load_rel(path + ".REL")

bolsa_conteo, bolsa_tf, bolsa_idf, cabecera = documentos_tf_idf(path, stopwords)
bolsa_tf_idf = bolsa_tf*bolsa_idf
consultas_binario = consultas_binario(path, stopwords, cabecera)

cantidad_documentos_recuperados = 100
print(f"Cantidad de documentos recuperados: {cantidad_documentos_recuperados}\n")

for k in range(1,11):

    print(f"##### Consulta {k} #####")
    modelo_rocchio(f"Consulta {k}", cantidad_documentos_recuperados, documentos_relevantes[k - 1], 0.8, 0.85)
    print("\n")