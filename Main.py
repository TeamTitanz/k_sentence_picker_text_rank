# coding=utf-8
from __future__ import print_function

from nltk.tokenize.punkt import PunktSentenceTokenizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx


def open_file(file_name):
    f = open(file_name, "r")
    return f.read()


def sentence_splitter(file_name):

    document = open_file(file_name)
    document.replace("." , " ")
    document = '. '.join(document.strip().split(' \n'))
    document.replace('\n', " ")

    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
    return sentences


def bag_of_words(sentences):
    for sentence in sentences:
        return Counter(word.lower().strip('.,') for word in sentence.split(' '))


def create_matrix(sentences):
    c = CountVectorizer()
    bow_matrix = c.fit_transform(sentences)
    return bow_matrix


def matrix_normalizer(bow_matrix):
    normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
    return normalized_matrix


def get_similairty_graph(bow_matrix):
    normalized_matrix = matrix_normalizer(bow_matrix)
    similarity_graph = normalized_matrix * normalized_matrix.T
    return similarity_graph


def sentence_ranker(similarity_graph):
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return scores


def score_sorter(scores, sentences):
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)),
                    reverse=True)
    return ranked


def ranker(file_name, k):
    sentences = sentence_splitter(file_name)
    bow_matrix = create_matrix(sentences)
    similarity_graph = get_similairty_graph(bow_matrix)
    scores = sentence_ranker(similarity_graph)
    ranked = score_sorter(scores, sentences)
    return ranked[0:k]


def store_sentences(n, k):
    r_matrix = []
    r_matrix.append([])
    for i in range(0, n):
        file_name = "Cases/case" + str(i) + str(".txt")
        ranked = ranker(file_name, k)

        file_name = "Output/case_" + str(i) + ".txt"
        f = open(file_name, "w")
        for j in range(0, k):


            print(ranked[j][1], file=f)



n = 26 # number of legal case docs
k = 20  # number of sentences to be returned by textrank
store_sentences(n, k)

# print(ranker("Cases/case0.txt", 10))
