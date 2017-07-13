from __future__ import print_function

import os
import os.path
from collections import Counter
from multiprocessing import Pool

import networkx as nx
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#nltk.download('punkt')


k = 40 # number of sentences to be returned by textrank
number_of_process = 4


def open_file(file_name):
    f = open(file_name, "r")
    return f.read()


def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])


def sentence_splitter(file_name):

    fp = open(file_name)
    data = remove_non_ascii(fp.read())

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    content = tokenizer.tokenize(data)
    sentences = []

    for sentence in content:
        sentence = sentence.replace('\r', '').replace('\n', '')
        sentences.append(sentence.replace(".", " "))
    # print ('\n'.join(sentences))

    content = ". ".join(sentences)
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(content)
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
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ranked


def ranker(file_name, k):
    sentences = sentence_splitter(file_name)
    bow_matrix = create_matrix(sentences)
    similarity_graph = get_similairty_graph(bow_matrix)
    scores = sentence_ranker(similarity_graph)
    ranked = score_sorter(scores, sentences)
    return ranked[0:k]


def process_file(file_name):
    global k
    print (file_name)
    rank_file_name = os.path.join("Cases", file_name + ".txt")
    ranked = ranker(rank_file_name, k)

    output_file_name = os.path.join("Output", file_name + ".txt")
    f = open(output_file_name, "w")
    for j in range(0, k):
        print(ranked[j][1].replace(" .", "."), file=f)

    return True


def store_sentences():
    global number_of_process
    r_matrix = []
    r_matrix.append([])

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Cases")
    # print path
    fileNames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    nameIndex = []
    for fn in fileNames:
        nameIndex.append(fn.split(".")[0])
    nameIndex.sort()

    p = Pool(number_of_process)
    p.map(process_file, nameIndex)


if __name__ == '__main__':
    print("k sentence picking is started")
    store_sentences()
    print("k sentence picking is finished")
