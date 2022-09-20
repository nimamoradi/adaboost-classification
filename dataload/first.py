import glob
import pickle
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import gensim
import numpy as np
import scipy.spatial.distance as spatial
from scipy.stats import entropy
from numpy.linalg import norm
import warnings

warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
theta = 0.9


def jsd(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def readTopic(path):
    topics = {}
    with open(path, 'r', encoding='utf8') as file:
        lines = file.readlines()
        ListWord = []
        for line in lines:
            line = line.split('\n')[0]
            word = line.split('\t')
            if word[0] != '':
                if len(ListWord) != 0:
                    topics[key] = ListWord
                key = word[0]
                ListWord = []

            ListWord.append(word[1])
    if len(ListWord) != 0:
        topics[key] = ListWord
    return topics


def Make_Vector_For_Topics(topic_count, Topics):
    topic_vec = {}
    pretrained_embeddings_path = "../data_files/vectors.bin"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
    vocabs = word2vec.vocab.keys()
    for topic, ListWord in Topics.items():
        counter = 0
        topic_dash = np.zeros(word2vec.vector_size)
        for word in ListWord:
            word = word.lower()
            # word = word.decode('utf-8')
            word = lemmatizer.lemmatize(porter_stemmer.stem(word))
            if word in vocabs:
                counter = counter + 1
                x = word2vec.wv[word]
                norm1 = x / np.linalg.norm(x)
                topic_dash = topic_dash + norm1
        if counter != 0: topic_dash = topic_dash / counter
        topic_vec[topic] = topic_dash
    return topic_vec


def Compute_Topic_Similarity(TopicVec):
    Similarity = [[0 for i in range(len(TopicVec))] for j in range(len(TopicVec))]
    i = 0
    for topici, vceti in TopicVec.items():
        j = 0
        for topicj, vectj in TopicVec.items():
            Similarity[i][j] = Similarity[j][i] = spatial.cosine(vceti, vectj)
            j = j + 1
        i = i + 1
    return Similarity


def Make_Transitive(Similarity, TopicVec):
    transitive = []
    Tuple = []
    for i in range(0, len(TopicVec)):
        for j in range(i, len(TopicVec)):
            if Similarity[i][j] > theta and i != j:
                Tuple.append([i, j])
    for i in range(len(TopicVec)):
        for j in range(len(TopicVec)):
            if Similarity[i][j] > theta:
                transitive.append((i, j))
    adjacency = {}
    for i, j in transitive:
        adjacency.setdefault(i, set()).add(j)

    true = []
    for a, related in adjacency.items():
        for b in related:
            for c in adjacency[b]:
                if c in related:
                    if {a, b, c} not in true and len({a, b, c}) == 3:
                        true.append({a, b, c})
    return true, Tuple


def save_matrix(a):
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_matrix():
    with open('filename.pickle', 'rb') as handle:
        return pickle.load(handle)


def matrix_draw(similarity, topic_count):
    labels = ['topics', 'topics']
    similarity = np.tril(similarity, -1)
    y, x = np.histogram(similarity, bins=np.linspace(0, 1, 16))
    fig, ax = plt.subplots()
    ax.plot(x[:-1], y)
    plt.title('Similarity Histogram for {} topics'.format(topic_count))

    plt.show()


def main():
    topic_list = glob.glob("./../WordsInTopics/*.txt")
    print(topic_list)
    topic_list.sort(key=lambda x: int(x.replace("./../WordsInTopics\\WordsInTopics-", "").replace('.txt', '')))
    for it in topic_list:
        topic_count = int(it.replace("./../WordsInTopics\\WordsInTopics-", "").replace('.txt', ''))
        topics = readTopic(it)
        # print(topics)
        TopicVec = Make_Vector_For_Topics(topic_count, topics)

        Similarity = Compute_Topic_Similarity(TopicVec)
        print("Similarity ", topic_count, np.nansum(Similarity) / (topic_count * topic_count))

        Transitive, Tuple = Make_Transitive(Similarity, TopicVec)

        print(Transitive)
        print(Tuple)
        # matrix_draw(np.array(Similarity, dtype=np.double), str(topic_count))


if __name__ == "__main__":
    main()
