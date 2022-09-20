import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_matrix():
    with open('filename.pickle', 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    A = load_matrix()
    A = np.array(A)
    print(A)
    A = A * 1000
    y, x = np.histogram(A, bins=np.arange(500))
    fig, ax = plt.subplots()
    ax.plot(x[:-1], y / 1000)

    plt.xlabel('topic count')
    plt.ylabel('similarity')

    fig.show()
