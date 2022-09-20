from dataload.DataPoint import DataPoint
from dataload.load_data import read_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import pandas as pd

    size = 502

    tweet_df = pd.read_csv("../data_files/no_filter/tweet_topic_per_user.csv", sep=' ')

    normal = [0] * size
    auti = [0] * size

    for row in tweet_df.iterrows():
        row = row[1]
        for topic in row['text'].split(','):

            topic = int(topic)
            if row['label'] == 1:
                normal[topic] += 1
            elif row['label'] == 0:
                auti[topic] += 1

    list_of_point = []
    whole = [sum(x) for x in zip(normal, auti)]
    print(type(whole))
    for inx, item in enumerate(whole):
        list_of_point.append(DataPoint(inx, item))

    # To return a new list, use the sorted() built-in function...
    list_of_point = sorted(list_of_point, key=lambda x: x.value, reverse=True)
    print("sorted ", list_of_point)

    transformed_list_auti = size * [0]
    transformed_list_normal = size * [0]
    size_plus = size - 1
    for inx, item in enumerate(list_of_point):
        if inx % 2.0 == 0:  # if it is even
            transformed_list_auti[inx] = auti[item.index]
            transformed_list_normal[inx] = normal[item.index]
        else:
            print(size_plus, inx)
            transformed_list_auti[size_plus - inx] = auti[item.index]
            transformed_list_normal[size_plus - inx] = normal[item.index]

    fig, ax = plt.subplots()
    ax.plot(transformed_list_normal, color="#00ff00", label='normal users')
    ax.plot(transformed_list_auti, color='#ff0000', label='autistic users')

    ax.legend()
    plt.xlabel('topic number')
    plt.ylabel('repentance')

    fig.show()
