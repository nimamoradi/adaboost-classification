import pandas as pd
import numpy as np

concat_char = "_"


def read_data(path):
    return pd.read_csv(path, header=None)


def save_data(path, df):
    df.to_csv(path, index=False)


def look_up_tuple(num1, new_headers):
    for tuple in new_headers:
        li = [int(n) for n in tuple.split(concat_char)]
        if num1 in li:
            return tuple
    return None


def new_items_dataframe(topic_tuples, old_df):
    new_headers = []
    for tuple in topic_tuples:
        li = list(tuple)
        so_li = sorted(li)
        col = str(so_li[0]) + concat_char + str(so_li[1])
        if col not in new_headers:
            new_headers.append(col)
    print(len(old_df))
    my_df = pd.DataFrame(np.zeros((len(old_df), len(new_headers))), columns=new_headers)

    for index, user in old_df.iterrows():

        for topic_id, topic_count in user[1:500].items():
            if topic_count > 0:
                col = look_up_tuple(topic_id, new_headers=new_headers)

                if col is not None:
                    my_df.iloc[index][col] = my_df.iloc[index][col] + 1

    return my_df


if __name__ == '__main__':
    topic_df = read_data(path='../no_filter/TopicCountsOnUsers.txt')

    # fill here with your own tuple list
    topic_tuples = [(1, 25), (0, 49), (7, 240), (49, 0)]
    tuples_dict = {(k, v): 0 for k, v in topic_tuples}
    print(tuples_dict)
    #  header adjustment
    columns = ['username']  # username header
    columns = columns + ['topic' + str(i) for i in range(500)]
    columns = columns + ['topic 501']  # just an empty semi col
    #
    new_df = new_items_dataframe(topic_tuples, old_df=topic_df)

    topic_df = pd.concat([topic_df, new_df], axis=1)
    save_data(path='../no_filter/TCU.txt', df=topic_df)
