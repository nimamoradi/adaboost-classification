import pandas as pd
import numpy as np

concat_char = "_"
split_char = ","


def read_data(path):
    topic_df = pd.read_csv(path, sep=' ')
    return topic_df


def save_data(path, df):
    for index, topic in df['text'].iteritems():
        concatenated_list = split_char.join(topic)
        df['text'].loc[index] = concatenated_list

    df.to_csv(path, index=False, sep=' ')


def look_up_tuple(num1, new_headers):
    for tuple in new_headers:
        li = [n for n in tuple.split(concat_char)]

        if num1 in li:
            return tuple
    return num1


def replace_with_cat(topic_tuples, old_df):
    new_headers = []
    for tuple in topic_tuples:
        li = list(tuple)
        so_li = sorted(li)
        col = str(so_li[0]) + concat_char + str(so_li[1])
        if col not in new_headers:
            new_headers.append(col)

    for index, topic in old_df['text'].iteritems():
        new_vals = topic
        replace_dict = {}
        for h_index, mix_head in enumerate(new_headers):
            li = [n for n in mix_head.split(concat_char)]
            if li[0] in new_vals and li[1] in new_vals:
                replace_dict[li[0]] = mix_head
                replace_dict[li[1]] = mix_head

        for t_index, tweet_topic_number in enumerate(topic):
            if tweet_topic_number in replace_dict:
                new_vals[t_index] = replace_dict[tweet_topic_number]
            else:
                new_vals[t_index] = tweet_topic_number

        old_df['text'].iloc[index] = new_vals
    return old_df


if __name__ == '__main__':
    topic_df = read_data(path='../data_files/no_filter/tweet_topic_per_user.csv')
    for index, item in topic_df.iterrows():
        topic_df.iloc[index] = [item[0], item[1].split(split_char)]
    print(topic_df)

    # fill here with your own tuple list
    topic_tuples = [(23, 64), (0, 49), (7, 240), (49, 0)]

    topic_df = replace_with_cat(topic_tuples, old_df=topic_df)
    print(topic_df)

    save_data(path='../data_files/no_filter/TTPU.txt', df=topic_df)
