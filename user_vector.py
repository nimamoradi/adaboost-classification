import math
import numpy as np
import pandas as pd

threshold = 0.02


def filter_rows(u_vector: pd.Series):
    col_names = ['label', 'text']
    new_user_vector = pd.DataFrame(columns=col_names)

    for index, user in u_vector['text'].iteritems():
        user = [int(i) for i in user.split(',')]
        users_sum = sum(user, 0)
        for inx, topic_weight in enumerate(user):

            if (topic_weight / users_sum) < threshold:
                user[inx] = 0

        new_user_vector.loc[index] = [u_vector['label'].loc[index], np.array(user)]
    return new_user_vector


def field_calculator(sum_weight, tweet_count, whole_users_count, this_top_users):
    topic_ratio_user = (1.0 * tweet_count) / sum_weight
    user_ratio_whole = (1.0 * this_top_users) / whole_users_count

    return math.log2((1 + topic_ratio_user) / (1 + user_ratio_whole)) + 1
    # return (topic_ratio_user) / math.log2((user_ratio_whole))


def calculate_df(tw_df: pd.DataFrame, tweet_number):
    exist_arr = 1 * (tw_df['text'].loc[tweet_number] > 0)
    return np.sum(np.array(exist_arr))


def find_tf_idf_weights(tw_df: pd.DataFrame):
    col_names = ['label', 'weights']
    new_user_vector = pd.DataFrame(columns=col_names)
    users_count = len(tw_df.index)

    for index, user in tw_df['text'].iteritems():
        users_sum = np.sum(user)
        weights = [0.0] * len(user)
        if users_sum == 0:
            print(tw_df.loc[index])
            exit(-2)
        for inx, topic_weight in enumerate(user):
            weights[inx] = field_calculator(users_sum, topic_weight, users_count, calculate_df(tw_df, inx))

        new_user_vector.loc[index] = [tw_df['label'].loc[index], weights]

    return new_user_vector


if __name__ == '__main__':
    size = 502

    tweet_df = pd.read_csv("./TopicCountsOnUsers.txt", sep=' ')

    user_vector = filter_rows(tweet_df)
    user_vector = find_tf_idf_weights(user_vector)

    import pickle

    with open('model.pickle', 'wb') as handle:
        pickle.dump(user_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
