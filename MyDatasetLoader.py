
import numpy as np


def my_data_loader(data_set, test_train_split, data_shuffle, batch_size):
    def give_batch_matrix(z_shape):
        batch_count = int(np.floor(z_shape / batch_size))
        numpy_cut = batch_count * batch_size
        if data_shuffle:
            ss = np.random.permutation(numpy_cut)
            andy_chosen = ss[:numpy_cut]

        else:
            andy_chosen = np.arange(numpy_cut)

        return andy_chosen.reshape((batch_count, batch_size)), batch_count
    if data_set == 'data':
        arr_andy, batch_count_andy = give_batch_matrix(170)
        arr_teeth1, batch_count_teeth1 = give_batch_matrix(234)
        arr_teeth2, batch_count_teeth2 = give_batch_matrix(248)
        arr_patient1, batch_count_patient1 = give_batch_matrix(534)
        arr_timo, batch_count_timo = give_batch_matrix(170)
    else:
        arr_andy, batch_count_andy = give_batch_matrix(150)
        arr_teeth1, batch_count_teeth1 = give_batch_matrix(180)
        arr_teeth2, batch_count_teeth2 = give_batch_matrix(170)
        arr_patient1, batch_count_patient1 = give_batch_matrix(380)
        arr_timo, batch_count_timo = give_batch_matrix(160)

    my_dict = {}

    for j in range(batch_count_andy):
        my_dict[j] = ('andy', arr_andy[j])

    for j in range(batch_count_teeth1):
        s = batch_count_andy + j
        my_dict[s] = ('teeth1', arr_teeth1[j])

    for j in range(batch_count_teeth2):
        s = batch_count_andy + batch_count_teeth1 + j
        my_dict[s] = ('teeth2', arr_teeth2[j])

    for j in range(batch_count_patient1):
        s = batch_count_andy + batch_count_teeth1 + batch_count_teeth2 + j
        my_dict[s] = ('patient1', arr_patient1[j])

    for j in range(batch_count_timo):
        s = batch_count_andy + batch_count_teeth1 + batch_count_teeth2 + batch_count_patient1 + j
        my_dict[s] = ('timo', arr_timo[j])

    all_batches_count = batch_count_andy + batch_count_teeth1 + batch_count_teeth2 + batch_count_patient1 + batch_count_timo

    if data_shuffle:
        batch_list = np.random.permutation(all_batches_count)
        batch_list_length = batch_list.__len__()
        test_batch_list_length = int(batch_list_length / test_train_split)
        test_batch_list = batch_list[:test_batch_list_length]
        train_batch_list = batch_list[test_batch_list_length:]
    else:
        my_batches = list(range(all_batches_count))
        batch_list = np.random.permutation(all_batches_count)
        batch_list_length = batch_list.__len__()
        test_batch_list_length = int(batch_list_length / test_train_split)
        test_batch_list = batch_list[:test_batch_list_length]
        train_batch_list = [s for s in my_batches if s not in test_batch_list]

    return train_batch_list, test_batch_list, my_dict

