import os
import numpy as np
import torch
import albumentations
import cv2


def load_my_image_batch(batch_nro, train_dict, my_path, train_batch_size, normalize):
    #print('train_batch_koko', train_batch_size)
    patient = train_dict[batch_nro][0]
    patient_images_folder = patient + '_images'
    image_folder = os.path.join(my_path, patient_images_folder)

    my_iterable = train_dict[batch_nro][1]
    for i, j in enumerate(my_iterable):
        #print('j = ', j)
        file = 'slice_' + str(j) + '.npy'
        # file = 'slice_' + str(128) + '.npy'

        images_path = os.path.join(image_folder, file)
        image = np.load(images_path)
        image = image.astype(np.float32)
        #print(image.shape)
        if normalize == '[-1,1]':
            np.clip(image, 500, 3500, out=image)
            image = image - 2000
            image = image / 1500
            # maxim, minim = np.max(image), np.min(image)
            # print('max, min image is =', maxim, minim)
            # exit()
        elif normalize == '[0,1]':
            np.clip(image, 500, 3500, out=image)
            image = image - 500
            image = image / 3000

        elif normalize == 'my_shift_and_[0,1]':
            np.clip(image, 600, 4000, out=image)
            if patient == 'andy':
                image = image + 94
            elif patient == 'teeth1':
                image = image - 162
            elif patient == 'teeth2':
                image = image + 41
            elif patient == 'patient1':
                image = image - 115
            elif patient == 'timo':
                image = image + 132

            np.clip(image, 800, 3800, out=image)
            image = image - 800
            image = image / 3000
        elif normalize == 'norm':
            s = np.std(image)
            mean = np.mean(image)
            image = (image - mean) / s
        if i == 0:
            # print('train_batch_size', train_batch_size)
            images = np.zeros((train_batch_size, image.shape[0], image.shape[1]))
            # print('image_shape=', images.shape)
        #print('my i=', i)
        images[i] = image
    #print(images.shape)
    return images

def load_my_target_batch(batch_nro, train_dict, my_path, train_batch_size):
    patient = train_dict[batch_nro][0]
    patient_targets_folder = patient + '_targets'
    target_folder = os.path.join(my_path, patient_targets_folder)

    my_iterable = train_dict[batch_nro][1]
    for i, j in enumerate(my_iterable):

        file = 'slice_' + str(j) + '.npy'
        # file = 'slice_' + str(128) + '.npy'
        target_path = os.path.join(target_folder, file)
        # print(target_path)
        target = np.load(target_path)
        if i == 0:
            targets = np.zeros((train_batch_size, target.shape[0], target.shape[1]))
        targets[i] = target
    return targets


