import numpy as np
import albumentations
import cv2


def transformations(size_y, size_x, p):
    p1, p2, p3 = 0.5 * p[0], 0.5 * p[1], 0.5 * p[2]
    train_transform = albumentations.Compose([
        albumentations.OneOf([albumentations.Resize(251, 301, p=1),
                              albumentations.RandomCrop(size_y, size_x, p=1),
                              albumentations.HorizontalFlip(p=1),
                              albumentations.GridDistortion(p=1),
                              albumentations.ElasticTransform(p=1),
                              albumentations.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT,
                                                              scale_limit=0.3,
                                                              rotate_limit=(10, 30),
                                                              p=1)], p=p1),

        albumentations.OneOf([albumentations.Blur(blur_limit=5, p=1),
                              albumentations.MotionBlur(blur_limit=7, p=1),
                              albumentations.RandomGamma(gamma_limit=(80, 300), p=1),
                              albumentations.MedianBlur(blur_limit=5, p=1),
                              albumentations.RandomBrightnessContrast(brightness_limit=0.2,
                                                                      contrast_limit=0.2, p=1)], p=p2),

        albumentations.OneOf([albumentations.Rotate(limit=90, interpolation=1, border_mode=4, always_apply=False, p=1),
                              albumentations.OpticalDistortion(distort_limit=1.5, shift_limit=1.5, interpolation=1,
                                                               border_mode=4, always_apply=False, p=1),
                              albumentations.RandomGridShuffle(p=1),
                              albumentations.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0,
                                                         always_apply=False, p=1)], p=p3)], p=0.5)
    return train_transform


def my_data_albumentations(images, targets, prop):
    images = np.squeeze(images, axis=0)
    images = images.astype(np.float32)
    targets = np.squeeze(targets, axis=0)
    image_y_dim = images.shape[0]
    image_x_dim = images.shape[1]
    size_y = int(0.7 * image_y_dim)
    size_x = int(0.7 * image_x_dim)
    transform = transformations(size_y, size_x, prop)
    transformed = transform(image=images, mask=targets)
    trans_image = transformed['image']
    trans_target = transformed['mask']

    return np.expand_dims(trans_image, axis=0), np.expand_dims(trans_target, axis=0)


def my_data_albumentations2(images, targets, my_albu, my_prop):
    images = np.squeeze(images, axis=0)
    images = images.astype(np.float32)
    targets = np.squeeze(targets, axis=0)
    image_y_dim = images.shape[0]
    image_x_dim = images.shape[1]
    size_y = int(0.7 * image_y_dim)
    size_x = int(0.7 * image_x_dim)

    if my_albu == 'Blur':
        transformations = albumentations.Blur(blur_limit=5, p=0.2)
    elif my_albu == 'MotionBlur':
        transformations = albumentations.MotionBlur(blur_limit=7, p=0.2)
    elif my_albu == 'RandomGamma':
        transformations = albumentations.RandomGamma(gamma_limit=(80, 300), p=0.2)
    elif my_albu == 'MedianBlur':
        transformations = albumentations.MedianBlur(blur_limit=5, p=0.2)
    elif my_albu == 'RandomBrightnessContrast':
        transformations = albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=my_prop)
    elif my_albu == 'Resize':
        transformations = albumentations.Resize(251, 301, p=0.2)
    elif my_albu == 'RandomCrop':
        transformations = albumentations.RandomCrop(size_y, size_x, p=0.2)
    elif my_albu == 'HorizontalFlip':
        transformations = albumentations.HorizontalFlip(p=0.2)
    elif my_albu == 'GridDistortion':
        transformations = albumentations.GridDistortion(p=0.2)
    elif my_albu == 'ElasticTransform':
        transformations = albumentations.ElasticTransform(p=0.2)
    elif my_albu == 'ShiftScaleRotate':
        transformations = albumentations.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT,
                                                          scale_limit=0.3,
                                                          rotate_limit=(10, 30),
                                                          p=0.2)
    elif my_albu == 'Rotate':
        transformations = albumentations.Rotate(limit=90, interpolation=1, border_mode=4, always_apply=False, p=0.2)
    elif my_albu == 'OpticalDistortion':
        transformations = albumentations.OpticalDistortion(distort_limit=1.5, shift_limit=1.5, interpolation=1,
                                                           border_mode=4, always_apply=False, p=0.2)
    elif my_albu == 'RandomGridShuffle':
        transformations = albumentations.RandomGridShuffle(p=0.2)
    elif my_albu == 'MaskDropout':
        transformations = albumentations.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0,
                                                     always_apply=False, p=0.2)

    elif my_albu == 'Transpose':
        transformations = albumentations.Transpose(p=my_prop)
    elif my_albu == 'RandomRotate90':
        transformations = albumentations.RandomRotate90(p=my_prop)
    elif my_albu == 'VerticalFlip':
        transformations = albumentations.VerticalFlip(p=my_prop)
    elif my_albu == 'CenterCrop':
        transformations = albumentations.CenterCrop(p=my_prop, height=image_y_dim, width=image_x_dim)
    elif my_albu == 'RandomSizedCrop':
        transformations = albumentations.RandomSizedCrop(min_max_height=(50, 150), height=image_y_dim, width=image_x_dim, p=my_prop)

    transformed = transformations(image=images, mask=targets)
    trans_image = transformed['image']
    trans_target = transformed['mask']

    return np.expand_dims(trans_image, axis=0), np.expand_dims(trans_target, axis=0)


def my_data_albumentations3(images, targets, my_albu1, my_albu2, my_prop):
    images = np.squeeze(images, axis=0)
    images = images.astype(np.float32)
    targets = np.squeeze(targets, axis=0)
    my_transformations = None
    if my_albu1 == 'RandomGamma' and my_albu2 == 'RandomBrightnessContrast':
        my_transformations = albumentations.Compose([
            albumentations.RandomGamma(gamma_limit=(80, 300), p=my_prop),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=my_prop), ])

    transformed = my_transformations(image=images, mask=targets)
    trans_image = transformed['image']
    trans_target = transformed['mask']

    return np.expand_dims(trans_image, axis=0), np.expand_dims(trans_target, axis=0)
