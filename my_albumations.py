import numpy as np
import albumentations
import cv2


def trasformations(size_y, size_x, p):
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
    transform = trasformations(size_y, size_x, prop)
    transformed = transform(image=images, mask=targets)
    trans_image = transformed['image']
    trans_target = transformed['mask']

    return np.expand_dims(trans_image, axis=0), np.expand_dims(trans_target, axis=0)


