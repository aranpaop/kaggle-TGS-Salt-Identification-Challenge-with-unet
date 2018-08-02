# data augmentation and train batch generation
import cv2
import random
import numpy as np


def data_aug(X_img, y_img):
    act = random.randint(0, 5)
    # horizonal flip
    if act == 1:
        X_img = cv2.flip(X_img, 1)
        y_img = cv2.flip(y_img, 1)
    # veritical flip
    elif act == 2:
        X_img = cv2.flip(X_img, 0)
        y_img = cv2.flip(y_img, 0)
    # rotate 90 degree
    elif act == 3:
        X_img = cv2.transpose(X_img, (1, 0))
        X_img = cv2.flip(X_img, 1)
        y_img = cv2.transpose(y_img, (1, 0))
        y_img = cv2.flip(y_img, 1)
    # rotate 180 degree
    elif act == 4:
        X_img = cv2.flip(X_img, -1)
        y_img = cv2.flip(y_img, -1)
    # rotate 270 degree
    elif act == 5:
        X_img = cv2.transpose(X_img, (1, 0))
        X_img = cv2.flip(X_img, 0)
        y_img = cv2.transpose(y_img, (1, 0))
        y_img = cv2.flip(y_img, 0)
    return X_img, y_img


def batch_gen(X_train, y_train, batch_size):
    batch_size = batch_size % 400
    part = random.randint(0, 9)
    X_aug = X_train[part*400:part*400+400]
    y_aug = y_train[part*400:part*400+400]
    aug_list = list(zip(X_aug, y_aug))
    random.shuffle(aug_list)
    X_aug[:], y_aug[:] = zip(*aug_list)
    for i in range(batch_size):
        X_aug[i], y_aug[i] = data_aug(X_aug[i], y_aug[i])
    return np.array(X_aug[:batch_size])[:, :, :, np.newaxis], np.array(y_aug[:batch_size])[:, :, :, np.newaxis]