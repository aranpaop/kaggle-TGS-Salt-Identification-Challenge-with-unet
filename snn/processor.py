import numpy as np
import tensorflow as tf
from keras import backend as K


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def calc_score(y_true, y_pred, threshold):
    scores = 0
    for k in range(y_pred.shape[0]):
        pre = y_pred[k, :, :, :]
        gt = y_true[k, :, :, :]
        pre = pre > threshold
        TP = np.sum(pre + gt == 2)
        FP = np.sum(((pre == 0) + gt) == 2)
        FN = np.sum((pre + (gt == 0)) == 2)
        if TP + FP + FN == 0:
            scores += 1
        else:
            scores += TP / (TP + FP + FN)
    return scores / y_pred.shape[0]


def pic2step(img, threshold):
    bytes = img.reshape(img.shape[0]*img.shape[1], order='F')
    bytes = bytes > threshold
    runs = ''
    l = bytes.shape[0]
    i = 0
    while i < l:
        if bytes[i] == 0:
            i += 1
            continue
        elif bytes[i] == 1:
            for step in range(l-i):
                if i + step == l - 1 and bytes[i+step] == 1:
                    runs = runs + str(i+1) + ' '
                    runs = runs + str(step+1) + ' '
                    i = l
                elif bytes[i+step] == 0:
                    runs = runs + str(i + 1) + ' '
                    runs = runs + str(step) + ' '
                    i += step + 1
                    break
    return runs[:-1]