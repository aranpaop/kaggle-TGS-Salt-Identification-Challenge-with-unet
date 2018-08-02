from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate
from keras import Model
from keras.optimizers import Adam
from snn import processor


def get_unet():
    input = Input(shape=(128, 128, 1))

    c1 = Conv2D(8, (3, 3), padding='same')(input)
    c1 = BatchNormalization(axis=3)(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(8, (3, 3), padding='same')(c1)
    c1 = BatchNormalization(axis=3)(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), padding='same')(p1)
    c2 = BatchNormalization(axis=3)(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(16, (3, 3), padding='same')(c2)
    c2 = BatchNormalization(axis=3)(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), padding='same')(p2)
    c3 = BatchNormalization(axis=3)(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(32, (3, 3), padding='same')(c3)
    c3 = BatchNormalization(axis=3)(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), padding='same')(p3)
    c4 = BatchNormalization(axis=3)(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(64, (3, 3), padding='same')(c4)
    c4 = BatchNormalization(axis=3)(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(128, (3, 3), padding='same')(p4)
    c5 = BatchNormalization(axis=3)(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(128, (3, 3), padding='same')(c5)
    c5 = BatchNormalization(axis=3)(c5)
    c5 = Activation('relu')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = BatchNormalization(axis=3)(u6)
    u6 = Concatenate(axis=3)([u6, c4])
    c6 = Conv2D(64, (3, 3), padding='same')(u6)
    c6 = BatchNormalization(axis=3)(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(64, (3, 3), padding='same')(c6)
    c6 = BatchNormalization(axis=3)(c6)
    c6 = Activation('relu')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = BatchNormalization(axis=3)(u7)
    u7 = Concatenate(axis=3)([u7, c3])
    c7 = Conv2D(32, (3, 3), padding='same')(u7)
    c7 = BatchNormalization(axis=3)(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(32, (3, 3), padding='same')(c7)
    c7 = BatchNormalization(axis=3)(c7)
    c7 = Activation('relu')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = BatchNormalization(axis=3)(u8)
    u8 = Concatenate(axis=3)([u8, c2])
    c8 = Conv2D(16, (3, 3), padding='same')(u8)
    c8 = BatchNormalization(axis=3)(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(16, (3, 3), padding='same')(c8)
    c8 = BatchNormalization(axis=3)(c8)
    c8 = Activation('relu')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = BatchNormalization(axis=3)(u9)
    u9 = Concatenate(axis=3)([u9, c1])
    c9 = Conv2D(8, (3, 3), padding='same')(u9)
    c9 = BatchNormalization(axis=3)(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(8, (3, 3), padding='same')(c9)
    c9 = BatchNormalization(axis=3)(c9)
    c9 = Activation('relu')(c9)

    output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input], outputs=[output])
    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=[processor.mean_iou])

    return model