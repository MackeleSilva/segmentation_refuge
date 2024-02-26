""" IMPORTS """

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Input, Concatenate, GlobalAveragePooling2D, Reshape, Dense, Multiply
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.models import Model
import os
import cv2
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import config
import numpy as np
from segmentation_models.metrics import iou_score, IOUScore, FScore
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from tensorflow.keras.regularizers import l1, l2
from math import ceil
from tensorflow.keras.metrics import MeanIoU
# from tensorflow.keras.metrics import DiceCoefficient
# TF CONFIGURATION
print("Número de GPUs disponíveis: ", len(
    tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU') #força usar a 0
        tf.config.experimental.set_memory_growth(gpus[0], True) #so usa memoria a medida q precisa
    except RuntimeError as e:
        print(e)


""" DATASET """

#TRAIN - caminho das pastas
train_images_path = "Dataset/train/Images_Cropped"
train_masks_path = "Dataset/train/Masks_Cropped"
#VAL
val_images_path = "Dataset/val/Images_Cropped"
val_masks_path = "Dataset/val/Masks_Cropped"
#TEST
# test_images_path = "Dataset/test/Images_Cropped"
# test_masks_path = "Dataset/test/Masks_Cropped"

#TRAIN - lista com todos os nomes dos arquivos de imagem
train_images_list = os.listdir(train_images_path)
train_masks_list = os.listdir(train_masks_path)
#VAL
val_images_list = os.listdir(val_images_path)
val_masks_list = os.listdir(val_masks_path)
#TEST
# test_images_list = os.listdir(test_images_path)
# test_masks_list = os.listdir(test_masks_path)

#TRAIN - ordenação dos nomes dos arquivos
train_images_list.sort()
train_masks_list.sort()
#VAL
val_images_list.sort()
val_masks_list.sort()
#VAL
# test_images_list.sort()
# test_masks_list.sort()

def image_mask_generator(images_list, images_path, masks_path, batch_size= config.BATCH_SIZE, output_size=config.IMAGE_SIZE):
    num_samples = len(images_list)
    while True:
        # Embaralhar os índices para garantir que os dados sejam apresentados de forma aleatória
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_images = []
            
            batch_masks = []
            for idx in batch_indices:
                img_filename = os.path.join(images_path, images_list[idx])
                mask_filename = os.path.join(masks_path, images_list[idx][:-4] + ".png")

                if not os.path.isfile(img_filename) or not os.path.isfile(mask_filename):
                    print(f"Arquivo de imagem ou máscara não encontrado: {images_list[idx]}")
                    continue

                img = cv2.imread(img_filename)
                mask = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_GRAYSCALE
                
                if img is None or mask is None:
                    print(f"Falha ao carregar imagem ou máscara: {images_list[idx]}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, output_size)
                mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
                
                img = img.astype(np.float32) / 255.0
                mask = tf.one_hot(mask, 3)

                batch_images.append(img)
                batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)

#TRAIN
gerador_treino = image_mask_generator(train_images_list, train_images_path, train_masks_path)
#VAL
gerador_validacao = image_mask_generator(val_images_list, val_images_path, val_masks_path)

""" MODELO """

""" Função Squeeze Excitation """

def SqueezeAndExcitation(inputs, ratio=8):
    b, h, w, c = inputs.shape

    ## Squeeze
    x = GlobalAveragePooling2D()(inputs)

    ## Excitation
    x = Dense(c//ratio, activation='relu', use_bias=False)(x)
    x = Dense(c, activation='sigmoid', use_bias=False)(x)

    ## Ensure x has dimensions (batch_size, 1, 1, channels)
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)

    ## Scaling
    x = inputs * x

    return x


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same', kernel_regularizer=l1(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same', kernel_regularizer=l2(0.01))(inputs)
    # Applying SE block
    x = SqueezeAndExcitation(x)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_efficient_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained Encoder """
    encoder = EfficientNetB2(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = encoder.get_layer[1].output                          ## 256
    s2 = encoder.get_layer("block2a_expand_activation").output       ## 128
    s3 = encoder.get_layer("block3a_expand_activation").output        ## 64
    s4 = encoder.get_layer("block4a_expand_activation").output        ## 32

    """ Bottleneck """
    b1 = encoder.get_layer("block6a_expand_activation").output        ## 16

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                                    ## 32
    d2 = decoder_block(d1, s3, 256)                                    ## 64
    d3 = decoder_block(d2, s2, 128)                                    ## 128
    d4 = decoder_block(d3, s1, 64)                                     ## 265

    """ Output """
    outputs = Conv2D(3, 1, padding="same", activation='softmax')(d4)

    model = Model(inputs, outputs, name="EfficientNetB2_UNET")
    
    return model

# Definindo a métrica MeanIoU para cada classe
# iou_metric = MeanIoU(num_classes=3)

def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
# Definindo a métrica Dice
def dice_metric(y_true, y_pred):
    return dice_coefficient(y_true, y_pred)

modelo = build_efficient_unet(input_shape=(224,224,3))
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= config.LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.OneHotMeanIoU(num_classes=3), dice_metric])


#[IOUScore(class_indexes=1, threshold=0.5, name='iou_disco'), IOUScore(class_indexes=2, threshold=0.5, name='iou_cup'), FScore(class_indexes=1, threshold=0.5, name='dice_disco'), FScore(class_indexes=2, threshold=0.5, name='dice_cup')])

# pretrain model decoder
callbacks_list = [EarlyStopping(monitor = 'val_one_hot_mean_io_u', patience = 25, mode = 'auto', verbose = 1), ModelCheckpoint(f"/home/arthur_guilherme/pibic_mack-24/segmentation_refuge/checkpoint/best_model_weights.h5", monitor = 'val_one_hot_mean_io_u', verbose = 1, save_best_only = True,save_weights_only = True, mode= 'max', initial_value_threshold=0.7)]

modelo.fit(gerador_treino,
           validation_data = gerador_validacao,
           epochs= config.EPOCHS,
           callbacks = callbacks_list,
           steps_per_epoch= ceil(len(train_images_list)/config.BATCH_SIZE),
           validation_steps= ceil(len(val_images_list)/config.BATCH_SIZE),
           )