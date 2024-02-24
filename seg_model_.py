## IMPORTS ##

import matplotlib.pyplot as plt
import config
# import visualkeras
import keras
import sklearn
import tensorflow as tf
import os
import cv2
import numpy as np
from math import ceil
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from segmentation_models.metrics import iou_score, IOUScore, FScore
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
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


## DATASET ##

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

# #TRAIN - ordenação dos nomes dos arquivos
# train_images_list.sort()
# train_masks_list.sort()
# #VAL
# val_images_list.sort()
# val_masks_list.sort()
# #VAL
# test_images_list.sort()
# test_masks_list.sort()


##PRÉ-PROCESSAMENTO##
#Criando um generator
def image_mask_generator(images_list, images_path, masks_path, batch_size= config.BATCH_SIZE, output_size=config.IMAGE_SIZE):
    num_samples = len(images_list)
    while True:
        #Embaralhar os índices para garantir que os dados sejam apresentados de forma aleatória
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
                mask = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED)
                
                if img is None or mask is None:
                    print(f"Falha ao carregar imagem ou máscara: {images_list[idx]}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, output_size)
                mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
                
                img = img.astype(np.float32) / 255.0
                mask = tf.one_hot(mask, 3) #Codificando aa máscaras em one_hot_encoding

                batch_images.append(img)
                batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)

gerador_treino = image_mask_generator(train_images_list, train_images_path, train_masks_path)
gerador_validacao = image_mask_generator(val_images_list, val_images_path, val_masks_path)


## Instancando o modelo##
#Compilando o modelo usando a função de perda

model = Unet(input_shape= (224,224,3), classes=3, backbone_name='efficientnetb1', encoder_weights='imagenet', encoder_freeze=True, activation='softmax')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= config.LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics= [IOUScore(class_indexes=1, threshold=0.5, name='iou_disco'), IOUScore(class_indexes=2, threshold=0.5, name='iou_cup'), FScore(class_indexes=1, threshold=0.5, name='dice_disco'), FScore(class_indexes=2, threshold=0.5, name='dice_cup')  ])

##Treinamento##
callbacks_list = [ModelCheckpoint(f"/home/arthur_guilherme/pibic_mack-24/checkpoint/best_model_weights.h5", monitor = 'val_iou_cup', verbose = 1, save_best_only = True, mode='max')]
#[EarlyStopping(monitor = 'val_iou_cup', patience = 25, mode = 'auto', verbose = 1),
model.fit(gerador_treino,
           validation_data = gerador_validacao,
           epochs= config.EPOCHS,
           callbacks = callbacks_list,
           steps_per_epoch= ceil(len(train_images_list)/config.BATCH_SIZE),
           validation_steps= ceil(len(val_images_list)/config.BATCH_SIZE),
           )



