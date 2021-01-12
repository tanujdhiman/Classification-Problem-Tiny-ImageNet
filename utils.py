import os
import zipfile
import urllib.request as url
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import imgaug as ia
from imgaug import augmenters as iaa 
             
def load_data(path, img_width=64, img_height=64, batch_size=128, augmentation=None, seed=None, load_test=False):
    print('Begin loading...')
    train_path = path+'/tiny-imagenet-200/train' #path to the train folder
    val_path = path+'/tiny-imagenet-200/val' #path to the val folder
    
    # load and rescale training data
    # if augmentation=None, no data augmentation
    # split 20% training data as validation set 
    train_datagen = ImageDataGenerator(preprocessing_function=augmentation, rescale=1./255, validation_split=0.2) 
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=(img_width, img_height),
                                                        class_mode='categorical',   #it's a classification problem so we use categorical
                                                        batch_size=batch_size,
                                                        subset='training',
                                                        shuffle=True, seed=seed)     # shuffle the data by shuffling 
    val_generator = train_datagen.flow_from_directory(train_path,
                                                      target_size=(img_width, img_height),
                                                      class_mode='categorical',  
                                                      batch_size=batch_size,
                                                      subset='validation',
                                                      shuffle=True, seed=seed)   
    print('Training data shape: {}'.format((train_generator.n,)+train_generator.next()[0].shape[1:]))
    print('Validation data shape: {}'.format((val_generator.n,)+val_generator.next()[0].shape[1:]))
    
    if not load_test:
        # don't load test data
        train_generator.reset()
        val_generator.reset()
        print('End loading!')
        return train_generator, val_generator
    else:
        test_df = pd.read_csv(val_path+'/val_annotations.txt', 
                              sep='\t', header=None, names=['filename','class','x','y','width','height'], 
                              usecols=['filename','class'])
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_dataframe(test_df,
                                                          directory=val_path+'/images/',
                                                          x_col=test_df.columns[0],
                                                          y_col=test_df.columns[1],
                                                          target_size=(img_width, img_height),
                                                          class_mode='categorical',
                                                          batch_size=batch_size,
                                                          shuffle=True, seed=seed)
        print('Testing data shape: {}'.format((test_generator.n,)+test_generator.next()[0].shape[1:]))

        train_generator.reset()
        val_generator.reset()
        test_generator.reset()
        print('End loading!')
        return train_generator, val_generator, test_generator
      
def data_augmentation(complicated=False,
                      CoarseDropout_range=(0.0, 0.05),
                      CoarseDropout_size_percent=(0.02, 0.25),
                      Affine_translate_percent=(-0.2, 0.2),
                      Affine_scale=(0.5, 1.5),
                      Affine_shear=(-20, 20),
                      Affine_rotate=(-45, 45),
                      Flip_percent=0.5,
                      GaussianBlur_sigma=(0.0, 3.0),
                      CropAndPad_percent=(-0.25, 0.25),
                      Multiply=(0.5, 1.5),
                      LinearContrast=(0.4, 1.6),
                      AdditiveGaussianNoise_scale=0.2*255):
    ia.seed(123)
    
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    if complicated:
        # augmentation for network 2
        aug = iaa.Sequential([
                             # apply to all images
                             iaa.Fliplr(Flip_percent),   # Flip/mirror horizontally
                             iaa.Flipud(Flip_percent),   # Flip/mirror vertically
                             
                             # apply to half of images
                             # sets rectangular areas within images to zero
                             sometimes(iaa.CoarseDropout(CoarseDropout_range, 
                                                         size_percent=CoarseDropout_size_percent)), 
                             # Apply affine transformations to images
                             sometimes(iaa.Affine(translate_percent=Affine_translate_percent,
                                                  scale=Affine_scale,
                                                  shear=Affine_shear,
                                                  rotate=Affine_rotate
                                                 )),  
                             # blur images using gaussian kernels
                             sometimes(iaa.GaussianBlur(sigma=GaussianBlur_sigma)),
                             # crop and pad images, and resize images back to their original size
                             sometimes(iaa.CropAndPad(percent=CropAndPad_percent)), 
                             # Multiply all pixels with a specific value
                             # thereby making the image darker or brighter
                             sometimes(iaa.Multiply(Multiply)), 
                             # 127 + alpha*(v-127)`, where v is a pixel value
                             # alpha is sampled uniformly from the interval
                             sometimes(iaa.LinearContrast(LinearContrast)) 
                         ], random_order=True)
    else:
        # augmentation for network 1
        aug = iaa.SomeOf((0, None),
                         [  # sets rectangular areas within images to zero
                             iaa.CoarseDropout(CoarseDropout_range, 
                                               size_percent=CoarseDropout_size_percent), 
                             # Apply affine rotation on the y-axis to input data
                             iaa.Affine(scale=Affine_scale,
                                        rotate=Affine_rotate
                                       ),  
                             # crop and pad images, and resize images back to their original size
                             iaa.CropAndPad(percent=CropAndPad_percent, keep_size=True),  
                             # Add gaussian noise to an image
                             iaa.AdditiveGaussianNoise(scale=AdditiveGaussianNoise_scale) 
                         ], random_order=True)
    return aug.augment_image

def get_lookup_tables(path, generator):
    # generate [class_id, class_name] lookup table
    lookup_des = pd.read_csv(path+'/tiny-imagenet-200/words.txt', 
                           sep='\t', header=None, names=['class_id','class_name'])
    lookup_des['class_name'] = lookup_des['class_name'].str.split(',').str[0]
    # generate [index, class_id] lookup table. Index is the index in train_generator
    lookup_id = pd.DataFrame(([y,x] for x,y in generator.class_indices.items()),columns=['indexes','class_id'])
    # generate {index: class_name} lookup dict
    lookup = lookup_des.merge(lookup_id, on='class_id', how='right').set_index('indexes')['class_name'].to_dict()
    return lookup

def get_labels(path, indexes, generator):
    lookup = get_lookup_tables(path, generator)
    labels = [lookup[i] for i in np.where(indexes == 1)[1]]
    return labels

  
def load_history(checkpoint_filepath, history_ls):
    if len(history_ls) == 0:
        # if no log file
        return(print('No log file is founded'))
    
    history = pd.read_csv(checkpoint_filepath+'/'+history_ls[0], sep=',')
    if len(history_ls) > 1:
        # concate all log files into one dataframe
        for name in history_ls[1:]:
            history = pd.concat([history, pd.read_csv(checkpoint_filepath+'/'+name, sep=',')])
    # renumber epochs
    history.reset_index(drop=True,inplace=True)
    history['epochs'] = history.index.to_list()
    return history
