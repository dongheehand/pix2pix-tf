import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob



class data_loader():
    
    def __init__(self, conf):
        
        self.in_memory = conf['in_memory']
        self.load_size = conf['load_size']
        self.crop_size = conf['crop_size']
        self.img_path = conf['img_path']
        self.channel = conf['channel']
        self.mode = conf['mode']
        self.batch_size = conf['batch_size']
        self.random_jitter = conf['random_jitter']
        self.mirroring = conf['mirroring']
        
    def build_loader(self):
        
        if self.mode == 'train':
            if self.in_memory:
                self.image_arr = tf.placeholder(shape = [None, self.load_size, 2 * self.load_size, self.channel], dtype = tf.uint8)
            else:
                self.image_arr = self.img_path

            self.tr_dataset = tf.data.Dataset.from_tensor_slices(self.image_arr)

            if not self.in_memory :
                self.tr_dataset = self.tr_dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)

            if self.random_jitter :
                self.tr_dataset = self.tr_dataset.map(self._random_jitter, num_parallel_calls = 4).prefetch(32)
            else:
                self.tr_dataset = self.tr_dataset.map(self._get_half, num_parallel_calls = 4).prefetch(32)

            if self.mirroring:
                self.tr_dataset = self.tr_dataset.map(self._mirroring, num_parallel_calls = 4).prefetch(32)
            self.tr_dataset = self.tr_dataset.shuffle(32)
            self.tr_dataset = self.tr_dataset.repeat()
            self.tr_dataset = self.tr_dataset.batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(self.tr_dataset.output_types, self.tr_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['tr_init'] = iterator.make_initializer(self.tr_dataset)
        
        elif self.mode == 'test':

            if self.in_memory:
                self.image_arr = tf.placeholder(shape = [None, self.load_size, 2 * self.load_size, self.channel], dtype = tf.float32)
            else:
                self.image_arr = self.img_path

            self.val_dataset = tf.data.Dataset.from_tensor_slices(self.image_arr)

            if not self.in_memory :
                self.val_dataset = self.val_dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)

            self.val_dataset = self.val_dataset.map(self._get_half, num_parallel_calls = 4).prefetch(32)
            self.val_dataset = self.val_dataset.batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(self.val_dataset.output_types, self.val_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['val_init'] = iterator.make_initializer(self.val_dataset)
            
    def _parse(self, _image):
        
        image = tf.read_file(_image)
        image = tf.image.decode_png(image)
        image = tf.image.resize_images(image, (self.load_size, 2 * self.load_size), tf.image.ResizeMethod.BICUBIC)
        
        return image
        
    def _random_jitter(self, image_):
        
        shape = tf.shape(image_)
        
        ih = shape[0]
        iw = shape[1] // 2
        
        ix = tf.random_uniform(shape = [1], minval = 0, maxval = iw - self.crop_size + 1, dtype = tf.int32)[0]
        iy = tf.random_uniform(shape = [1], minval = 0, maxval = ih - self.crop_size + 1, dtype = tf.int32)[0]
        
        image_A = image_[iy : iy + self.crop_size, ix : ix + self.crop_size]
        image_B = image_[iy : iy + self.crop_size, ix + self.load_size : ix + self.crop_size + self.load_size]
        
        return image_A, image_B
    
    def _get_half(self, image_):
        
        image_A = image_[:, :self.load_size]
        image_B = image_[:, self.load_size:]
        
        return image_A, image_B
    
    def _mirroring(self, image_A, image_B):
        
        flip_rl = tf.random_uniform(shape = [1], minval = 0, maxval = 3, dtype = tf.int32)[0]
            
        rl = tf.equal(tf.mod(flip_rl, 2),0)
        
        image_A = tf.cond(rl, true_fn = lambda : tf.image.flip_left_right(image_A), false_fn = lambda : (image_A))
        image_B = tf.cond(rl, true_fn = lambda : tf.image.flip_left_right(image_B), false_fn = lambda : (image_B))
        
        return image_A, image_B
    
    
    
    
    

