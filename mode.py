
# coding: utf-8

# In[1]:


import glob
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from model import pix2pix


# In[ ]:


def train(conf):

    img_path = glob.glob(os.path.join(conf['tr_data_path'], '*.%s'%(conf['data_ext'])))
    conf['img_path'] = img_path
    
    model = pix2pix(conf)
    model.build_graph()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep = None)
    step = len(conf['img_path']) // conf['batch_size']
    
    if conf['in_memory']:
        img_list = []
        for img in conf['img_path']:
            img_ = Image.open(img)
            img_ = img_.resize((2*conf['load_size'], conf['load_size']), Image.BICUBIC)
            img_ = np.array(img_)
            img_list.append(img_)
        img_list = np.asarray(img_list)
        
        sess.run(model.data_loader.init_op['tr_init'], feed_dict = {model.data_loader.image_arr : img_list})
        
            
    else:
        sess.run(model.data_loader.init_op['tr_init'])
        
    
    for i in range(conf['epoch']):
        for t in range(step):
            _, d_loss = sess.run([model.discriminator_train, model.discriminator_loss])
            _, gene_image, g_loss = sess.run([model.generator_train, model.fake_image, model.generator_loss])
            print("%02d _ %05d"%(i, t))
            print('d_loss : %0.6f, g_loss : %0.6f' % (d_loss, g_loss))

        gene_image = gene_image[0]
        gene_image = ((gene_image + 1.0) * 255.0) / 2.0
        gene_image = np.round(gene_image)
        gene_image = np.clip(gene_image, 0.0, 255.0)
        gene_image = gene_image.astype(np.uint8)
        gene_image = Image.fromarray(gene_image)
        gene_image.save('temp/%s_%04d.png'%(conf['model_name'], i))
        
    saver.save(sess, os.path.join('./model/pix2pix_%s'%conf['model_name']))
                
def test(conf):
    
    img_path = glob.glob(os.path.join(conf['val_data_path'], '*.%s'%(conf['data_ext'])))
    conf['img_path'] = sorted(img_path)
    
    model = pix2pix(conf)
    model.build_graph()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep = None)
    saver.restore(sess, conf['pre_trained_model'])
    step = len(conf['img_path']) // conf['batch_size']
    
    if conf['in_memory']:
        img_list = []
        for img in conf['img_path']:
            img_ = Image.open(img)
            img_ = img_.resize((2*conf['load_size'], conf['load_size']), Image.BICUBIC)
            img_ = np.array(img_)
            img_list.append(img_)
        img_list = np.asarray(img_list)
        
        sess.run(model.data_loader.init_op['val_init'], feed_dict = {model.data_loader.image_arr : img_list})
        
    else:
        sess.run(model.data_loader.init_op['val_init'])
    
    if not os.path.exists('result_%s'%(conf['model_name'])):
        os.makedirs('result_%s'%(conf['model_name']))
    
    for t in range(step):

        gene_image, img_A, GT = sess.run([model.fake_image, model.image_A, model.image_B])

        gene_image = gene_image[0]
        gene_image = ((gene_image + 1.0) * 255.0) / 2.0
        gene_image = np.round(gene_image)
        gene_image = np.clip(gene_image, 0.0, 255.0)
        gene_image = gene_image.astype(np.uint8)
        gene_image = Image.fromarray(gene_image)
        gene_image.save('result_%s/%s_gene.png'%(conf['model_name'], conf['img_path'][t].split('/')[-1].split('.')[0]))


        img_A = img_A[0]
        img_A = ((img_A + 1.0) * 255.0) / 2.0
        img_A = np.round(img_A)
        img_A = np.clip(img_A, 0.0, 255.0)
        img_A = img_A.astype(np.uint8)
        img_A = Image.fromarray(img_A)
        img_A.save('result_%s/%s_input.png'%(conf['model_name'], conf['img_path'][t].split('/')[-1].split('.')[0]))


        GT = GT[0]
        GT = ((GT + 1.0) * 255.0) / 2.0
        GT = np.round(GT)
        GT = np.clip(GT, 0.0, 255.0)
        GT = GT.astype(np.uint8)
        GT = Image.fromarray(GT)
        GT.save('result_%s/%s_GT.png'%(conf['model_name'], conf['img_path'][t].split('/')[-1].split('.')[0]))

