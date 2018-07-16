import ops
import tensorflow as tf
from data_loader import data_loader


class pix2pix():
    
    def __init__(self, conf):
        
        self.data_loader = data_loader(conf)
        self.channel = conf['channel']
        self.mode = conf['mode']
        self.transfer_type = conf['transfer_type']
        self.learning_rate = conf['learning_rate']
        self.beta1 = conf['beta1']
        self.beta2 = conf['beta2']
        self.eps = conf['eps']
        
    def generator(self, x, name = 'generator', reuse = False):
        
        with tf.variable_scope(name_or_scope = name, reuse = reuse):
            encoder_tensor = []
            
            # Encoder
            x = ops._conv('conv1', x, 4, self.channel, 64, 2, batch_norm = False)
            encoder_tensor.append(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv2', x, 4, 64, 128)
            encoder_tensor.append(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv3', x, 4, 128, 256)
            encoder_tensor.append(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv4', x, 4, 256, 512)
            encoder_tensor.append(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv5', x, 4, 512, 512)
            encoder_tensor.append(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv6', x, 4, 512, 512)
            encoder_tensor.append(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv7', x, 4, 512, 512)
            encoder_tensor.append(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv8', x, 4, 512, 512, 2, batch_norm = False)
            
            i = 0
            
            # Decoder
            
            x = tf.nn.relu(x)
            
            x = ops._deconv('deconv1', x, 4, 512, 512, 2, True)
            x = tf.concat([x, encoder_tensor[-1-i]], axis = 3)
            x = tf.nn.relu(x)
            i += 1
            
            x = ops._deconv('deconv2', x, 4, 1024, 512, 2, True)
            x = tf.concat([x, encoder_tensor[-1-i]], axis = 3)
            x = tf.nn.relu(x)
            i += 1
            
            x = ops._deconv('deconv3', x, 4, 1024, 512, 2, True)
            x = tf.concat([x, encoder_tensor[-1-i]], axis = 3)
            x = tf.nn.relu(x)
            i += 1
            
            x = ops._deconv('deconv4', x, 4, 1024, 512, 2)
            x = tf.concat([x, encoder_tensor[-1-i]], axis = 3)
            x = tf.nn.relu(x)
            i += 1
            
            x = ops._deconv('deconv5', x, 4, 1024, 256, 2)
            x = tf.concat([x, encoder_tensor[-1-i]], axis = 3)
            x = tf.nn.relu(x)
            i += 1
            
            x = ops._deconv('deconv6', x, 4, 512, 128, 2)
            x = tf.concat([x, encoder_tensor[-1-i]], axis = 3)
            x = tf.nn.relu(x)
            i += 1
            
            x = ops._deconv('deconv7', x, 4, 256, 64, 2)
            x = tf.concat([x, encoder_tensor[-1-i]], axis = 3)
            x = tf.nn.relu(x)
            i += 1
            
            x = ops._deconv('deconv8', x, 4, 128, 3, 2, batch_norm = False)
            x = tf.nn.tanh(x)
            
            return x
    
    def discriminator(self, x, name = 'discriminator', reuse = False):
        
        with tf.variable_scope(name_or_scope = name, reuse = reuse):
            x = ops._conv('conv1', x, 4, self.channel * 2, 64, 2, batch_norm = False)
            x = tf.nn.leaky_relu(x, 0.2)

            x = ops._conv('conv2', x, 4, 64, 128, 2)
            x = tf.nn.leaky_relu(x, 0.2)

            x = ops._conv('conv3', x, 4, 128, 256, 2)
            x = tf.nn.leaky_relu(x, 0.2)

            x = ops._conv('conv4', x, 4, 256, 512, 1)
            x = tf.nn.leaky_relu(x, 0.2)
            
            x = ops._conv('conv5', x, 4, 512, 1, 1, False)
            x = tf.nn.sigmoid(x)
            
            return x
    
    def build_graph(self):
        
        self.data_loader.build_loader()
        image = self.data_loader.next_batch
        if self.transfer_type == 'B_to_A':
            self.image_A = image[1]
            self.image_B = image[0]
            self.image_A = tf.cast(self.image_A, tf.float32)
            self.image_B = tf.cast(self.image_B, tf.float32)
            self.image_A = (self.image_A - 127.5) / 127.5
            self.image_B = (self.image_B - 127.5) / 127.5
        else:
            self.image_A = image[0]
            self.image_B = image[1]
            self.image_A = tf.cast(self.image_A, tf.float32)
            self.image_B = tf.cast(self.image_B, tf.float32)
            self.image_A = (self.image_A - 127.5) / 127.5
            self.image_B = (self.image_B - 127.5) / 127.5
        
        if self.mode == 'train':
            self.fake_image = self.generator(self.image_A, reuse = False)
            self.L1_loss = tf.reduce_mean(tf.abs(self.image_B - self.fake_image))
            real_label = self.discriminator(tf.concat([self.image_A, self.image_B], axis = 3), reuse = False)
            fake_label = self.discriminator(tf.concat([self.image_A, self.fake_image],axis = 3), reuse = True)

            self.G_loss = - tf.reduce_mean(tf.log(fake_label + self.eps))
            self.D_loss_1 = - tf.reduce_mean(tf.log(real_label))
            self.D_loss_2 = - tf.reduce_mean(tf.log(1 - fake_label + self.eps))

            self.generator_loss = self.G_loss + 100.0 * self.L1_loss
            self.discriminator_loss = ( self.D_loss_2 + self.D_loss_1) / 2.0

            t_vars = tf.trainable_variables()
            gene_vars = [var for var in t_vars if 'generator' in var.name]
            dis_vars = [var for var in t_vars if 'discriminator' in var.name]

            adam_opt_G = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = self.beta1, beta2 = self.beta2)
            adam_opt_D = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = self.beta1, beta2 = self.beta2)

            self.generator_train = adam_opt_G.minimize(self.generator_loss, var_list = gene_vars)
            self.discriminator_train = adam_opt_D.minimize(self.discriminator_loss, var_list = dis_vars)
            
        elif self.mode == 'test':
            self.fake_image = self.generator(self.image_A, reuse = False)

