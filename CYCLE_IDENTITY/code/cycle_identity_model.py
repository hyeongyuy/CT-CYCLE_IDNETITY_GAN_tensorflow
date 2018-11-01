# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018

@author: yeohyeongyu
"""

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import cycle_identity_module as md
import inout_util as ut


class cycle_identity(object):
    def __init__(self, sess, args):
        self.sess = sess        
        
        ####patients folder name
        self.train_patent_no = [d.split('/')[-1] for d in glob(args.dcm_path + '/*') if ('zip' not in d) & (d not in args.test_patient_no)]     
        self.test_patent_no = args.test_patient_no    

        #module
        self.discriminator = md.discriminator
        self.generator = md.generator

        #network options
        OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim \
                              img_channel is_training')
        self.options = OPTIONS._make((args.ngf, args.nglf, args.ndf, 
                                      args.img_channel, args.phase == 'train'))

        """
        load images
        """
        print('data load... dicom -> numpy') 
        self.image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path, \
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, \
             is_unpair = args.unpair, model = args.model)
                                     
        self.test_image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path,\
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, \
             is_unpair = args.unpair, model = args.model)

        t1 = time.time()
        if args.phase == 'train':
            self.image_loader(self.train_patent_no)
            self.test_image_loader(self.test_patent_no)
            print('data load complete !!!, {}\nN_train : {}, N_test : {}'.format(time.time() - t1, len(self.image_loader.LDCT_image_name), len(self.test_image_loader.LDCT_image_name)))
            [self.patch_X , self.patch_Y] = self.image_loader.input_pipeline(self.sess, args.patch_size, args.end_epoch)
        else:
            self.test_image_loader(self.test_patent_no)
            print('data load complete !!!, {}, N_test : {}'.format(time.time() - t1, len(self.test_image_loader.LDCT_image_name)))
        
            self.patch_X =  tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'LDCT')
            self.patch_Y =  tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'NDCT')
     
        
        """
        build model
        """
        #### image placehold(for test)
        self.test_X = tf.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='X')
        self.test_Y = tf.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='Y')


        #### Generator & Discriminator
        #Generator
        self.G_X = self.generator(self.patch_X, self.options, False, name="generatorX2Y")
        self.F_GX = self.generator(self.G_X, self.options, False, name="generatorY2X")
        self.F_Y = self.generator(self.patch_Y, self.options, True, name="generatorY2X")
        self.G_FY = self.generator(self.F_Y, self.options, True, name="generatorX2Y")
        
        self.G_Y = self.generator(self.patch_Y, self.options, True, name="generatorX2Y")  #G : x->y 
        self.F_X = self.generator(self.patch_X, self.options, True, name="generatorY2X")  #F : y->X
                
        #Discriminator
        self.D_GX = self.discriminator(self.G_X, self.options, reuse=False, name="discriminatorY")
        self.D_FY = self.discriminator(self.F_Y, self.options, reuse=False, name="discriminatorX")
        self.D_Y = self.discriminator(self.patch_Y, self.options, reuse=True, name="discriminatorY")
        self.D_X = self.discriminator(self.patch_X, self.options, reuse=True, name="discriminatorX")

        #### Loss
        #generator loss
        self.cycle_loss = md.cycle_loss(self.patch_X, self.F_GX, self.patch_Y, self.G_FY, args.L1_lambda)
        self.identity_loss = md.identity_loss(self.patch_Y, self.G_Y, self.F_X, self.patch_X, args.L1_gamma)
        self.G_loss_X2Y = md.least_square(self.D_GX, tf.ones_like(self.D_GX)) 
        self.G_loss_Y2X = md.least_square(self.D_FY, tf.ones_like(self.D_FY)) 
        
        self.G_loss = self.G_loss_X2Y + self.G_loss_Y2X + self.cycle_loss + self.identity_loss

        #dicriminator loss
        self.D_loss_patch_Y = md.least_square(self.D_Y, tf.ones_like(self.D_Y))
        self.D_loss_patch_GX = md.least_square(self.D_GX, tf.zeros_like(self.D_GX))
        self.D_loss_patch_X = md.least_square(self.D_X, tf.ones_like(self.D_X))
        self.D_loss_patch_FY = md.least_square(self.D_FY, tf.zeros_like(self.D_FY))
        
        self.D_loss_Y = (self.D_loss_patch_Y + self.D_loss_patch_GX)
        self.D_loss_X = (self.D_loss_patch_X + self.D_loss_patch_FY)
        self.D_loss = (self.D_loss_X + self.D_loss_Y) / 2
    
        #### variable list
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        
        
        #### optimizer
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.D_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.G_loss, var_list=self.g_vars)
        
        """
        Summary
        """
        #### loss summary
        #generator
        self.G_loss_sum = tf.summary.scalar("1_G_loss", self.G_loss, family = 'Generator_loss')
        self.cycle_loss_sum = tf.summary.scalar("2_cycle_loss", self.cycle_loss, family = 'Generator_loss')
        self.identity_loss_sum = tf.summary.scalar("3_identity_loss", self.identity_loss, family = 'Generator_loss')
        self.G_loss_X2Y_sum = tf.summary.scalar("4_G_loss_X2Y", self.G_loss_X2Y, family = 'Generator_loss')
        self.G_loss_Y2X_sum = tf.summary.scalar("5_G_loss_Y2X", self.G_loss_Y2X, family = 'Generator_loss')
        self.g_sum = tf.summary.merge([self.G_loss_sum, self.cycle_loss_sum, self.identity_loss_sum, self.G_loss_X2Y_sum, self.G_loss_Y2X_sum])


        #discriminator
        self.D_loss_sum = tf.summary.scalar("1_D_loss", self.D_loss, family = 'Discriminator_loss')
        self.D_loss_Y_sum = tf.summary.scalar("2_D_loss_Y", self.D_loss_patch_Y, family = 'Discriminator_loss')
        self.D_loss_GX_sum = tf.summary.scalar("3_D_loss_GX", self.D_loss_patch_GX, family = 'Discriminator_loss')
        self.d_sum = tf.summary.merge([self.D_loss_sum, self.D_loss_Y_sum, self.D_loss_GX_sum])

        #### image summary
        self.test_G_X = self.generator(self.test_X, self.options, True, name="generatorX2Y")
        self.train_img_summary = tf.concat([self.patch_X, self.patch_Y, self.G_X], axis = 2)
        self.summary_image_1 = tf.summary.image('1_train_patch_image', self.train_img_summary)
        self.test_img_summary = tf.concat([self.test_X, self.test_Y, self.test_G_X], axis = 2)
        self.summary_image_2 = tf.summary.image('2_test_whole_image', self.test_img_summary)

        #### psnr summary
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", ut.tf_psnr(self.test_X, self.test_Y, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", ut.tf_psnr(self.test_Y, self.test_G_X, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])

        
        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)

        print('--------------------------------------------\n# of parameters : {} '.\
             format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        

    def train(self, args):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        #pretrained model load
        self.start_step = 0 #load SUCESS -> self.start_step 파일명에 의해 초기화... // failed -> 0
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")


        #iteration -> epoch
        self.start_epoch =  int((self.start_step + 1) / len(self.image_loader.LDCT_image_name))

        print('Start point : iter : {}, epoch : {}'.format(self.start_step, self.start_epoch))    
        
        
        start_time = time.time()
        lr = args.lr
        for epoch in range(self.start_epoch, args.end_epoch):
           
            batch_idxs = len(self.image_loader.LDCT_image_name)

            #decay learning rate
            if epoch > args.decay_epoch:
                lr = args.lr - (epoch - (args.decay_epoch)) * ((args.lr / (args.end_epoch - args.decay_epoch)))
                                  
            for _ in range(0, batch_idxs):
                # Update G network
                
                X, Y, F_Y, G_X, _, summary_str = self.sess.run(
                    [self.patch_X , self.patch_Y, self.F_Y, self.G_X, self.g_optim, self.g_sum], feed_dict = { self.lr:lr})

                self.writer.add_summary(summary_str, self.start_step)

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],  feed_dict = {self.patch_X:X , self.patch_Y:Y, self.lr:lr})

                self.writer.add_summary(summary_str, self.start_step)


                if (self.start_step+1) % args.print_freq == 0:
                    currt_step = self.start_step % len(self.image_loader.LDCT_image_name) if epoch != 0 else self.start_step
                    print(("Epoch: {} {}/{} time: {} lr {}: ".format(epoch, currt_step, batch_idxs, time.time() - start_time, lr)))
                    
                    #summary trainig sample image
                    summary_str1 = self.sess.run(self.summary_image_1)
                    self.writer.add_summary(summary_str1, self.start_step)
                    
                    #check sample image
                    self.check_sample(args, self.start_step)

                if (self.start_step+1) % args.save_freq == 0:
                    self.save(args.checkpoint_dir, self.start_step)
                
                self.start_step += 1
        
        self.image_loader.coord.request_stop()
        self.image_loader.coord.join(self.image_loader.enqueue_threads)
        
    #summary test sample image during training
    def check_sample(self, args, idx):
        sltd_idx = np.random.choice(range(len(self.test_image_loader.LDCT_image_name)))

        sample_X_image, sample_Y_image  = self.test_image_loader.LDCT_images[sltd_idx], self.test_image_loader.NDCT_images[sltd_idx]


        G_X = self.sess.run(
            self.test_G_X,
            feed_dict={self.test_Y: sample_Y_image.reshape([1] + self.test_Y.get_shape().as_list()[1:]), 
            self.test_X: sample_X_image.reshape([1] + self.test_Y.get_shape().as_list()[1:])})

        G_X = np.array(G_X).astype(np.float32)

        summary_str1, summary_str2 = self.sess.run(
            [self.summary_image_2, self.summary_psnr],
            feed_dict={self.test_X : sample_X_image.reshape([1] + self.test_X.get_shape().as_list()[1:]), 
                       self.test_Y : sample_Y_image.reshape([1] + self.test_Y.get_shape().as_list()[1:]),
                       self.test_G_X: G_X.reshape([1] + self.test_G_X.get_shape().as_list()[1:])})
      
        self.writer.add_summary(summary_str1, idx)
        self.writer.add_summary(summary_str2, idx)  

       
                
    # save model    
    def save(self, checkpoint_dir,  step):
        model_name = "cycle_identity.model"
        checkpoint_dir = os.path.join('.', checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    # load model    
    def load(self, checkpoint_dir = 'checkpoint'):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join('.', checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

            
    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)    
        npy_save_dir = os.path.join('.', args.test_npy_save_dir)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)
            
            
        ## test
        for idx in range(len(self.test_image_loader.LDCT_images)):
            test_X = self.test_image_loader.LDCT_images[idx]
            
            mk_G_X = self.sess.run(self.test_G_X, feed_dict={self.test_X: test_X.reshape([1] + self.test_X.get_shape().as_list()[1:])})
            
            save_file_nm_g = 'Gen_from_' + self.test_image_loader.LDCT_image_name[idx]
            
            np.save(os.path.join(npy_save_dir, save_file_nm_g), mk_G_X)
