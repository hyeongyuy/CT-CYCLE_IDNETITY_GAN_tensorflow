# CYCLE_IDENTITY_GAN-tensorflow
Kang, E., Koo, H. J., Yang, D. H., Seo, J. B., & Ye, J. C. (2018). Cycle Consistent Adversarial Denoising Network for Multiphase Coronary CT Angiography. arXiv preprint arXiv:1806.09748.<br>

* WGAN_VGG
>	* paper : https://arxiv.org/pdf/1806.09748.pdf
>	* reference code:  
>     * cyclegan : https://github.com/xhujoy/CycleGAN-tensorflow


## I/O (DICOM file -> .npy)
* Input data Directory  
  * DICOM file extension = [<b>'.IMA'</b>, '.dcm']
> $ os.path.join(dcm_path, patient_no, [LDCT_path|NDCT_path], '*.' + extension)

## Ntwork architecture  
* Generator(revised image)
> reference : Kang, E., Chang, W., Yoo, J., & Ye, J. C. (2018). Deep convolutional framelet denosing for low-dose ct via wavelet residual network. IEEE transactions on medical imaging, 37(6), 1358-1369.
![generator architecture](https://github.com/hyeongyuy/CT-CYCLE_IDNETITY_GAN_tensorflow/blob/master/img/revised_generator_arc.JPG)<br>

> 1. To reduce network complexity, images are used directly as inputs to the network instead of the wavelet transform coefficients<br>

> 2. The First convolution layer uses 128 set of 3 * 3 convolution kernels to produce 128 channel feature maps.

> 3. We have 6 set of module composed of 3 sets of convolution, batch normalization, and ReLU layers, and one bypass connection with a ReLU layer.  Convolution layers in the modules use 128 set of 3 * 3 * 128 convolution kernels. 

> 4. In addition, the proposed network has a concatenation layer that concatenates the inputs of each module and the output of the last module.

> 5. followed by the convolution layer with 128 set of 3 * 3 * 896 convolution kernels. The last convolution layer uses 15 sets of 3 * 3 * 128 convolution kernels. Finally, we add an end-to-end bypass connection.


* discriminator
![discriminator architecture](https://github.com/hyeongyuy/CT-CYCLE_IDNETITY_GAN_tensorflow/blob/master/img/discriminator_arc.JPG)<br>
> * it consists of 5 convolution layers including the last fully-connected layer. The first convolution layer uses 64 sets of 4 * 4 convolution kernels, and the number of convolution kernels in the following layers is twice that of the previous layer except the last fully connected layer.

## Training detail  
> * mini batch size : 10
> * patch size : 56*56
> * opt : Adam(learning rate = 0.0002, beta1 = 0.5, beta2 = 0.999)
> * learning rate decay : first 100 eppoch 0.0002 and linearly decreased it to zero over the next epochs.
> * lambda(weight of cycle loss) : 10
> * gamma(weight of identity loss) : 5
> * epoch : 160
> * normalized input : -1 ~ 1

## Main file(main.py) Parameters
* Directory
> * dcm_path : dicom file directory
> * LDCT_path : LDCT image folder name
> * NDCT_path : NDCT image folder name
> * test_patient_no : test patient id list(p_id1,p_id2...) (train patient id : (patient id list - test patient id list)
> * checkpoint_dir : save directory - trained model
> * test_npy_save_dir : save directory - test numpy file

* Image info
> * patch_size : patch size 
> * whole_size : whole size
> * img_channel : image channel
> * img_vmax : max value
> * img_vmin : min value
* Train/Test
> * model : red_cnn, wgan_vgg, cyclegan, cycle_identity (for image preprocessing)
> * phase : train | test

* Training detail
> * end_epoch : end epoch (default = 160)
> * lr : learning rate (default=0.0002)
> * batch_size : batch size (default=10)
> * L1_lambda : weight of cyclic loss (default=10)
> * L1_gamma : weight of identity loss (default=5)
> * beta1 : Adam optimizer parameter (default=0.5)
> * beta2 : Adam optimizer parameter (default=0.999)
> * ngf : # of generator filters in first conv layer
> * nglf : # of generator filters in last conv layer
> * ndf : # of discriminator filters in first conv layer

* others
> * save_freq : save a model every step (iterations)
> * print_freq : print frequency (iterations)
> * continue_train : load the latest model: true, false
> * gpu_no : visible devices(gpu no)
> * unpair : unpaired image(cycle loss), (default=True)

## Run
* train
> python main.py
* test
> python main.py --phase=test
