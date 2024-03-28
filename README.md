# GSoC--2024-DeepLense
![ml4sci_logo_angled](https://github.com/0-kush-0/GSoC--2024-DeepLense/assets/98215349/92f19fd2-feac-425d-8a45-a88090665a67)                   ![GSoC-icon-192](https://github.com/0-kush-0/GSoC--2024-DeepLense/assets/98215349/19ef0353-a205-4dd7-9d23-5dffe6dc40e2)  


March 2024
## About Me  
I am Kush Mathukiya, a final year Computer Science student at PDEU (Pandit Deendayal Energy University). I have been interested in and working towards ML since my second year. However, I have been working on Generative AI, specifically on diffusion models for about 1 year. Previously, I have worked on developing diffusion models for Morphing Attack Detection (MAD) where I used hard example mining and two-scale diffusion architecture. This year, I'm interning at ISRO (Indian Space Research Organisation) where I'm working on SISR (Single Image Super Resolution) through diffusion models.  


The given repository contains solutions to the following:  
### **The model weights for each test are stored in Google Drive, the link to which is provided in each CommonTest1/model_weights/link.txt and SpecificTest4/model_weights/link.txt**  
## 1) Common Test I. Multi-Class Classification  
**Data**  
For this test, we were required to classify a dataset comprising 30,000 training images belonging to three classes: no dark matter substructure, vortex substructure, and spherical substructure. The validation dataset comprising 7500 images (2500 for each class) was split in a 90:10 validation-to-testing ratio. The images were resized to 100 x 100 resolution to make them compatible with the model.  
**Model**  
 I chose Efficient-Net B2 [1] pre-trained architecture as the backbone for my model.  
EfficientNets [1] was first introduced by two engineers from Google Brain, Mingxing Tan, and Quoc V. Le in May 2019.  
They are a convolutional neural network architecture and scaling method that uniformly scales all depth/width/resolution dimensions using a compound coefficient.
It scales the model according to the available resources and the input resolution of images. Thus they are a good choice for typical classification problems like this one.

* Firstly features are extracted through the backbone as mentioned above which are then passed through a self-attention layer. Attention layers are crucial in capturing important parts of an image and also help in finding out the long-range dependencies (in our case, pixels) of the input.
* The output from the attention layer is then passed through three blocks each comprising of Linear, PReLU (to further add non-linearity in the model), BatchNorm (to prevent the output values from exploding), and Dropout (to prevent overfitting) layers to further fine-tune the model.
* Finally a linear layer outputs three values corresponding to each class which are then converted to probabilities using a softmax function (during testing) from which the maximum's label is treated as the final prediction for that image.

**Evaluation**  
**The testing accuracy obtained on the testing dataset was 94.25%**. The ROC curve obtained on the testing dataset for the three classes is given below.  
![roc-auc classification](https://github.com/0-kush-0/GSoC--2024-DeepLense/assets/98215349/31998a91-17a4-4426-b444-33890590f016)  
AUC (Area under the curve) for the three classes was 0.9899 for no substructure, 0.9879 for vortex substructure, and 0.9848 for spherical substructure.  

## 2) Specific Test IV. Diffusion Models  
**Data**  
The dataset for training a DDPM (Denoising Diffusion Probabilistic Model) comprised 10,000 strong lensing images with a resolution of 150 x 150. These images were resized to 144 x 144 to make them compatible with my UNet model.  
**Model**  
DDPMs (first proposed by Ho et al. in 2020 [2]) are a type of deep generative models that are trained to learn a given data distribution so as to ultimately sample novel images from this learned distribution.  
Their impressive generative capabilities were highlighted by Dhariwal and Nichol [3] where they showed that diffusion models were able to outperform GANs (Generative Adversarial Networks) which were considered state-of-the-art at the time.  
How do they work?  
* The forward process gradually destroys the image signal by corrupting it with Gaussian Noise at every timestep till it becomes isotropic Gaussian noise. The 'extent' to which the image is corrupted is determined by a beta schedule.
* The backward process performs 'reverse diffusion' whereby at every timestep, a deep learning model (which is treated as a noise predictor or as a score estimator in score-based models) typically a UNet, is trained to predict the amount of noise at a particular timestep. After training, a Gaussian noise vector is iteratively denoised to produce a sample from the learned distribution.  

**Architecture and strategy**  
Parts of the forward diffusion and mean, variance estimation are based on the implementation by Karras et al. [4].  
The beta/noise/variance schedule is based on [3] whereby the authors proved that the cosine beta schedule performs better than a linear schedule at optimally noising the images.  
My UNet model comprises 4 downsampling blocks, two middle blocks where a single attention layer is used, and 4 upsampling blocks. Channel expansion is done twice.  
The training was done for 100 epochs, batch size was set to 16, and the training loss used was L1 as L2 loss usually blurs out edges. 1000 sampling timesteps were used but 16 samples were generated in 15 seconds.  

**Evaluation**  
Some generated samples:  
![lens1](https://github.com/0-kush-0/GSoC--2024-DeepLense/assets/98215349/efe1658c-05d8-414d-be72-b556bd5db7d0) ![lens2](https://github.com/0-kush-0/GSoC--2024-DeepLense/assets/98215349/687d41fd-1608-4629-996b-ffb89bb773fd)  
Visually, these generated samples are not easy to differentiate from actual samples. But qualitatively? To assess the quality and diversity of generated images while also assessing the similarity to ground truth data, we use FID (Fréchet Inception Distance).   
**Note**  
It is important to note that FID scores calculated on features with dimensionality less than 2048 cannot be compared with those calculated on features with 2048 dimensionality. The former scores might also no longer correlate with visual quality. [5]    
Hence I have calculated FID 3040 images with 2048 feature dimensionality. **The FID score obtained was 6.011121595851975.** 

# References  
[1] Tan, M. and Le, Q.V. (2019) EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Proceedings of the 36th International Conference on Machine Learning, ICML 2019, Long Beach, 9-15 June 2019, 6105-6114.  
[2] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. ArXiv, abs/2006.11239.  
[3] Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. ArXiv, abs/2105.05233.  
[4] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. ArXiv, abs/2206.00364.  
[5] https://github.com/mseitzer/pytorch-fid  
