# **Heart U-net**
This work is an application of the U-net architecture for anatomical image segmentation, in particular, for the segmentation of the blood pool and myocardium from a Cardiovascular Magnetic Resonance (CMR) image, as well as a comparison between the original architecture and one in which the convolution operation is replaced by the use of separable convolution.

The data employed in this work was generated for the:
**HVSMR 2016**: MICCAI Workshop on Whole-Heart and Great Vessel Segmentation from 3D Cardiovascular MRI in Congenital Heart Disease, which its publicly available at
http://segchd.csail.mit.edu/index.html

<p align="center">
    <img src=https://drive.google.com/uc?id=1MdXBNYRpX4wjxRr0PF6zQVoiGZhKWrM4 alt="HVSMR"  />

</p>

## **Model arquitectures**
In this work two models were implemented, both of them making use of the Keras API.

The first model (U-net 1) is the standard U-net [1]

<p align="center">
    <img src=https://drive.google.com/uc?id=1e0s3kU3ekYksXQ060_i-SZ5Xy1spindX alt="HVSMR"  />

</p>

Some minor modifications were made to this model, like the incorporation of Batch Normalization [2] layers, as well as the use of Spatial DropOut [3] as a regularization method. Both of this modifications have already shown to lead to a faster and more stable learning process. 

As a latter approach (U-net 2), a modification was made in the original architecture, this is the replacement of most of the **convolution** operations for **separable convolutions** which are described and employed in works like [4,5,6].
**left** )Standard convolution, **right** )Separable convolution

<p align="center">
    <img src=https://drive.google.com/uc?id=13D75B9kCuS9XJYCpzDXZjsSgPMk4vZ9T alt="Image"  />

</p>

A direct advantage of the replacement of this operation can be observed by comparing the number of parameter used by an standard convolution, with the number of parameters used by a separable convolution, which ratio is:

<p align="center">
    <img src=https://drive.google.com/uc?id=1u7v0DuDZfLBwDYkHb5ce0yXhTJ2959lW alt="Image"  />

</p>

## **Experimental Results** 
Segmentation generated by both models is compared with the ground truth, pixels belonging to the *blood pool* class are bounded by the red contour, while pixels belonging to the *myocardium* class are bounded by the yellow contour.


**left**  )Ground Truth, **midle**  )U-net 1, **right**  )U-net 2

<p align="center">
    <img src=https://drive.google.com/uc?id=1J8dbnIEjmBz7gP2dREUizQxv9b0ctGbN alt="Image"  />

</p>


<p align="center">
    <img src=https://drive.google.com/uc?id=1Fvj2us1CHqdav7QXtkwOU1MMqduwhgwk alt="Image"  />

</p>



<p align="center">
    <img src=https://drive.google.com/uc?id=1thu6yJuPoFDZ3Rbau0OuHjrkcl2m_5ox alt="Image"  />

</p>

The output has tree chanels because each chanel encodes the information of its carresponding class.


<p align="center">
    <img src=https://drive.google.com/uc?id=1JW5XtFtKv5DU53LLm0Tnbbvyx4ldTPg0 alt="Image"  />
    

</p>


## **References**
[1] **U-Net: Convolutional Networks for Biomedical Image Segmentation**<br />
Olaf Ronneberger, Philipp Fischer, Thomas Brox <br />
[[link]](https://arxiv.org/abs/1505.04597).

[2] **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate shift**<br />
Sergey Ioffe, Christian Szegedy <br />
[[link]](https://arxiv.org/abs/1502.03167). In ICML, 2015.

[3] **Efficient Object Localization Using Convolutional Networks**<br />
Jonathan Tompson, Ross Goroshin, Arjun Jain, Yann LeCun, Christopher Bregler <br />
[[link]](https://arxiv.org/abs/1411.4280)

[4] **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.<br />
[[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.

[5] **Rigid-Motion Scattering For Image Classification** <br />
Laurent Sifre <br />
[[link]](https://www.di.ens.fr/data/publications/papers/phd_sifre.pdf)

[6 ] **Xception: Deep Learning with Depthwise Separable Convolutions**<br />
Fran�ois Chollet<br />
[[link]](https://arxiv.org/abs/1610.02357). In CVPR, 2017.