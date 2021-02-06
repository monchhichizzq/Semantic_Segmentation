# Semantic_Segmentation
Image segmentation refers to dividing an image into several disjoint regions based on features such as grayscale, color, spatial texture, geometric shape, etc., so that these features show consistency or similarity within the same region, while showing significant differences between different regions


### Up-sampling/ Transpose convolution based segmentation method
a. FCN(Fully Convolutional Network)
- FCN [parameters: 134,473,084]
<p align="center">
      <img src="images/FCN.png", width="640", height='480'>

- Advantages

 1. FCN classifies images at the pixel level, thus solving the semantic level image segmentation problem
 2. The FCN can accept input images of any size, it can maintain the spatial information in the original input image

- Disadvantages:

 1. The up-sampled feature maps are blurred and smooth due to upsampling 
 2. It is insensitive to the details in the image
 3. Separate classification of individual pixels, without adequate consideration of pixel-to-pixel relationships, lacking spatial consistency

b. Segnet
<p align="center">
      <img src="images/SegNet.png", width="640", height='480'>

SegNet is a deep network proposed by Cambridge aimed at solving image semantic segmentation for autonomous driving or intelligent robots. 
SegNet is based on FCN and is very similar to the idea of FCN, except that its encoder-decoder is slightly different from that of FCN, using depooling in its decoder to upsample the feature map and maintaining the integrity of high-frequency details in the sub-sample; while the encoder does not use a fully connected layer. Therefore, it is a lightweight network with fewer parameters.
[去池化上采样, 少参数, 边界, 重叠处理不好]

- Advantages；
1. Preservation of the integrity of the High Frequent component.
2. Non-bulky networks with fewer parameters, which are more lightweight.

- Disadvantages:
1. Lower confidence in the location of the boundaries for classification.
2. For indistinguishable categories, such as people and bicycles, uncertainty increases if there is mutual overlap between the two.

