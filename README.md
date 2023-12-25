### Instance Segmentation using UNET CNN implemented from scratch
First of all this project is based on this amazing tutorial: https://www.youtube.com/watch?v=IHq1t7NxS8k

Instance segmentation of images is the task of predicting a class label for every pixel in the input image.  
In this project I used the Carvana car's dataset which contains only two labels, "car" : 1 & "background" : 0.  

The chosen model was a famous type of CNN used mainly for segmentation tasks called UNET,  
as presented in the original paper https://arxiv.org/abs/1505.04597

We can think of UNET as an encoder-decoder architecture.  
Encoder extracts features while downsampling the original input, and caching some intermediate representations of the input for later usage.  
Decoder uses the features as input and via transposed convolutions and concatenation with the cached input representations, performs an upsampling back to the original input shape with channels equals to the number of classes.

The output of the model are num_classes logits matrices, and after Softmax activation on the channel dimension we get probability matrices, in which each pixel has a value representing the probability of this pixel to be of the channel's class.
The segmentation quality was measured with pixel accuracy & dice score metrics as show in the main notebook.

In the figures below you can view the UNET architecture, model's inference on test samples & TB graphs from the training.

  
![UNET_architectue_paper](https://github.com/matfain/Semantic-Segmentation-UNET-from_scratch/assets/132890076/df3a690b-cab3-4594-9bca-b18b76aebe0f)

![image](https://github.com/matfain/Semantic-Segmentation-UNET-from_scratch/assets/132890076/27643529-e01e-4e37-9605-b18ae30eea13)

![TB_training_loss](https://github.com/matfain/Semantic-Segmentation-UNET-from_scratch/assets/132890076/3559ad90-7da9-40f2-8e3d-8bff07b06c15)
![TB_binary_dice_score](https://github.com/matfain/Semantic-Segmentation-UNET-from_scratch/assets/132890076/e5c42b71-bab1-44fd-9798-b841ec0e9818)
