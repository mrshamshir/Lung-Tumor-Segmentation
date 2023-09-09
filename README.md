## Introduction

First, we need to obtain and preprocess the data for the segmentation task
The data is provided by the medical segmentation decathlon challenge(http://medicaldecathlon.com/) <br />

(Data License: CC-BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/) <br/>
![alt text](images/my_animation_1.gif?raw=true)

## Preprocessing

1. CT images have a fixed range from -1000 to 3071. **Thus we can normalize by dividing by 3071** <br /> we don't need to compute mean and standard deviation for this task
2. As we want to focus on lung tumors, we can crop away parts of the lower abdomen to reduce the complexity and help the network learn. As an example, **we might skip the first 30 slices (from lower abdomen to the neck)** (last axis)
3. Resize the single slices and masks to (256, 256) (when resizing the mask, pass interpolation=cv2.INTER_NEAREST to the resize function to apply nearest neighbour interpolation)
4. As we want to tackle this task on a slice level (2D) and not on a subject level (3D) to reduce the computational cost **we should store the preprocessed data as 2D files**, because reading a single slice is much faster than loading the complete NIfTI file.


## DataSet Creation
We need to implement the following functionality:
1. Create a list of all 2D slices. To so we need to extract all slices from all subjects
2. Extract the corresponding label path for each slice path
3. Load slice and label
4. Data Augmentation.
5. Return slice and mask <br/>
![alt text](https://github.com/)

## Model
then, we will create the model for the lung tumor segmentation! <br />
We will use the most famous architecture for this task, the U-NET (https://arxiv.org/abs/1505.04597). <br/>

The idea behind a UNET is the Encoder-Decoder architecture with additional skip-connctions on different levels:
The encoder reduces the size of the feature maps by using downconvolutional layers.
The decoder reconstructs a mask of the input shape over several layers by upsampling.
Additionally skip-connections allow a direct information flow from the encoder to the decoder on all intermediate levels of the UNET.
This allows for a high quality of the produced mask and simplifies the training process.<br />

## Training
We will implement full segmentaion model with pytorch-lightning.
### Oversampling to tackle strong class imbalance
Lung tumors are often very small, thus we need to make sure that our model does not learn a trivial solution which simply outputs 0 for all voxels.<br />
We will use oversampling to sample slices which contain a tumor more often.

To do so we can use the **WeightedRandomSampler** provided by pytorch which needs a weight for each sample in the dataset.
### Loss

As this is a harder task to train you might try different loss functions:
We achieved best results by using the Binary Cross Entropy instead of the Dice Loss. <br/>
Computed Dice-score: 0.878

## Visualization
This is the visualization of model predictions on one of the test samples, which we don't have lables for them.

![alt text](images/my_animation_2.gif?raw=true)
