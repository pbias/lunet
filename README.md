# LU-Net: An Efficient Network for 3D LiDAR Point Cloud Semantic Segmentation Based on End-to-End-Learned 3D Features and U-Net
_By P. Biasutti, V. Lepetit, J-F. Aujol, M. Brédif, A. Bugeau (LaBRI, IMB, IGN, GEOSAT)_

This repository contains the implementation of LU-Net, a CNN designed for semantic segmentation of LiDAR point clouds. The implementation is done in Python and Tensorflow.


![alt text](https://github.com/pbias/lunet/blob/master/images/2100_pred_3d.png "3D semantic segmentation")
**_<p align=center>Result of semantic segmentation using LU-Net</p>_**

For complete details about the model, please refer to our LU-Net paper: [https://arxiv.org/abs/1908.11656](https://arxiv.org/abs/1908.11656). If you use this code or this work, please consider citing:
```latex
@inproceedings{biasutti2019lunet,
    title={LU-Net: An Efficient Network for 3D LiDAR Point Cloud Semantic Segmentation 
      Based on End-to-End-Learned 3D Features and U-Net},
    author={Biasutti, P. and Lepetit, V. and Aujol, J-F. and Brédif, M. and Bugeau, A.},
    booktitle={ICCV Workshop},
    year={2019},
}
```

## Requirements
The instructions have been tested on Ubunutu 16.04 with Python 3.6 and Tensorflow 1.6 with GPU support.

First, clone the repository:
```bash
git clone https://github.com/pbias/lunet.git
```

Then, download the SqueezeSeg dataset as explained [here](https://github.com/xuanyuzhou98/SqueezeSegV2) in Section "Dataset". Then, in **make_tfrecord_pn.py** line 29, set the variable "**semantic_base**" to the path of the SqueezeSeg dataset folder.
```python
semantic_base = "/path/to/squeezeseg/dataset/"
```

## Train and validate
You can generate the _training_ and _validation_ **TFRecords** by running the following command:
```bash
python make_tfrecord_pn.py --config=config/lunet.cfg
```

Once done, you can start training the model by running:
```bash
python train.py --config=config/lunet.cfg --gpu=0
```
You can set which GPU is being used by tuning the `--gpu` parameter.

While training, you can run the validation as following:
```bash
python test.py --config=config/lunet.cfg --gpu=1
```
This script will run the validation on each new checkpoint as soon as they are created, and will store the scores in a text file.

During training, the code will produce logs for Tensorboard, as specified in the configuration file (see after). You can run Tensorboard while training with the following command:
```bash
tensorboard --logdir=training_path/logs
```

## Customize the settings
You can easily create different configuration settings by editing the configuration file located in **config/lunet.cfg**. Here is a small description of each setting:
```yaml
[DATA]
tfrecord_train : data/train.tfrecord # Train TFRecord filepath
tfrecord_val   : data/val.tfrecord   # Validation TFRecord filepath
augmentation   : ["original"]            
n_size         : [3, 3]              # Dimensions of the 2D neighborhood for the 3D feature extraction module
channels       : xyzdr                   
pointnet       : True                # If 3D feature extraction module should be used

[NETWORK]
n_classes : 4                        # Number of classes of the dataset
img_width : 512                      # Range-image width
img_height: 64                       # Range-image height

[TRAINING]
unet_depth       : 5                 # Scale levels of the U-Net module
batch_size       : 2                 # Batch size
learning_rate    : 0.0001            # Learning rate
lr_decay_interval: 500000            # Decay interval (if needed)
lr_decay_value   : 0.1               # Decay value
focal_loss       : True              # Use focal loss or not
num_iterations	 : 500000            # Training iterations
val_interval     : 100               # Validation interval (for tensorboard, only validate on one batch)

[TRAINING_OUTPUT]

path         : training/             # Output directory 
logs         : logs/                 # Tensorboard logdir in the output directory
model        : model.ckpt            # Checkpoint name
save_interval: 5000                  # Checkpoint saving interval (in iteration)

[TEST]
output_path          : validation/   # Output path for validation
```
## TODO / Remarks
* Add the path of the dataset as a field in the configuration file
* Tensorboard logs are computed batch-wise for the moment
