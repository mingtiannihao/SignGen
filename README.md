<h1 align="center"> SignGen: End-to-End Sign Language Video Generation with Latent Diffusion</h1>




## Method


<img src="pic/framework_10.png" alt="Editor" width="900">





## Experiment Results

#### RWTH-2014

![case1](pic/gif/1.gif "case1")![case2](pic/gif/2.gif "case2")![case3](pic/gif/3.gif "case3")  
![case4](pic/gif/4.gif "case4")![case4](pic/gif/5.gif "case4")![case4](pic/gif/6.gif "case4")  
![case4](pic/gif/10.gif "case4")![case4](pic/gif/11.gif "case4")![case4](pic/gif/12.gif "case4")   
![case4](pic/gif/19.gif "case4")![case4](pic/gif/22.gif "case4")![case4](pic/gif/21.gif "case4")   

#### RWTH-2014T

![case1](pic/gif/7.gif "case1")![case2](pic/gif/8.gif "case2")![case3](pic/gif/9.gif "case3")  
![case4](pic/gif/13.gif "case4")![case4](pic/gif/14.gif "case4")![case4](pic/gif/15.gif "case4")  
![case4](pic/gif/16.gif "case4")![case4](pic/gif/17.gif "case4")![case4](pic/gif/23.gif "case4")  

#### AUTSL
![case4](pic/gif/24.gif "case4")  
  
![case4](pic/gif/25.gif "case4")  

## Running by Yourself

### 1. Installation 

create a conda environment.
```
conda create -n  xxx  python==3.8.5 
```

Then you  can create the same environment as ours with the following command:
```
 pip install -r requirements.txt # install all requirements 
```

### 2. Download model weights

#### For LPIPS

The code will do it for you!
> Code will download [Alex](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) and move it into: `models/weights/v0.1/alex.pth`

#### For FVD

The code will do it for you!

> Code will download i3D model pretrained on [Kinetics-400](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI)
> Use `models/fvd/convert_tf_pretrained.py` to make `i3d_pretrained_400.pt`

### 3. Datasets

You can download these datasets such  as [RWTH-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/),[RWTH-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and [AUTSL](https://chalearnlap.cvc.uab.cat/dataset/40/data/66/description/).

> **How the data was processed:**
> 1. Download  AUTSL dataset to `/path/to/AUTSL`:\
> 2. Convert 128x128 images to HDF5 format:\
> `python datasets/sign_language_convert.py --sl_dir 'datasets/videos' --split 'train'  --out_dir 'datasets/signLanguages/train' --image_size 128  --force_h5 False

### Training -The code is coming soon.


