# S3Net

## Introduction

PyTorch code for the ICME 2021 paper [Selective, Structural, Subtle: Trilinear Spatial-Awareness for Few-Shot Fine-Grained Visual Recognition](http://cvteam.net/papers/2021_ICME_Selective,%20Structural,%20Subtle%20Trilinear%20Spatial-Awareness%20for%20Few-Shot%20Fine-Grained%20Visual%20Recognition.pdf).


## Dependencies

- conda env create -f pytorch.yml

## Training

- For example: Standford Cars dataset (1-shot)
- python mytrain_cars.py --nExemplars 1 --gpu-devices 0

## Testing

- For example: Standford Cars dataset (1-shot)
<<<<<<< HEAD

=======
>>>>>>> 95f9aa4d9841ecaf03abe73a1895f60ef372553c
- python test_car.py --nExemplars --gpu-devices 0 --resume ./result/car/5-shot-seed1-conv4_myspp_globalcos_few_loss/best_model.pth.tar


## Citation


<<<<<<< HEAD
<<<<<<< HEAD
=======


>>>>>>> c6530a797a42ced9467023c0492b5830c1c33695
=======


>>>>>>> 95f9aa4d9841ecaf03abe73a1895f60ef372553c
```html
@inproceedings{wu2021selective, 
title={Selective, Structural, Subtle: Trilinear Spatial-Awareness for Few-Shot Fine-Grained Visual Recognition}, 
author={Wu, Heng and Zhao, Yifan and Li, Jia}, 
booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
pages={1--6}, year={2021}, organization={IEEE} 
}
```


