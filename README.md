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
- python test_car.py --nExemplars --gpu-devices 0 --resume ./result/car/CAM/5-shot-seed1-conv4_myspp_globalcos_few_loss/best_model.pth.tar


## Citation




```html
@inproceedings{wu2021selective, 
title={Selective, Structural, Subtle: Trilinear Spatial-Awareness for Few-Shot Fine-Grained Visual Recognition}, 
author={Wu, Heng and Zhao, Yifan and Li, Jia}, 
booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
pages={1--6}, year={2021}, organization={IEEE} 
}
```


