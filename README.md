# X-STARS

This is the official PyTorch implementation of the ["*Cross-sensor self-supervised training and alignment for remote sensing*"](https://arxiv.org/abs/2405.09922) paper, published on [JSTARS](https://ieeexplore.ieee.org/abstract/document/10982156).

This repository builds upon the original [DINO implementation](https://github.com/facebookresearch/dino). You can follow that repo to install the required packages.

## Pretraining

To run the pretraining from scratch, you can run:
```bash
python pretraining.py --arch vit_tiny --data_path data/path/ --output_dir /output/directory --epochs 400 --batch_size_per_gpu 4 --use_msad --msad_embedding_dim 192 --sensors Sentinel Landsat --mean 0.15590523 0.15850738 0.10111853 --std 0.14238988 0.11567883 0.0910672
```

To run the continual pretraining, you can run:
```bash
python continual_pretraining.py --arch vit_tiny --data_path data/path/ --output_dir /output/directory --epochs 400 --batch_size_per_gpu 12 --use_msad --msad_embedding_dim 192 --sensors Sentinel Landsat --mean 0.15590523 0.15850738 0.10111853 --std 0.14238988 0.11567883 0.0910672 --adapt_sensor Landsat --pretrained_weights pretrain/net/weights 
```

## Dataset

The dataset class is shaped on the *MSC-France dataset*, presented in the already mentioned paper. The name of the images is the same for each sensor. The directories are organized as follows:
```bash
MSC-France
├─Sentinel
   ├─Bordeaux
   ├─Grenoble
   ...
   └─Toulouse
├─Landsat
   ├─Bordeaux
   ├─Grenoble
   ...
   └─Toulouse
└─SPOT
   ├─Bordeaux
   ├─Grenoble
   ...
   └─Toulouse
```

Dataset is downloadable [here](https://www.eotdl.com/datasets/MSC-France).

## Model weights

The pre-trained models are available at this [link](https://drive.google.com/drive/folders/1NSZqUdytaDq6yFC188dG8K27YKHgtFGU?usp=sharing).

## Citation

```
@article{marsocci2025cross,
  title={Cross-sensor self-supervised training and alignment for remote sensing},
  author={Marsocci, Valerio and Audebert, Nicolas},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```
