# HW2
## Training Segmentation Model
* **Model:** MobileNetV2dilated-C1_deepsup
* **training setting:** config.yaml
```yaml 
DATASET:
  root_dataset: "./data/"
  list_train: "./data/dataset0/training.odgt"
  list_val: "./data/dataset0/validation.odgt"
  num_class: 101
  imgSizes: (300, 375, 450, 525, 600)
  imgMaxSize: 2000
  padding_constant: 8
  segm_downsampling_rate: 8
  random_flip: True

MODEL:
  arch_encoder: "mobilenetv2dilated"
  arch_decoder: "c1_deepsup"
  fc_dim: 320

TRAIN:
  batch_size_per_gpu: 3
  num_epoch: 30
  start_epoch: 20
  epoch_iters: 5000
  optim: "SGD"
  lr_encoder: 0.02
  lr_decoder: 0.02
  lr_pow: 0.9
  beta1: 0.9
  weight_decay: 1e-4
  deep_sup_scale: 0.4
  fix_bn: False
  workers: 16
  disp_iter: 20
  seed: 304

VAL:
  visualize: True
  checkpoint: "epoch_25.pth"

TEST:
  checkpoint: "epoch_25.pth"
  result: "./"

DIR: "ckpt/dataset0-mobilenetv2"
```
* **training result:**

    apartment0: Accuracy 98.23%, Loss 0.079407 
    
    other scenes: Accuracy 97.33%, Loss 0.080057
* **evaluation result:**
    
    apartment0: mIOU 0.3658, Accuracy 97.37%
    
    other scenes: mIOU 0.0422, Accuracy 56.53%
    
## Reconstruct 3D Semantic Map
* For first floor
    ```
    python 3d_semantic_map.py \
    --test_scene=Data_collection/first_floor \
    --floor=1
    ```

* For second floor
    ```
    python 3d_semantic_map.py \
    --test_scene=Data_collection/second_floor \
    --floor=1
    ```

* Directory Structure
    ```
    Data_collection
    ├───first_floor
    │      ├───depth
    │      ├───dataset0_pred # aparentment0 model predict
    │      └───other_pred # other scene model predict
    └───second_floor
           ├───depth
           ├───dataset0_pred
           └───other_pred
    ```
![](https://i.imgur.com/5aXsMTO.png)
![](https://i.imgur.com/U99bjU4.png)
