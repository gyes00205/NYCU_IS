# Hw1
## Data Collection: load.py
* output_dir: the directory where you would store the rgb and depth folders
* floor: which floor you want to collect the data
* step: you will take a photo every few steps

1. For example: first floor
```
python load.py \
--output_dir=Data_collection/first_floor \
--floor=1 \
--step=2
```
2. For example: second floor
```
python load.py \
--output_dir=Data_collection/second_floor \
--floor=2 \
--step=1
```
## 3D reconstruction: reconstruct.py
* test_scene: the directory that store rgb and depth folders
* floor: which floor you want to reconstruct
* use_open3d: set 1 if use open3d icp, else set 0 to use own icp

1. For example: first floor
```
python reconstruct.py \
--test_scene=Data_collection/first_floor \
--floor=1 \
--use_open3d=1
```
2. For example: second floor
```
python reconstruct.py \
--test_scene=Data_collection/second_floor \
--floor=2 \
--use_open3d=1
```

## Result
1. first_floor_open3d v.s. first_floor_my
<img src="https://i.imgur.com/4zFLKNR.png" width="300px"><img src="https://i.imgur.com/XfTe1gP.png" width="250px">
2. second_floor_open3d v.s. second_floor_my
<img src="https://i.imgur.com/964PdRM.png" width="300px"><img src="https://i.imgur.com/jvM7Oo2.png" width="300px">