# HW3

## Build 2D Map
Please execute `build_2S_map.py` by command shown below.
```
python Codes/build_2D_map.py
```
Double click a start point and then input the target which you want to go.
In `build_2D_map.py`, I import RRT algorithm which implement in `RRT.py` to search the route to target.
```python=
from RRT import RRT
.
.
.
route = RRT(
    start=data['start'],
    target=data['target'],
    image='Map/map.png',
    target_sample_rate=5,
    step_size=10
)
route.planning()
```
It will generate `map.png` and `route.png` in `Map` directory and `position.csv` which store habitat coordinate information.

* **map.png**

![](https://i.imgur.com/D5V0iON.png)

* **route.png**

![](https://i.imgur.com/mOF36Kn.png)

## Navigation
Please execute `navigation.py` by the command shown below.
```
python Codes/navigation.py
```
It will output `Observation.avi`.

## File Formate

```
HW3_310551083
├───apartment_0
├───semantic_3d_pointcloud
│      ├───color01.npy
│      ├───color0255.npy
│      └───point.npy
├───Codes
│      ├───build_2D_map.py
│      ├───RRT.py
│      └───navigation.py
├───Map
│      ├───map.png
│      └───route.png
├───Observation.avi
└───position.csv
```
