# NYCU Intelligent System
###### tags: `Intelligent System`

## 3D Reconstruction

The goal of [homework1](https://github.com/gyes00205/NYCU_IS/tree/main/HW1) is to reconstruct the scene in [replica datasets](https://github.com/facebookresearch/Replica-Dataset) by using [Habitat simulator](https://github.com/facebookresearch/habitat-lab) to collect multi-view RGB-D images.

There are three main tasks to complete: **Data Collection**, **Point Cloud Alignment**, and **Camera Pose Estimation**.

1. Data Collection:
To reconstruct a scene, you need to control the agent walking through the whole scene, meanwhile saving the observation of the agent (RGB images, depth images), and the poses of the sensors.
2. Point Cloud Alignment and Reconstruction:
Since the agent is moving, the position of the sensor is changing by the time, so we need to align them using the **ICP algorithm**. 
3. Camera Pose Estimation and Visualization:
Visualizing the trajectory and show the difference between the trajectory we computed by ICP algorithm and ground truth trajectory.

<img src="https://i.imgur.com/4zFLKNR.png" width="300px">
<img src="https://i.imgur.com/964PdRM.png" width="300px">

## 3D Semantic Segmentation

In the [homework2](https://github.com/gyes00205/NYCU_IS/tree/main/HW2), we will train a semantic segmentation model, using data collected from the scenes (apartment_1, apartment_2, frl_apartment_0, frl_apartment_1, room_0, room_1, room_2, hotel_0, office_0, office_1) in the replica dataset. We will test the model on apartment_0 (which is the same scene used in the HW1). Then, we will label the reconstructed point cloud with semantic labels obtained from your trained model. Finally, a 3D semantic map can be generated.
[Datasets](https://docs.google.com/document/d/1PCaJ2L7kWUCN7w7erHnxOBDoCcsuIic5/edit?usp=sharing&ouid=114222386363369914303&rtpof=true&sd=true)

There are three main tasks in HW2, i.e., **Data Collection**, **finetune a semantic segmentation Model**, and **reconstruct a 3D semantic map**.
1. Data Collection:
In the data collection phase, we will collect two sets of training data. A set of data is similar to the distribution of testing data. The set of data is different from the distribution of testing data. The goal is to let you learn the impact of distribution shift on the performance of semantic segmentation.
2. Finetune a Semantic Segmentation Model:
Please check out the link: semantic-segmentation-pytorch, which has many semantic segmentation models pretrained on the ADE20k dataset. It is free to select your model of interest.
3. Reconstruct a 3D Semantic Map:
With predicted semantic images, the same method in HW1 can be used to generate 3D semantic maps.

![](https://i.imgur.com/5aXsMTO.png)

## 3D Navigation

In the HW2, we successfully reconstructed a 3D semantic map of apartment_0. Our next goal is to have a robot move to a desired location (for example, navigate the robot to find a specific item). Therefore, in the [homework3](https://github.com/gyes00205/NYCU_IS/tree/main/HW3), we focus on how to navigate from point A to B on the first floor of apartment_0 using the RRT algorithm.

There are three main tasks, i.e., **2D semantic map construction**, **RRT algorithm implementation**, **robot navigation**.

1. 2D semantic map construction:
In the second homework, we already have a semantic point cloud. We need a 2D semantic map for navigation.
2. RRT algorithm implementation:
We use the RRT algorithm to find a navagitable path from starting point A to goal point B (a specific item).
3. Robot navigation:
An agent can navigate automatically by following the path
calculated by RRT on the first floor of apartment_0 to find specific items.

![](https://i.imgur.com/mOF36Kn.png)