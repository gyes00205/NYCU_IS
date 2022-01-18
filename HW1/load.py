# Usage: python load.py --output_dir=Data_collection/first_floor --floor=1 --step=2

import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--test_scene", default="apartment_0/apartment_0/habitat/mesh_semantic.ply")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--floor", default=1)
parser.add_argument("--step", default=1)
args = parser.parse_args()

# Create directory to store image and pose
rgb_path = os.path.join(args.output_dir, 'rgb')
depth_path = os.path.join(args.output_dir, 'depth')
semantic_path = os.path.join(args.output_dir, 'semantic')
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)
os.makedirs(semantic_path, exist_ok=True)


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = args.test_scene

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0]) if int(args.floor) == 1 else  np.array([0.0, 1.0, 0.0])# agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
img_cnt = int(args.step)
pose_data = []
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")


def navigateAndSee(action="", img_cnt=0):
    global pose_data
    if action in action_names:
        img_num = int(img_cnt)//int(args.step)
        observations = sim.step(action)
        #print("action: ", action)
        rgb_img = transform_rgb_bgr(observations["color_sensor"])
        depth_img = transform_depth(observations["depth_sensor"])
        semantic_img = transform_semantic(observations["semantic_sensor"])
        rgb_filename = os.path.join(rgb_path, str(img_num)+'.png')
        depth_filename = os.path.join(depth_path, str(img_num)+'.png')
        semantic_filename = os.path.join(semantic_path, str(img_num)+'.png')
        
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        
        cv2.imshow("RGB", rgb_img)
        cv2.imshow("depth", depth_img)
        cv2.imshow("semantic", semantic_img)
        if (int(img_cnt) % int(args.step)) == 0:
            cv2.imwrite(rgb_filename, rgb_img)
            cv2.imwrite(depth_filename, depth_img)
            cv2.imwrite(semantic_filename, semantic_img)
            pose_data.append([sensor_state.position[0], sensor_state.position[1], sensor_state.position[2], sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z])
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)


action = "move_forward"
navigateAndSee(action, img_cnt)

while True:
    keystroke = cv2.waitKey(0)
    img_cnt += 1
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        navigateAndSee(action, img_cnt)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        navigateAndSee(action, img_cnt)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        navigateAndSee(action, img_cnt)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        df = pd.DataFrame(pose_data, columns=["x", "y", "z", "rw", "rx", "ry", "rz"])
        df.to_csv(os.path.join(args.output_dir, 'GT_Pose.txt'), index=False)
        break
    else:
        print("INVALID KEY")
        continue
