import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import pandas as pd
import math
from copy import deepcopy
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "apartment_0/habitat/mesh_semantic.ply"
path = "apartment_0/habitat/info_semantic.json"

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)
######

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

def calc_cross_product(dx1, dz1, dx2, dz2):
    return dx1*dz2 - dz1*dx2


def calc_angle(dx1,dz1,dx2,dz2):
    return math.acos((dx1*dx2+dz1*dz2)/((dx1**2+dz1**2)**0.5+(dx2**2+dz2**2)**0.5))

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

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

df = pd.read_csv('position.csv', index_col=False)
category_id = df.iloc[0,0]
df = df[['x', 'z']]
cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([df.iloc[0, 0], -1.5, df.iloc[0, 1]])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)
agent_state = agent.get_state()
sensor_state = agent_state.sensor_states['color_sensor']
print("camera pose: x y z rw rx ry rz")
print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Observation.avi', fourcc, 10.0, (512,  512))
for i in range(1, len(df)):
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    startx, startz = sensor_state.position[0], sensor_state.position[2]
    observation = sim.step('move_forward')
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']

    x, z = sensor_state.position[0], sensor_state.position[2]
    dx1, dz1 = x-startx, z-startz
    dx2, dz2 = df.iloc[i, 0]-startx, df.iloc[i, 1]-startz
    theta = calc_angle(dx1,dz1,dx2,dz2)*180/math.pi
    theta = round(theta, -1)

    cross_product = calc_cross_product(dx1, dz1, dx2, dz2)
    action = 'turn_left' if cross_product < 0 else 'turn_right'
    
    agent_state.position = np.array([startx, -1.5, startz])
    agent.set_state(agent_state)
    for j in range(int(theta)//10):
        observation = sim.step(action)
        mask = np.ones((512,512,3), dtype=np.uint8)
        mask[:,:] = [0,0,255]
        valid = (id_to_label[observation["semantic_sensor"]] == category_id).reshape((512,512,1))
        mask = np.where(valid, mask, 0)
        rgb = cv2.addWeighted(transform_rgb_bgr(observation['color_sensor']), 1.0, mask, 0.2, 0)
        cv2.imshow("RGB", rgb)
        out.write(rgb)
        cv2.waitKey(50)
    dist = (dx2**2+dz2**2)**0.5

    for j in range(int(dist/0.25)):
        observation = sim.step('move_forward')
        mask = np.ones((512,512,3), dtype=np.uint8)
        mask[:,:] = [0,0,255]
        valid = (id_to_label[observation["semantic_sensor"]] == category_id).reshape((512,512,1))
        mask = np.where(valid, mask, 0)
        rgb = cv2.addWeighted(transform_rgb_bgr(observation['color_sensor']), 1.0, mask, 0.2, 0)
        cv2.imshow("RGB", rgb)
        out.write(rgb)
        cv2.waitKey(50)

    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    print("camera pose: x y z rw rx ry rz")
    print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

out.release()
cv2.destroyAllWindows()
