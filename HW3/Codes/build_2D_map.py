import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from RRT import RRT
import pandas as pd


def mouse_handler(event, x, y, flags, data): 
    if event == cv2.EVENT_LBUTTONDBLCLK:
        drawimg = data['image'].copy()
        data['start'] = [x, y]
        cv2.circle(drawimg, (x, y), 5, (0, 0, 0), -1)
        cv2.imshow("image", drawimg)


def search_target(target: str, image):
    targetx, targety = 0, 0
    valid_color = 0
    category_id = 0
    if target == 'refrigerator':
        valid_color = [0, 0, 255]
        category_id = 67
    elif target == 'rack':
        valid_color = [133, 255, 0]
        category_id = 66
    elif target == 'cushion':
        valid_color = [92, 9, 255]
        category_id = 29
    elif target == 'lamp':
        valid_color = [20, 150, 160]
        category_id = 47
    elif target == 'cooktop':
        valid_color = [224, 255, 7]
        category_id = 32
    else:
        print('No target')
        exit()
    valid = np.mean(np.where(np.all(image == valid_color, axis=-1)), axis=1)
    print(valid)
    targety, targetx = valid
    return int(targetx), int(targety), category_id

def convert_pixel_to_coordinate(x, y, image, category_id):
    rgb = cv2.imread(image)
    pixel_base = np.where(np.all(rgb == [0, 0, 0], axis=-1))
    print(f'habitat coordinate (x, z): (2.0, 10.0)')
    print(f'image pixel (x, y): ({pixel_base[1][0]}, {pixel_base[0][0]})')
    print(f'habitat coordinate (x, z): (3.0, 10.0)')
    print(f'image pixel (x, y): ({pixel_base[1][1]}, {pixel_base[0][1]})')
    ratio = pixel_base[1][1] - pixel_base[1][0]
    position = []
    for i in range(len(x)):
        x[i] = 2.0 + (x[i]-pixel_base[1][0]) / ratio
        y[i] = 10.0 - (y[i]-pixel_base[0][0]) / ratio
        position.append([category_id, x[i], y[i]])
    df = pd.DataFrame(position, columns=["id", "x", "z"])
    df.to_csv('position.csv', index=False)
    return x, y

if __name__ == '__main__':
    points = np.load('semantic_3d_pointcloud/point.npy')
    colors = np.load('semantic_3d_pointcloud/color0255.npy')
    floor_color = np.array([255, 194, 7])
    ceiling_color = np.array([8, 255, 214])
    rug_color = np.array([255, 153, 0])
    mat_color = np.array([250, 10, 15])

    valid = ((colors != floor_color) &
        (colors != ceiling_color) &
        (colors != rug_color) &
        (colors != mat_color)).any(-1)
    points = points[valid] * 10000.0 / 255.0
    colors = colors[valid]
    valid = ((points[:,1] < -0.5) & (points[:,1] > -1.1)).reshape(-1)
    points = points[valid]
    colors = colors[valid]

    print(f'Our selected target list: refrigerator, rack, cushion, lamp, cooktop')
    print(f'Input your target: ')
    target = input()
    print(f'Heading to {target}')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    # o3d.visualization.draw_geometries([pcd])

    x = points[:,0].reshape(-1)
    z = points[:,2].reshape(-1)
    fig, ax = plt.subplots()
    ax = plt.scatter(x,z,s=7,c=colors/255.0)
    ax = plt.scatter([2.0,3.0],[10.0,10.0],c='k',s=1) # for pixel to habit
    plt.axis('off')
    plt.axis("equal")
    plt.savefig("Map/map.png", bbox_inches='tight',pad_inches = 0)
    # plt.show()

    image = cv2.imread('Map/map.png')
    image_gray = cv2.imread('Map/map.png', 0)
    data = {}
    data['image'] = image
    data['start'] = 0
    data['target'] = [0, 0]
    data['target'][0], data['target'][1], category_id = search_target(target, image)
    cv2.circle(image, tuple(data['target']), 5, (0, 0, 0), -1)

    cv2.imshow('image', image)
    cv2.setMouseCallback("image", mouse_handler, data)
    print(data['start'])
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(data['start'])

    route = RRT(
        start=data['start'],
        target=data['target'],
        image='Map/map.png',
        target_sample_rate=5,
        step_size=10
    )
    route.planning()
    num_node = len(route.node_list)
    print(len(route.node_list[num_node-1].parent_x))
    print(len(route.node_list[num_node-1].parent_y))
    convert_pixel_to_coordinate(
        route.node_list[num_node-1].parent_x.copy(),
        route.node_list[num_node-1].parent_y.copy(),
        'Map/map.png',
        category_id
    )