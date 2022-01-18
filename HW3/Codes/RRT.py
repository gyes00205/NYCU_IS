import cv2
import numpy as np
import math
import random


class RRT:

    class Node:

        def __init__(self, x, y) -> None:
            self.x = x
            self.y = y
            self.parent_x = []
            self.parent_y = []

    def __init__(self,
                 start,
                 target,
                 image,
                 target_sample_rate=5,
                 step_size = 10
                 ) -> None:
        self.start = start
        self.target = target
        self.target_sample_rate = target_sample_rate
        self.image = cv2.imread(image)
        self.image_gray = cv2.imread(image, 0)
        self.node_list = []
        self.height, self.width = self.image_gray.shape
        self.step_size = step_size


    def planning(self):
        self.node_list.append(self.Node(self.start[0], self.start[1]))
        self.node_list[0].parent_x.append(self.start[0])
        self.node_list[0].parent_y.append(self.start[1])

        cv2.circle(self.image, (self.start[0], self.start[1]), 5, (0, 0, 0), -1)
        cv2.circle(self.image, (self.target[0], self.target[1]), 5, (0, 0, 0), -1)

        i = 1
        while True:
            
            sample_x, sample_y = self.get_random_smaple()
            nearest_idx = self.nearest_node(sample_x, sample_y)
            nearest_x = self.node_list[nearest_idx].x
            nearest_y = self.node_list[nearest_idx].y
            if sample_x == nearest_x and sample_y == nearest_y:
                continue
            mx, my = self.get_middle_sample(nearest_x, nearest_y, sample_x, sample_y)
            dist = self.calc_dist(mx, my, self.target[0], self.target[1])
            node_collision = self.check_collision(nearest_x, nearest_y, mx, my)
            target_collision = self.check_collision(mx, my, self.target[0], self.target[1])

            new_node = self.Node(mx, my)
            new_node.parent_x = self.node_list[nearest_idx].parent_x.copy()
            new_node.parent_y = self.node_list[nearest_idx].parent_y.copy()
            new_node.parent_x.append(mx)
            new_node.parent_y.append(my)

            if (not (node_collision or target_collision)) or dist <= 15:
                self.node_list.append(new_node)

                cv2.circle(self.image, (int(mx), int(my)), 2, (0, 0, 255), -1)
                cv2.line(self.image, (int(mx), int(my)), (int(nearest_x), int(nearest_y)), (0, 255, 0), 1)
                cv2.line(self.image, (int(mx), int(my)), (self.target[0], self.target[1]), (255, 0, 0), 2)
                
                print('Found Path')
                for j in range(len(self.node_list[i].parent_x)-1):
                    cv2.line(
                        self.image,
                        (int(new_node.parent_x[j]), int(new_node.parent_y[j])),
                        (int(new_node.parent_x[j+1]), int(new_node.parent_y[j+1])),
                        (255, 0, 0),
                        2
                    )
                self.node_list[i].parent_x.append(self.target[0])
                self.node_list[i].parent_y.append(self.target[1])
                
                cv2.imwrite(f'Map/route.png', self.image)
                cv2.imshow('rrt', self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

            elif not node_collision:
                self.node_list.append(new_node)
                cv2.circle(self.image, (int(mx), int(my)), 2, (0, 0, 255), -1)
                cv2.line(self.image, (int(mx), int(my)), (int(nearest_x), int(nearest_y)), (0, 255, 0), 1)
                # cv2.imwrite(f'media/{i}.png', self.image)
                cv2.imshow('rrt', self.image)
                cv2.waitKey(1)
                i += 1
            
            else:
                del new_node

    def get_middle_sample(self, x1, y1, x2, y2):
        dist = self.calc_dist(x1, y1, x2, y2)
        if dist == 0:
            return x1, x2
        new_x = x1 + ((x2 - x1) * self.step_size / dist)
        new_y = y1 + ((y2 - y1) * self.step_size / dist)
        if new_x >= self.width or new_y >= self.height or new_x < 0 or new_y < 0:
            new_x, new_y = x2, y2
        return new_x, new_y

    def check_collision(self, x1, y1, x2, y2):
        color = []
        if (x2 == x1) and (y2 == y1):
            x = [x1]
            y = [y1]
        elif x2 == x1:
            y = list(np.arange(y1,y2+(y2-y1)/10,(y2-y1)/10))
            x = [x1]*len(y)
        elif y2 == y1:
            x = list(np.arange(x1,x2+(x2-x1)/10,(x2-x1)/10))
            y = [y1]*len(x)
        else:
            x = np.arange(x1,x2+(x2-x1)/10,(x2-x1)/10)
            y = list(((y2-y1)/(x2-x1))*(x-x1) + y1)
            x = list(x)
        for i in range(len(x)):
            x[i] = int(x[i])
            y[i] = int(y[i])
            if y[i]>=self.height or y[i]<0 or x[i]>=self.width or x[i]<0:
                return True
            color.append(self.image_gray[y[i], x[i]])
        color = np.array(color)
        if (color == 255).all():
            return False # no collision
        else:
            return True # collision
         

    def calc_dist(self, x1, y1, x2, y2):
        dist = (((x1-x2)**2)+((y1-y2)**2))**0.5
        return dist

    def nearest_node(self, x, y):
        temp_list = []
        for i in range(len(self.node_list)):
            node_x, node_y = self.node_list[i].x, self.node_list[i].y
            dist = self.calc_dist(x, y, node_x, node_y)
            temp_list.append(dist)
        return temp_list.index(min(temp_list))

    def get_random_smaple(self):
        if random.randint(0, 100) > self.target_sample_rate:
            sample_x = random.randint(0, self.width)
            sample_y = random.randint(0, self.height)
        else:
            sample_x = self.target[0]
            sample_y = self.target[1]
        return sample_x, sample_y


