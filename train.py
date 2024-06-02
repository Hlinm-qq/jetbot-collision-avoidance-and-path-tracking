from jetbotSim import Robot, Env
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import sys
import signal
import time
import threading
from pynput import keyboard

import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from PIL import Image
from predict import *

DIR = "./imgs"
ISSAVEIMG = False
ISSAVEIMGNOW = False
USEKEYBOARD = True
DATASETDIR = "dataset2"
# 'left', 'none', 'right'
SAVEDIR = DATASETDIR + "/left"

blocked_left = 0
blocked_right = 0
model_path = "./unet/checkpoints/checkpoint_epoch4.pth"
recent_distances = []  # List to store recent distance measurements
MAX_HISTORY = 4

if not os.path.exists(DIR):
    os.makedirs(DIR)

def step(img, action, dist=-1):
    global robot
    global blocked_left, blocked_right

    if blocked_left: 
        blocked_left -= 1
    if blocked_right: 
        blocked_right -= 1

    if action == 0:
        robot.set_motor(0.5, 0.5)
    elif action == 1:
        robot.set_motor(0.2, 0.)
    elif action == 2:
        robot.set_motor(0., 0.2)
    elif action == 3:
        robot.stop()
    elif action == 4:
        robot.reset()
    elif action == 99:
        Kp = 0.0005

        if dist < -10:
            # if dist < -100:
            #     blocked_left += 8
            # robot.set_motor(0.2, 0.2 - (blocked_right/100 + dist/1000))
            robot.right(Kp * abs(dist))
        elif dist > 10:
            # if dist > 100:
            # dist /= 10
            # if dist > 100:
            #     blocked_right += 8
            # robot.set_motor(0.2 - 0.2*(blocked_right/100 + dist/1000), 0.2)
            robot.left(Kp * abs(dist))
        else:
            # if dist < -100:
            # dist /= 10
            
            robot.set_motor(0.2, 0.2)

def execute(obs):
    global frames
    global ISSAVEIMGNOW

    frames += 1
    img = obs["img"]
    if(frames % 25 == 0 and ISSAVEIMG):
        cv2.imwrite(f'{DIR}/test_img_{frames}.png', img)
    
    if(ISSAVEIMGNOW and USEKEYBOARD):
        cv2.imwrite(f'{DIR}/{SAVEDIR}/test_img_{frames}.png', img)
        ISSAVEIMGNOW = False

    dist = get_range(img, "")

    # Smooth the distance using a moving average
    recent_distances.append(dist)
    if len(recent_distances) > MAX_HISTORY:
        recent_distances.pop(0)
    smoothed_dist = np.mean(recent_distances)

    reward = obs['reward']
    done = obs['done']
    if(frames <= 0):
        step(None, 4, 0)

    step(img, 99, smoothed_dist)

    print(f'\rframes:{frames}, reward:{reward}, done:{done}, dist:{dist}', end="")

def get_range(image_bgr, path):
    global frames

    if(path != ""):
        image_bgr = cv2.imread(path)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Create a PIL Image object from the RGB image
    ori_img = Image.fromarray(image_rgb)

    # print(image_rgb.shape)

    path_img, obstacle_img = predict(ori_img, model_path=model_path)

    # Convert the PIL Image object to a NumPy array
    path_img = np.array(path_img)

    # # Initialize the mask array with all white pixels
    # path_mask = np.ones((path_img.shape[0], path_img.shape[1], 3), dtype=np.uint8) * 255

    # # Iterate over the boolean array and set black pixels where values are True
    # for y in range(path_img.shape[0]):
    #     for x in range(path_img.shape[1]):
    #         if path_img[y, x]:
    #             path_mask[y, x] = [0, 0, 0]  # Set black pixel

    # Initialize the mask array with all white pixels and set black pixels where values are True
    path_mask = np.where(path_img[:, :, None], [0, 0, 0], [255, 255, 255]).astype(np.uint8)

    # path_img = path_img.astype(int)
    # obstacle_img = obstacle_img.astype(int)

    # print(path_img.shape)
    # print(type(path_img)) # <class 'numpy.ndarray'>

    # Convert the PIL Image object to a NumPy array
    # path_img = cv2.cvtColor(path_img, cv2.COLOR_RGB2BGR)
    # obstacle_img = cv2.cvtColor(obstacle_img, cv2.COLOR_RGB2BGR)
    # obstacle_img = np.array(obstacle_img)


    img = cv2.resize(path_mask, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([10, 10, 10])

    mask_road = cv2.inRange(hsv, lower_black, upper_black)
    fin_img = cv2.bitwise_and(img, img, mask=mask_road)

    # plt.imshow(hsv)
    # plt.show()

    coord1 = cv2.findNonZero(mask_road[80:81])
    coord2 = cv2.findNonZero(mask_road[100:101])
    left1 = np.min(coord1, axis=0)
    right1 = np.max(coord1, axis=0)
    left2 = np.min(coord2, axis=0)
    right2 = np.max(coord2, axis=0)
    # print(coord.shape)
    # print(coord[:5, :])
    line_mean1 = 64
    line_mean1 = int(np.mean([left1[0][0], right1[0][0]]))
    line_mean2 = 64
    line_mean2 = int(np.mean([left2[0][0], right2[0][0]]))

    line_mean3 = int(np.mean([line_mean1, line_mean2]))
    
    print(left1, right1)
    print(left2, right2)
    dist = 64 - line_mean3

    if(path != ""):
        cv2.line(fin_img, (0, 80), (128, 80), (255,0,0), 2)
        cv2.line(fin_img, (0, 100), (128, 100), (255,0,0), 2)
        cv2.line(mask_road, (0, 80), (128, 80), (255,255,255), 2)
        cv2.line(mask_road, (0, 100), (128, 100), (255,255,255), 2)

        # cv2.circle(fin_img, (320, 320), 10,(255,215,0), 2)
        cv2.circle(fin_img, (left1[0][0], left1[0][1]+80), 2, (255, 255, 255), 2)
        cv2.circle(fin_img, (right1[0][0], right1[0][1]+80), 2, (255, 255, 255), 2)
        cv2.circle(fin_img, (line_mean1, 80), 2, (255,215,0), 2)
        cv2.circle(fin_img, (left2[0][0], left2[0][1]+100), 2, (255, 255, 255), 2)
        cv2.circle(fin_img, (right2[0][0], right2[0][1]+100), 2, (255, 255, 255), 2)
        cv2.circle(fin_img, (line_mean2, 80), 2, (255,215,0), 2)
        cv2.circle(fin_img, (line_mean3, 100), 2, (0,215,0), 2)

        # print(left, right)
        print(f"dist: {dist}")

        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask_road)
        plt.subplot(1, 2, 2)
        plt.imshow(fin_img)
        plt.show()

    return dist

def on_press(key):
    global ISSAVEIMGNOW
    global SAVEDIR
    try:
        if(key.char == 'l'):
            ISSAVEIMGNOW = True
            SAVEDIR = DATASETDIR + "/left"
        elif(key.char == 'n'):
            ISSAVEIMGNOW = True
            SAVEDIR = DATASETDIR + "/none"
        elif(key.char == 'r'):
            ISSAVEIMGNOW = True
            SAVEDIR = DATASETDIR + "/right"
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        return False

def cleanup_before_exit(signum, frame):
    print("\nCleanup before exit...")
    robot.stop()
    time.sleep(0.5)
    print("Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    # imgPath = "./imgs/test_img_3175.png"
    # imgPath = "./dataset/left/test_img_71/img.png"

    # img = cv2.imread(imgPath)
    # img2 = Image.open(imgPath)

    # get_range(None, imgPath)

    signal.signal(signal.SIGTERM, cleanup_before_exit)
    signal.signal(signal.SIGINT, cleanup_before_exit)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()


frames = 0
robot = Robot()
env = Env()
env.run(execute)