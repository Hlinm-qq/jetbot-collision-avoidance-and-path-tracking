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

# import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from PIL import Image


DIR = "./imgs"
ISSAVEIMG = False
ISSAVEIMGNOW = False
USEKEYBOARD = False
# 'left', 'none', 'right'
SAVEDIR = "dataset/l"

blocked_left = 0
blocked_right = 0
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
        Kp = 0.001

        if dist < -50:
            if dist < -100:
                blocked_left += 8
            # robot.set_motor(0.2, 0.2 - (blocked_right/100 + dist/1000))
            robot.right(Kp * abs(dist))
        elif dist > 50:
            # if dist > 100:
            # dist /= 10
            if dist > 100:
                blocked_right += 8
            # robot.set_motor(0.2 - 0.2*(blocked_right/100 + dist/1000), 0.2)
            robot.left(Kp * abs(dist))
        else:
            # if dist < -100:
            # dist /= 10
            
            robot.set_motor(0.2, 0.2)
        

def execute(obs):
    # Visualize
    global frames
    global ISSAVEIMGNOW


    frames += 1
    img = obs["img"]
    if(frames%25 == 0 and ISSAVEIMG):
        cv2.imwrite(f'{DIR}/test_img_{frames}.png', img)
    
    if(ISSAVEIMGNOW and USEKEYBOARD):
        cv2.imwrite(f'{DIR}/{SAVEDIR}/test_img_{frames}.png', img)
        ISSAVEIMGNOW = False

        # print("image save")
    dist = get_range(img, "")

    # Smooth the distance using a moving average
    recent_distances.append(dist)
    if len(recent_distances) > MAX_HISTORY:
        recent_distances.pop(0)
    smoothed_dist = np.mean(recent_distances)

    if(frames <= 0):
        step(None, 4, 0)
        
    # update every 5 frames
    if frames % 1 == 0:
        step(img, 99, smoothed_dist)
        
    reward = obs['reward']
    done = obs['done']

    print(f'\rframes:{frames}, reward:{reward}, done:{done}, dist:{dist}', end = "")

def get_range(oriImg, path):
    if(path != ""):
        oriImg = cv2.imread(path)

    img = cv2.resize(oriImg, (640, 360))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # black: ([100,30,20]) ([130,75,85])
    # red: (100,20,20) (50,50,50)

    lower_black = np.array([100,30,20])
    upper_black = np.array([130,75,85])

    mask_road = cv2.inRange(hsv, lower_black, upper_black)
    fin_img = cv2.bitwise_and(img, img, mask=mask_road)

    coord1 = cv2.findNonZero(mask_road[180:181])
    coord2 = cv2.findNonZero(mask_road[230:231])
    left1 = np.min(coord1, axis=0)
    right1 = np.max(coord1, axis=0)
    left2 = np.min(coord2, axis=0)
    right2 = np.max(coord2, axis=0)
    # print(coord.shape)
    # print(coord[:5, :])
    line_mean1 = 320
    line_mean1 = int(np.mean([left1[0][0], right1[0][0]]))
    line_mean2 = 320
    line_mean2 = int(np.mean([left2[0][0], right2[0][0]]))

    line_mean3 = int(np.mean([line_mean1, line_mean2]))
    
    # print(left1, right1)
    # print(left2, right2)
    dist = 320 - line_mean3

    if(path != ""):

        cv2.line(fin_img, (0, 180), (640, 180), (255,0,0), 2)
        cv2.line(fin_img, (0, 230), (640, 230), (255,0,0), 2)
        cv2.line(mask_road, (0, 180), (640, 200), (255,255,255), 2)
        cv2.line(mask_road, (0, 230), (640, 300), (255,255,255), 2)

        # cv2.circle(fin_img, (320, 320), 10,(255,215,0), 2)
        cv2.circle(fin_img, (left1[0][0], left1[0][1]+180), 10, (255, 255, 255), 2)
        cv2.circle(fin_img, (right1[0][0], right1[0][1]+180), 10, (255, 255, 255), 2)
        cv2.circle(fin_img, (line_mean1, 180), 10, (255,215,0), 2)
        cv2.circle(fin_img, (left2[0][0], left2[0][1]+230), 10, (255, 255, 255), 2)
        cv2.circle(fin_img, (right2[0][0], right2[0][1]+230), 10, (255, 255, 255), 2)
        cv2.circle(fin_img, (line_mean2, 230), 10, (255,215,0), 2)
        cv2.circle(fin_img, (line_mean3, 205), 10, (0,215,0), 2)

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
        # print('alphanumeric key {0} pressed'.format(
        #     key.char))
        # print(key.char == 'a' or key.char == "A")
        if(key.char == 'l'):
            ISSAVEIMGNOW = True
            SAVEDIR = "left"
        elif(key.char == 'n'):
            ISSAVEIMGNOW = True
            SAVEDIR = "none"
        elif(key.char == 'r'):
            ISSAVEIMGNOW = True
            SAVEDIR = "right"
    except AttributeError:

        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def cleanup_before_exit(signum, frame):
    print("\nCleanup before exit...")
    robot.stop()
    time.sleep(0.5)
    # Add your cleanup code here
    print("Exiting...")
    sys.exit(0)

if __name__ == "__main__":
    # # imgPath = "./imgs/test_img_1200.png"
    # imgPath = "./dataset/left/test_img_71/img.png"

    # img = cv2.imread(imgPath)
    # img2 = Image.open(imgPath)

    # get_range(None, imgPath)

    # Register the signal handler for SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, cleanup_before_exit)
    signal.signal(signal.SIGINT, cleanup_before_exit)

    # # Start the key listener thread
    # key_listener_thread = threading.Thread(target=key_listener)
    # key_listener_thread.daemon = True
    # key_listener_thread.start()

    # listener = keyboard.Listener(
    #                             on_press=on_press, 
    #                             on_release=on_release)
    # listener.start()

    # while True:
    #     None
    

frames = 0
robot = Robot()
env = Env()
env.run(execute)

    
