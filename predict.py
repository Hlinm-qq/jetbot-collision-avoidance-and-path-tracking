import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from unet import UNet
# from utils.utils import mask_to_image

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    # img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    # img = img.unsqueeze(0)
    # img = img.to(device=device, dtype=torch.float32)
    r, g, b = full_img.split()
    img = Image.merge("RGB", (b, g, r))
    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=TF.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.unsqueeze(0).to(device=device)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def predict(img, scale_factor=1, out_threshold=0.5, model_path=""):
    # load model
    net = UNet(n_channels=3, n_classes=3, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1, 2])
    net.load_state_dict(state_dict)

    # Load the image
    # img = Image.open(img_path).convert("RGB")


    mask = predict_img(net=net, 
                      full_img=img,
                      device=device,
                      scale_factor=scale_factor,
                      out_threshold=out_threshold)
    # plot_img_and_mask(img, mask)
    return robot_control(mask)

def isPath(pathMask):
    pass

def isObstacle(obstacleMask):
    pass

def robot_control(mask):
    img = mask
    flippedImg = cv2.flip(img, 0)

    return mask==1, mask==2
    pass

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
