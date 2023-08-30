import torch.optim as op
import re
import base64
from io import BytesIO
import imageio
import cv2
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import matplotlib
matplotlib.use('TkAgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predictLetter(model, image):
    invImage = cv2.bitwise_not(image)

    invImage = np.rot90(invImage, k=-1)
    invImage = np.flip(invImage, axis=1)

    grayImage = np.dot(invImage[..., :3], [0.2989, 0.5870, 0.1140])

    normalizeImage = grayImage.astype(np.float32) / 255.0
    normalizeImage = np.reshape(normalizeImage, (1, 28, 28))

    tensorImage = torch.tensor(normalizeImage).unsqueeze(0)
    tensorImage = tensorImage.to(device)

    with torch.no_grad():
        output = model(tensorImage)

    _, predictedClass = output.max(1)
    predictedClass = predictedClass.item()

    predictedLetter = chr(predictedClass + ord('A'))

    return predictedLetter
