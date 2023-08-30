import torch.optim as op
import re
import base64
from io import BytesIO
from model.model import emnistNet
from model.predicting import predictLetter
import cv2
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import matplotlib
matplotlib.use('TkAgg')


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("letterRecognition.html")


@app.route('/predictLetter', methods=['POST'])
def predictLetterRoute():
    output = ""
    model = emnistNet()
    model.load_state_dict(torch.load('./model/emnist_model.pt'))
    model.eval()

    if request.method == 'POST':
        image_data = re.sub('^data:image/.+;base64,', '',
                            request.form['imageBase64'])
        decoded_image = base64.b64decode(image_data)
        image = np.array(Image.open(BytesIO(decoded_image)))
        output = predictLetter(model, image)
        print(output)

    return jsonify({"output": output})


if __name__ == '__main__':
    app.run(debug=True)
