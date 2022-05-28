import torch
from models.BiSpecNet import BinSpecCNN
import cv2 as cv
import numpy as np
import torchaudio
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from torch.optim import Adam


def wavToSpec(wavname):
	waveform, sample_rate = torchaudio.load(
		filepath=wavname, frame_offset=10000, num_frames=16000)
	if len(waveform[0]) == 0:
		print("ERROR")
		return

	specgram = torchaudio.transforms.Spectrogram()(waveform)
	if specgram.size(2) != 81:
		print("WARN")
		return
	image_arr = specgram.log2()[0, :, :].numpy()
	filepath = '1.png'
	plt.imsave(fname=filepath, arr=image_arr, format='png', cmap='gray')


def read_img():
	image = '1.png'
	pic = cv.imread(image, cv.IMREAD_GRAYSCALE)
	transf = transforms.ToTensor()
	imageTensor = transf(pic)
	data = imageTensor.reshape((1, 1, 201, 201))
	return data

wavToSpec(wavname)
data = read_img()

model = BinSpecCNN(classCount=10)
model = torch.load