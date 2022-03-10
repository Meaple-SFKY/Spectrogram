import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import utils
import cv2 as cv

from tqdm import tqdm

# print(torch.__version__)
# print(torchaudio.__version__)

audio_path = './audio/'
data_path = './data/'
doc_path = './docs/'
img_path = './image/'
mdl_path = './model/'
src_path = './src/'
log_errPath = './logs/dataPreErr.log'
log_infoPath = './logs/dataPreInfo.log'
log_warnPath = './logs/dataPreWarn.log'
dataClasses = ['dev/', 'test/', 'train/']
source_path = '/Volumes/External/Speaker-Rec/Dataset/'


def proFilePath(sourcePath, docPath):
	for root, dirs, datasets in os.walk(sourcePath):
		if root.count('/') >= 6:
			datasets.sort()
			wavClass = root[root[:-6].rfind('/') + 1:-6] + '/'
			wavDirPath = wavClass + root[-5:]
			if not os.path.exists(docPath + wavClass):
				os.mkdir(docPath + wavClass)
			f = open(docPath + wavDirPath, 'w')
			for dataset in tqdm(datasets, desc=wavDirPath):
				if 'wav' in dataset:
					f.write(dataset + '\n')
			f.close()


def wavToSpec(wavPath, wavName, wavClass, specDir, plot):
	fileName = wavPath + wavName[:-1]
	waveform, sample_rate = torchaudio.load(filepath=fileName, frame_offset=10000, num_frames=16000)
	if len(waveform[0]) == 0:
		loggerErr = utils.getLogger(log_errPath)
		loggerErr.error(fileName)
		return
	specPath = img_path + wavClass + specDir
	if not os.path.exists(specPath):
		os.mkdir(specPath)

	specgram = torchaudio.transforms.Spectrogram()(waveform)
	loggerInfo = utils.getLogger(log_infoPath)
	loggerInfo.info(wavClass + specDir + wavName[:-1] + ', ' + "Shape of spectromgram:{}".format(specgram.size()))
	plot.imshow(specgram.log2()[0, :, :].numpy(), cmap='gray')
	plot.axis('off')
	plot.savefig(specPath + wavName[:-5] + '.png', bbox_inches='tight', pad_inches=0.0)
	plot.clf()


def SpecGenerator(dataDocDirs):
	plt.figure()

	for docDir in dataDocDirs:
		dataDocPath = doc_path + docDir
		dataWavPath = source_path + docDir
		for root, dirs, datasets in os.walk(dataDocPath):
			wavDir = ''
			begin = root.find('/') + 6
			end = root.rfind('/')
			wavDir = root[begin:end] + '/'
			datasets.sort()
			for dataset in datasets:
				if dataset[0] == '.':
					continue
				dataDocFile = open(root + dataset, 'r')
				wavPath = source_path + wavDir + dataset + '/'
				fileNames = dataDocFile.readlines()
				for fileName in tqdm(fileNames, desc=wavDir + dataset):
					wavToSpec(source_path + wavDir + dataset + '/', fileName, wavDir, dataset + '/', plot=plt)
				dataDocFile.close()
	plt.close()

def RGBToGray():
	for root, dirs, names in os.walk(img_path):
		className = os.path.join(data_path + root[-5:])
		if not os.path.exists(className) and className[-5] == 'S':
			os.mkdir(className)
			for name in tqdm(names, desc=className):
				fileName = os.path.join(root, name)
				image = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
				cv.imwrite(os.path.join(className, os.path.basename(name)), image)

if __name__ == '__main__':
	# proFilePath(source_path, doc_path)
	# SpecGenerator(dataDocDirs=dataClasses)
	# RGBToGray()
	print('OK')
