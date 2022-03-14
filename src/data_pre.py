import os
import torch
import torchaudio
import utils
import cv2 as cv
import matplotlib.pyplot as plt

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
dataClasses = ['train', 'dev', 'test']
source_path = '/Volumes/External/Speaker-Rec/Dataset/'


def proFilePath(sourcePath, docPath):
	for root, dirs, datasets in os.walk(sourcePath):
		classes = os.path.basename(root)
		if classes in dataClasses:
			doc_wav = os.path.join(doc_path, classes)
			if not os.path.exists(doc_wav):
				os.mkdir(doc_wav)
			for dir_item in dirs:
				sub_class = os.path.join(source_path, classes, dir_item)
				for person, _, wav_samples in os.walk(sub_class):
					f = open(os.path.join(doc_wav, os.path.basename(person)), 'w')
					for wav_sample in tqdm(wav_samples, desc=person):
						if wav_sample != '.DS_Store':
							f.write(os.path.join(person, wav_sample) + '\n')
					f.close()

def wavToSpec(wavname, wavdir, wav_class):
	waveform, sample_rate = torchaudio.load(filepath=wavname, frame_offset=10000, num_frames=16000)
	if len(waveform[0]) == 0:
		loggerErr = utils.getLogger(log_errPath)
		loggerErr.error(wavname)
		return
	specPath = os.path.join(img_path, wavdir)
	if not os.path.exists(specPath):
		os.mkdir(specPath)

	specgram = torchaudio.transforms.Spectrogram()(waveform)
	if specgram.size(2) != 81:
		loggerWarn = utils.getLogger(log_warnPath)
		loggerWarn.warning(wavname + ', ' + "Shape of spectromgram:{}".format(specgram.size()))
		return
	loggerInfo = utils.getLogger(log_infoPath)
	loggerInfo.info(wavname + ', ' + "Shape of spectromgram:{}".format(specgram.size()))
	image_arr = specgram.log2()[0, :, :].numpy()
	filepath = os.path.join(img_path, wavdir, os.path.basename(wavname)[:-3] + 'png')
	plt.imsave(fname=filepath, arr=image_arr, format='png', cmap='gray')


def SpecGenerator(dataDocDirs):

	for docDir in dataDocDirs:
		dataDocPath = os.path.join(doc_path, docDir)
		for root, _, datasets in os.walk(dataDocPath):
			class_name = os.path.basename(root)
			datasets.sort()
			for dataset in datasets:
				f = open(os.path.join(root, dataset), 'r')
				wavpaths = f.readlines()
				f.close()
				for wavfile in tqdm(wavpaths, desc=dataset):
					wavToSpec(wavfile[:-1], dataset, class_name)

if __name__ == '__main__':
	# proFilePath(source_path, doc_path)
	# SpecGenerator(dataDocDirs=dataClasses)
	print('OK')
