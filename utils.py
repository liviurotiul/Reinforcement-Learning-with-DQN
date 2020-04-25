# import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

def convert2bw(image): #numpy array LWC
	length, width, _ = np.shape(image)
	return_image = np.zeros((length, width, 1))
	for i in range(len(image)):
		for j in range(len(image[0])):
			pixel = image[i][j]
			if pixel[0] == 255:
				# print(pixel)
				return_image[i][j][0] = 1
	return return_image

def show_bw_image(image):
	length, width, _ = np.shape(image)
	return_image = np.zeros((length, width, 3))
	for i in range(len(image)):
		for j in range(len(image[0])):
			pixel = image[i][j]
			if pixel[0]:
				return_image[i][j][0:3] = 1
	return return_image

def mask(x):
	if x is not None:
		return True
	return False



def show(img, RGB):
	print(np.shape(img))
	if RGB:
		npimg = img.squeeze(0).cpu().numpy()
		npimg = np.transpose(npimg, (1,2,0))
		plt.imshow(npimg, interpolation='nearest')
		plt.show()
		input("salut")
		return
	npimg = img.squeeze(0).cpu().numpy()
	npimg = np.transpose(npimg, (1,2,0))
	plt.imshow(np.concatenate([npimg, npimg, npimg], axis=2), interpolation='nearest')
	plt.show()
	input("salut")


""" Debug utils.
"""
import gc
from collections import defaultdict, OrderedDict
import torch
import numpy as np
def count_tensors(numpy=False):
	stuff = defaultdict(int)
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (
				hasattr(obj, "data") and torch.is_tensor(obj.data)
			):
				stuff[f"{type(obj)} {obj.size()}"] += 1
			if numpy:
				if isinstance(obj, np.ndarray):
					stuff[f"{type(obj)} {obj.shape}"] += 1
		except:
			pass
	stuff = OrderedDict(
		sorted(stuff.items(), key=lambda kv: kv[1], reverse=True)
	)
	print(80 * "-")
	print(f"Found {sum(stuff.values())} items in memory.")
	print(80 * "-")
	for k, v in stuff.items():
		print(f"{k:60} {v} items.")
	print(80 * "-")