
import os,sys
import numpy as np

def load_batched_data(data,batch_size,imsize=28,num_channels=1):
	'''
		Function to load data as batches
	'''
	num_samples = len(data)
	randIxs = np.random.permutation(num_samples)
	start,end =0,batch_size
	while(end<=num_samples):
		batchInputs_img = np.zeros((batch_size, imsize, imsize, num_channels))
		for batchI, origI in enumerate(randIxs[start:end]):
				batchInputs_img[batchI,:] = data[origI]		

		start += batch_size
		end += batch_size
		yield batchInputs_img


def add_noise(image, noise_typ="gauss", rows=28, cols=28, num_channels=1):
		if noise_typ == "gauss":
			mean = 0
			var = 0.1
			sigma = var**0.5
			gauss = np.random.normal(mean,sigma,(rows,cols,num_channels))
			gauss = gauss.reshape(rows,cols,num_channels)
			noisy = image + gauss
			return noisy
		elif noise_typ == "s&p":
			s_vs_p = 0.5
			amount = 0.004
			out = np.copy(image)
			# Salt mode
			num_salt = np.ceil(amount * image.size * s_vs_p)
			coords = [np.random.randint(0, i - 1, int(num_salt))
			      for i in image.shape]
			out[coords] = 1

			# Pepper mode
			num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
			coords = [np.random.randint(0, i - 1, int(num_pepper))
			      for i in image.shape]
			out[coords] = 0
			return out
		elif noise_typ == "poisson":
			vals = len(np.unique(image))
			vals = 2 ** np.ceil(np.log2(vals))
			noisy = np.random.poisson(image * vals) / float(vals)
			return noisy
		elif noise_typ =="speckle":
			gauss = np.random.randn(rows,cols,num_channels)
			gauss = gauss.reshape(rows,cols,num_channels)        
			noisy = image + image * gauss
			return noisy

def make_noisy(image):
	image = add_noise(image, noise_typ="gauss")
	image = add_noise(image, noise_typ="speckle")
	#image = add_noise(image, noise_typ="poisson")
	#image = add_noise(image, noise_typ="s&p")
	return image

