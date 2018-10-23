
# This script is for training source tasks, transfer(full and partial)


import itertools
import numpy as np
import tensorflow as tf
import argparse
import time
import os
import cPickle
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from model.disen_task_vae import ConvVAE
from utils.tools import *
from utils.loader import *
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def sample_minibatch(dataset,batch_size=256):
  indices = np.random.choice(range(len(dataset)), batch_size, replace=False)
  return indices

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_epochs', type=int, default=30, help='training epochs')
	parser.add_argument('--display_step', type=int, default=20, help='display step')
	parser.add_argument('--checkpoint_step', type=int, default=5, help='checkpoint step')
	parser.add_argument('--task', type=str, default="autoencoding", help='autoencoding|denoising|normal|edge_texture|depth_euclidean')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--z_dim', type=int, default=20, help='z dim')
	parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
	parser.add_argument('--dataset', type=str, default='taskonomy', help='dataset')
	parser.add_argument('--restore',type=int, default=0, help='restore')
	parser.add_argument('--transfer', type=bool, default=False,help="")
	parser.add_argument('--source_task', type=str, default="autoencoding")
	parser.add_argument('--target_task', type=str, default="denoising")
	parser.add_argument('--mode', type=str, default="train",help="train|test|mi|edge_detection") # test or train or edge-detection
	parser.add_argument('--remove_dims',type=str,default='')
	parser.add_argument('--gpu',type=str,default='0')
	args = parser.parse_args()

	#set gpu
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	if(args.mode=="train"):
		return train(args)
	elif(args.mode=="test"):
		return test(args)
	elif(args.mode=="mi"):
		return MI(args)
	elif(args.mode=="edge-detection"):
		return edge_detection_kl(args)

def vec2str(a):
	s=""
	for i in range(a.shape[0]):
		s = s+str(a[i])
	return s

def str2vec(s):
	l = []
	for i in s.split(","):
		l.append(int(i))
	return l

def train(args):
	learning_rate = args.learning_rate
	batch_size = args.batch_size
	training_epochs = args.training_epochs
	display_step = args.display_step
	checkpoint_step = args.checkpoint_step 			# save training results every check point step
	z_dim = args.z_dim 								# number of latent variables.

	#set the selector matrix
	A = np.zeros(shape=(z_dim,z_dim)).astype(np.float32)
	count=0
	if (args.remove_dims!=''):
		for i in range(z_dim):
		    if i in str2vec(args.remove_dims):
		        continue
		    else:
		        A[count][i] = 1
		        count += 1 

	transfer_dims = args.remove_dims
	if(transfer_dims==''):
		A = np.identity(args.z_dim, dtype=np.float32)
		transfer_dims = "full"
		print("Full Transfer")
	if args.transfer==False:
		print ("no tranfer")
		print ("currently training the task : "+args.task)
	else:
		print ("source task is "+ args.source_task)
		print ("target task is "+ args.target_task)

	if args.dataset=="mnist":
		dataset_x = np.reshape(mnist.train.images, [-1, 28, 28, 1])
		dataset_y = np.reshape(mnist.train.images, [-1, 28, 28, 1])
	elif args.dataset=="taskonomy":
		dataset_x=np.load("./preprocess/autoencoding.npy")
		if(args.transfer==True):
			if args.target_task=="denoising":
				dataset_x=np.load("./preprocess/denoising.npy")
				dataset_y=np.load("./preprocess/autoencoding.npy")
			else:
				dataset_y=np.load("./preprocess/"+args.target_task+".npy")
		else:
			if args.task=="denoising":
				dataset_x=np.load("./preprocess/denoising.npy")
				dataset_y=np.load("./preprocess/autoencoding.npy")
			else:
				dataset_y = np.load("./preprocess/"+args.task+".npy")


	if (args.transfer==True):
		target_checkpoint_dir = "./checkpoints/transfers/"+str(args.z_dim)+"/"+args.source_task+"->"+args.target_task+"/"+transfer_dims
		source_checkpoint_dir = "./checkpoints/source_tasks/"+str(args.z_dim)+"/"+args.source_task
		out_dir = "./out/transfers/"+str(args.z_dim)+"/"+args.source_task+"->"+args.target_task+str(args.z_dim)+"_"+transfer_dims
		target_checkpoint_path = target_checkpoint_dir+"/"+args.source_task+"->"+args.target_task+str(args.z_dim)+"_"+transfer_dims+'.ckpt'
		ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
		
		dataset_x=dataset_x[int(0.64*len(dataset_x)):int(0.8*len(dataset_x))]
		dataset_y=dataset_y[int(0.64*len(dataset_y)):int(0.8*len(dataset_y))]

	else:
		target_checkpoint_dir="./checkpoints/source_tasks/"+str(args.z_dim)+"/"+args.task
		out_dir="./out/source_tasks/"+str(args.z_dim)+"/"+ args.task
		ckpt=None
		target_checkpoint_path=target_checkpoint_dir+"/"+args.task+"_"+str(args.z_dim)+'_model.ckpt'

		dataset_x=dataset_x[0:int(0.64*len(dataset_x))]
		dataset_y=dataset_y[0:int(0.64*len(dataset_y))]		

	dataset_x=dataset_x.astype(np.float32)
	dataset_y=dataset_y.astype(np.float32)
	min_x=np.min(dataset_x)
	max_x=np.max(dataset_x)
	min_y=np.min(dataset_y)
	max_y=np.max(dataset_y)
	dataset_x=(dataset_x-min_x)/(max_x-min_x)
	dataset_y=(dataset_y-min_y)/(max_y-min_y)
	imsize  = dataset_y[0].shape[1]  
	num_channels_x= dataset_x[0].shape[-1]
	num_channels_y= dataset_y[0].shape[-1]
	
	# print(target_checkpoint_dir)

	if not os.path.exists(target_checkpoint_dir):
		os.makedirs(target_checkpoint_dir)

	vae = ConvVAE(args=args, A=A.T,num_channels_x=num_channels_x,num_channels_y=num_channels_y,imsize=imsize)

	train_set=(dataset_x,dataset_y)
	n_samples = train_set[0].shape[0]
	print ("x shape ", dataset_x.shape)
	print ("y shape ", dataset_y.shape)

	if(args.transfer==True):
		restore = True
	else:
		restore = args.transfer

	if ckpt and restore:
		vae.load_model(source_checkpoint_dir)
	if ckpt==False and restore==True:
		print ("no saved model to load")

	for epoch in range(training_epochs):
		avg_cost = 0.
		reconstr_loss=0
		kl_loss=0
		disc_loss=0
		# img_d_loss=0
		# img_g_loss=0
		total_batch = int(n_samples / batch_size)
		for i in range(total_batch):
			indices = sample_minibatch(train_set[0], args.batch_size)
			x= train_set[0][indices]
			y= train_set[1][indices]
			vae_tr_loss, recons_cost,kl, disc = vae.partial_fit(x,y)
			avg_cost+=vae_tr_loss
			reconstr_loss+=recons_cost
			kl_loss+=kl
			disc_loss+=disc
			if(i%display_step==0):
				print "Epoch:", '%04d' % (epoch+1), \
					"batch:", '%04d' % (i), \
				    "vae_cost:" , "{:.6f}".format(vae_tr_loss), \
				    "reconstr_loss =", "{:.6f}".format(reconstr_loss), \
				    "kl_loss =", "{:.6f}".format(kl), \
				    "disc_loss =", "{:.6f}".format(disc)
					# "d_loss = ", "{:.6f}".format(img_d_loss), \
					# "g_loss = ", "{:.6f}".format(img_g_loss) 

			# img_d_loss+=d_loss
			# img_g_loss+=g_loss
		avg_cost = avg_cost/ n_samples * batch_size
		reconstr_loss=reconstr_loss/n_samples*batch_size
		kl_loss=kl_loss/n_samples*batch_size
		disc_loss=disc_loss/n_samples*batch_size
		# img_d_loss=img_d_loss/n_samples*batch_size
		# img_g_loss=img_g_loss/n_samples*batch_size
		# Display logs per epoch step
		# if i % display_step == 0:
			

		if not os.path.exists(out_dir):
		  	os.makedirs(out_dir)
		recons = vae.reconstruct(x)
		save_images(y, image_manifold_size(y.shape[0]),out_dir+'/epoch_{}_actual.png'.format(str(epoch+1)))
		save_images(recons, image_manifold_size(recons.shape[0]),out_dir+'/epoch_{}_recons.png'.format(str(epoch+1)))
		if args.transfer==False:
			print ("current task : "+ args.task)
		else:
			print ("source task :"+args.source_task+" target task: "+args.target_task)
		print "Epoch:", '%04d' % (epoch+1), \
				"batch:", '%04d' % (i), \
			    "vae_cost:" , "{:.6f}".format(avg_cost), \
			    "reconstr_loss =", "{:.6f}".format(reconstr_loss), \
			    "kl_loss =", "{:.6f}".format(kl_loss), \
			    "disc_loss =", "{:.6f}".format(disc_loss)
				# "d_loss = ", "{:.6f}".format(img_d_loss), \
				# "g_loss = ", "{:.6f}".format(img_g_loss)  

		vae.save_model(target_checkpoint_path, epoch)
	return vae


def test(args):  
	learning_rate = args.learning_rate
	batch_size = args.batch_size
	training_epochs = args.training_epochs
	display_step = args.display_step
	checkpoint_step = args.checkpoint_step 			# save training results every check point step
	z_dim = args.z_dim 								# number of latent variables.

	#set the selector matrix
	A = np.zeros(shape=(z_dim,z_dim)).astype(np.float32)
	count=0
	if (args.remove_dims!=''):
		for i in range(z_dim):
		    if i in str2vec(args.remove_dims):
		        continue
		    else:
		        A[count][i] = 1
		        count += 1 

	if(args.remove_dims==''):
		A = np.identity(args.z_dim, dtype=np.float32)
		print("Full Transfer")

	transfer_dims = args.remove_dims
	if(transfer_dims==''):
		transfer_dims = "full"

	if args.dataset=="mnist":
		dataset_x = np.reshape(mnist.train.images, [-1, 28, 28, 1])
		dataset_y = np.reshape(mnist.train.images, [-1, 28, 28, 1])
	elif args.dataset=="taskonomy":
		dataset_x=np.load("./preprocess/autoencoding.npy")
		if(args.transfer==True):
			if args.target_task=="denoising":
				dataset_x=np.load("./preprocess/denoising.npy")
				dataset_y=np.load("./preprocess/autoencoding.npy")
			else:
				dataset_y=np.load("./preprocess/"+args.target_task+".npy")
		else:
			if args.task=="denoising":
				dataset_x=np.load("./preprocess/denoising.npy")
				dataset_y=np.load("./preprocess/autoencoding.npy")
			else:
				dataset_y = np.load("./preprocess/"+args.task+".npy")

	dataset_x=dataset_x[int(0.8*len(dataset_x)):]
	dataset_y=dataset_y[int(0.8*len(dataset_y)):]
	dataset_x=dataset_x.astype(np.float32)
	dataset_y=dataset_y.astype(np.float32)
	min_x=np.min(dataset_x)
	max_x=np.max(dataset_x)
	min_y=np.min(dataset_y)
	max_y=np.max(dataset_y)
	dataset_x=(dataset_x-min_x)/(max_x-min_x)
	dataset_y=(dataset_y-min_y)/(max_y-min_y)
	imsize  = dataset_y[0].shape[1]  
	num_channels_x= dataset_x[0].shape[-1]
	num_channels_y= dataset_y[0].shape[-1]

	if (args.transfer==True):
		target_checkpoint_dir = "./checkpoints/transfers/"+str(args.z_dim)+"/"+args.source_task+"->"+args.target_task+"/"+transfer_dims
		source_checkpoint_dir = "./checkpoints/source_tasks/"+str(args.z_dim)+"/"+args.source_task
		out_dir = "./out/transfers/"+str(args.z_dim)+"/"+args.source_task+"->"+args.target_task+str(args.z_dim)+"_"+transfer_dims
		target_checkpoint_path = target_checkpoint_dir+"/"+args.source_task+"->"+args.target_task+str(args.z_dim)+"_"+transfer_dims+'.ckpt'
		ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
	else:
		target_checkpoint_dir="./checkpoints/source_tasks/"+str(args.z_dim)+"/"+args.task
		out_dir="./out/source_tasks/"+str(args.z_dim)+"/"+ args.task
		ckpt=None
		target_checkpoint_path=target_checkpoint_dir+"/"+args.task+"_"+str(args.z_dim)+'_model.ckpt'

	if(args.transfer):
		test_results_file="logging/transfers/" +str(args.z_dim)+"/"+ args.source_task+"->"+args.target_task+"_transfer.txt"
	else:
		test_results_file = "logging/source_tasks/"+str(args.z_dim)+"/"+args.task

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	

	vae = ConvVAE(args=args, A=A.T,imsize=imsize,num_channels_x=num_channels_x,num_channels_y=num_channels_y)

	ckpt = tf.train.get_checkpoint_state(target_checkpoint_dir)
	if ckpt:
		vae.load_model(target_checkpoint_dir, load_transfer=True)
		print ("Loaded model")


	avg_recon_cost = []
	avg_vae_cost=[]

	n_batches=int(dataset_y.shape[0]/args.batch_size)
	print("Number of batches are %d"%n_batches)
	for i in range(n_batches):
		x = dataset_x[i*args.batch_size:(i+1)*args.batch_size]
		y = dataset_y[i*args.batch_size:(i+1)*args.batch_size]
		# n_samples = test_set.shape[0]
		vae_cost, reconstr_loss, kl_div= vae.test_loss(x,y,eps[i])#vae.partial_fit(x_batch,y_batch)
		x_prime = vae.reconstruct(x_batch)
		save_images(np.array(x_prime), image_manifold_size(len(x_prime)),out_dir+'/test_{}_recons.png'.format(str(i+1)))
		save_images(np.array(y), image_manifold_size(len(y)),out_dir+'/test_{}_actual.png'.format(str(i+1)))

	avg_recon_cost.append(reconstr_loss)
	avg_vae_cost.append(vae_cost)


	if remove_dims!='':
		print "removed dims: "+ args.remove_dims
	else:
		print "full transfer"
		print "average vae_cost:" , "{:.6f}".format(np.mean(vae_cost)), \
		            "average reconstr_loss =", "{:.6f}".format(np.mean(reconstr_loss)) 

	with open(test_results_file,"a") as text_file:
		if args.transfer==True:
			text_file.write("Removed dims :"+args.remove_dims+"\n")
			text_file.write("source task :"+args.source_task+"\n")
			text_file.write("target task :"+args.target_task+"\n")
		else:
			text_file.write("task :"+args.task+"\n")

		text_file.write("RECONS LOSS : "+str(np.mean(reconstr_loss))+"\n")
		text_file.write("vae cost : "+str(np.mean(vae_cost))+"\n")
		text_file.write("kl_div : "+str(np.mean(kl_div))+"\n")
		text_file.write("regularizer loss : "+str(np.mean(reg_loss))+"\n")
		text_file.write("---------------------------------"+"\n")
		text_file.close() 


def MI(args):
	learning_rate = args.learning_rate
	batch_size = args.batch_size
	training_epochs = args.training_epochs
	display_step = args.display_step
	checkpoint_step = args.checkpoint_step 			# save training results every check point step
	z_dim = args.z_dim 								# number of latent variables.

	#set the selector matrix
	A = np.zeros(shape=(z_dim,z_dim)).astype(np.float32)
	count=0
	if (args.remove_dims!=''):
		for i in range(z_dim):
		    if i in str2vec(args.remove_dims):
		        continue
		    else:
		        A[count][i] = 1
		        count += 1 

	if(args.remove_dims==''):
		A = np.identity(args.z_dim, dtype=np.float32)
		print("Full Transfer")

	transfer_dims = args.remove_dims
	if(transfer_dims==''):
		transfer_dims = "full"

	if args.dataset=="mnist":
		dataset_x = np.reshape(mnist.train.images, [-1, 28, 28, 1])
		dataset_y = np.reshape(mnist.train.images, [-1, 28, 28, 1])
	elif args.dataset=="taskonomy":
		dataset_x=np.load("./preprocess/autoencoding.npy")
		if(args.transfer==True):
			if args.target_task=="denoising":
				dataset_x=np.load("./preprocess/denoising.npy")
				dataset_y=np.load("./preprocess/autoencoding.npy")
			else:
				dataset_y=np.load("./preprocess/"+args.target_task+".npy")
		else:
			if args.task=="denoising":
				dataset_x=np.load("./preprocess/denoising.npy")
				dataset_y=np.load("./preprocess/autoencoding.npy")
			else:
				dataset_y = np.load("./preprocess/"+args.task+".npy")

	dataset_x=dataset_x[int(0.8*len(dataset_x)):]
	dataset_y=dataset_y[int(0.8*len(dataset_y)):]
	dataset_x=dataset_x.astype(np.float32)
	dataset_y=dataset_y.astype(np.float32)
	min_x=np.min(dataset_x)
	max_x=np.max(dataset_x)
	min_y=np.min(dataset_y)
	max_y=np.max(dataset_y)
	dataset_x=(dataset_x-min_x)/(max_x-min_x)
	dataset_y=(dataset_y-min_y)/(max_y-min_y)
	imsize  = dataset_y[0].shape[1]  
	num_channels_x= dataset_x[0].shape[-1]
	num_channels_y= dataset_y[0].shape[-1]

	#set all directories for saving checkpoints and saving results and logs
	if (args.transfer==True):
		target_checkpoint_dir = "./checkpoints/transfers/"+str(args.z_dim)+"/"+args.source_task+"->"+args.target_task+"/"+transfer_dims
		source_checkpoint_dir = "./checkpoints/source_tasks/"+str(args.z_dim)+"/"+args.source_task
		out_dir = "./out/transfers/"+str(args.z_dim)+"/"+args.source_task+"->"+args.target_task+str(args.z_dim)+"_"+transfer_dims
		target_checkpoint_path = target_checkpoint_dir+"/"+args.source_task+"->"+args.target_task+str(args.z_dim)+"_"+transfer_dims+'.ckpt'
		ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
	else:
		target_checkpoint_dir="./checkpoints/source_tasks/"+str(args.z_dim)+"/"+args.task
		out_dir="./out/source_tasks/"+str(args.z_dim)+"/"+ args.task
		ckpt=None
		target_checkpoint_path=target_checkpoint_dir+"/"+args.task+"_"+str(args.z_dim)+'_model.ckpt'
	
	mi_checkpoint_dir = "./checkpoints/mi/"+str(args.z_dim)+"/"
	mi_checkpoint_path = mi_checkpoint_dir+"mi_params.ckpt"


	if(args.transfer):
		mi_results_file="logging/mi/" +str(args.z_dim)+"/"+ args.source_task+"->"+args.target_task+"_transfer.txt"
	else:
		mi_results_file = "logging/source_tasks/"+str(args.z_dim)+"/"+args.task

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if not os.path.exists(mi_checkpoint_dir):
		os.makedirs(mi_checkpoint_dir)

	vae = ConvVAE(args=args, A=A.T,imsize=imsize,num_channels_x=num_channels_x,num_channels_y=num_channels_y)

	ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
	if ckpt:
		vae.load_model(source_checkpoint_dir, load_transfer=True)
		print ("Loaded model")

	ckpt = tf.train.get_checkpoint_state(mi_checkpoint_dir)
	if ckpt:
		vae.load_model(mi_checkpoint_dir, load_mi=True)
		print ("Loaded model")

	mi_loss_avg=[]
	for epoch in range(30):
		# break
		np.random.shuffle(indices)
		test_set=dataset_x[indices]
		test_y_set=dataset_y[indices]

		for i in range(test_set.shape[0]//batch_size):
		# batch_x_corrupt = test_set[i*batch_size:(i+1)*batch_size]
			batch_x = test_set[i*batch_size:(i+1)*batch_size]
			batch_y = test_y_set[i*batch_size:(i+1)*batch_size]
			vae.train_mi(batch_x, batch_y)
			mi_loss=vae.mi_value(batch_x,batch_y)
			mi_loss_avg.append(mi_loss)
			print ("mi loss for epoch: ",epoch,"is ",np.mean(mi_loss_avg))

	mi_loss_avg=[]  
	for i in range(test_set.shape[0]//args.batch_size):
		# batch_x_corrupt = test_set[i*args.batch_size:(i+1)*args.batch_size]
		batch_x = test_set[i*batch_size:(i+1)*batch_size]
		batch_y = test_y_set[i*batch_size:(i+1)*batch_size]
		vae.train_mi(batch_x, batch_y)
		mi_loss=vae.mi_value(batch_x,batch_y)
		mi_loss_avg.append(mi_loss)
		print ("final mi is : ",np.mean(mi_loss_avg) )

	with open(mi_results_file,"a") as text_file:
		# if args.transfer==True:
		text_file.write("dims-selected :"+str(args.remove_dims)+"\n")
		text_file.write("MI value: "+str(np.mean(mi_loss_avg))+"\n")
		# return (np.mean(mi_loss_avg))


def edge_detection_kl(args):
  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  z_dim = args.z_dim # number of latent variables.
  imsize  = args.imsize
  possible_combs=list(itertools.combinations([ "autoencoding","normal"],2))
  z_1=tf.placeholder(tf.float32, [None, args.z_dim])
  z_2=tf.placeholder(tf.float32, [None, args.z_dim])
  print (possible_combs)
  z1_latents=[]
  z2_latents=[]
  for comb in possible_combs:
    print comb[0],"------------------"
    print comb[1],"------------------"
    if comb[0]=="denoising":
      X1 = np.load("/DATA1/taskonomy-resized/noisy_rgb.npy").astype(np.float32)
      Y1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    elif comb[0]=="autoencoding":
      X1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
      Y1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    else:
      Y1 = np.load("/DATA1/taskonomy-resized/"+comb[0]+".npy").astype(np.float32)
      X1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)

    if len(Y1.shape)==3:
      num_channels=1
      Y1=np.reshape(Y1,[-1,Y1.shape[1],Y1.shape[2],1])
    else:
      num_channels = Y1.shape[3]
    vae = ConvVAE(args=args,num_channels=num_channels)
    if comb[1]=="denoising":
      X2 = np.load("/DATA1/taskonomy-resized/noisy_rgb.npy").astype(np.float32)
      Y2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    elif comb[1]=="autoencoding":
      X2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
      Y2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    else:
      Y2 = np.load("/DATA1/taskonomy-resized/"+comb[1]+".npy").astype(np.float32)
      X2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)

    if len(Y2.shape)==3:
      num_channels=1
      Y2=np.reshape(Y2,[-1,Y2.shape[1],Y2.shape[2],1])
    else:
      num_channels=Y2.shape[3]
    Y1=(Y1-np.min(Y1))/(np.max(Y1)-np.min(Y1))
    X1=(X1-np.min(X1))/(np.max(X1)-np.min(X1))
    Y2=(Y2-np.min(Y2))/(np.max(Y2)-np.min(Y2))
    X2=(X2-np.min(X2))/(np.max(X2)-np.min(X2))
    checkpoint_dir="./checkpoints/"+comb[0]
    vae.load_model(checkpoint_dir)
    mu1=[]
    cov1=[]
    mu2=[]
    cov2=[]
    for i in range(len(X1[7568:])):
      mu, cov = vae.get_distribution(X1[i].reshape((1,X1.shape[1],X1.shape[2],X1.shape[3])))
      mu1.append(mu)
      cov1.append(cov)
    checkpoint_dir="./checkpoints/"+comb[1]
    vae.load_model(checkpoint_dir)
    for i in range(len(X2[7568:])):
      mu, cov = vae.get_distribution(X2[i].reshape((1,X2.shape[1],X2.shape[2],X2.shape[3])))
      mu2.append(mu)
      cov2.append(cov)
    mu1,mu2,cov1,cov2=np.mean(mu1,axis=0),np.mean(mu2,axis=0),np.mean(cov1,axis=0),np.mean(cov2,axis=0)
    kl1 = kl_div(mu1, cov1, mu2, cov2)
    kl2 = kl_div(mu2, cov2, mu1, cov1)
    print ("comb = ", comb)
    print ("kl1 = ", kl1)
    print ("kl2 = ", kl2)
    with open("kl_results.txt",'a') as kl_file:
      kl_file.write("Task 1: %s, Task 2: %s"%(comb[0],comb[1]))
      kl_file.write("1->2 kl loss: %f"%kl1)
      kl_file.write("2->1 kl loss: %f"%kl2)
      kl_file.write("################################################")
      kl_file.close()

def kl_div(mu1,cov1,mu2,cov2):
	trace_term = np.sum(cov1/cov2,axis=1)
	term1=0
	term2=0
	for i in range(cov2.shape[1]):
	  term1+=np.log(cov2[0][i]+1e-8)
	  term2+=np.log(cov1[0][i]+1e-8)
	det_term = term1-term2 #np.log(cov2)-np.log(np.prod(cov1,axis=1)+)
	mean_term = np.sum(np.square(mu2-mu1)/cov2,axis=1)
	return np.mean(0.5*(trace_term+det_term+mean_term))   

if __name__ == '__main__':
  main()
