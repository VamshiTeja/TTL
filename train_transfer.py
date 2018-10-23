
#This script is exclusively for training and getting the test losses 
# on full transfer of latent or completely no transfer of latent.

import numpy as np
import tensorflow as tf
import argparse
import time
import os
import cPickle
import pickle
# import cv2
from tensorflow.examples.tutorials.mnist import input_data
from model.vae_for_transfer import ConvVAE
#from utils.loader import load_batched_data
from utils.tools import *
from utils.loader import *
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = np.reshape(mnist.train.images, [-1, 28, 28, 1])[:-2000]
# test_set  = np.reshape(mnist.train.images, [-1, 28, 28, 1])[-2000:]
# test_set  = test_set[:1792]
def sample_minibatch(batch_size=256, test=False):
  if test is False:
      indices = np.random.choice(range(len(train_set)), batch_size, replace=False)
      sample = train_set[indices]
  elif test is True:
      indices = np.random.choice(range(len(test_set)), batch_size, replace=False)
      sample = test_set[indices]
  return sample

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_epochs', type=int, default=30,
                      help='training epochs')
  parser.add_argument('--display_step', type=int, default=5,
                      help='display step')
  parser.add_argument('--checkpoint_step', type=int, default=5,
                      help='checkpoint step')
  parser.add_argument('--task', type=str, default="autoencoding",
                      help='task name(denoising or autoencoding)')
  parser.add_argument('--batch_size', type=int, default=256,
                      help='batch size')
  parser.add_argument('--z_dim', type=int, default=20,
                      help='z dim')
  parser.add_argument('--learning_rate', type=float, default=0.0002,
                      help='learning rate')
  parser.add_argument('--dataset', type=str, default='mnist',
                      help='dataset')
  parser.add_argument('--imsize', type=int, default=28,
                      help='imsize')
  parser.add_argument('--num_channels', type=int, default=1,
                      help='num_channels')
  parser.add_argument('--restore',type=int, default=0, help='restore')
  parser.add_argument('--transfer', type=bool, default=True)
  parser.add_argument('--source_task', type=str, default="autoencoding")
  parser.add_argument('--target_task', type=str, default="denoising")
  parser.add_argument('--mode', type=str, default="test")
  parser.add_argument('--remove_dims',type=str,default='')
  parser.add_argument('--mi_dims',type=str,default='1')
  parser.add_argument('--gpu',type=str,default='0')
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  if(args.mode=="train"):
    return train(args)
  elif(args.mode=="test"):
    return test(args)
  elif(args.mode=="mi"):
    return MI(args)

def vec2str(a):
  s=""
  for i in range(a.shape[0]):
    s = s+str(a[i])
  return s

def str2vec(s):
  l = []
  for i in s.split(","):
    if(i!=""):
      l.append(int(i))
  return l

def train(args):

  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  z_dim = args.z_dim # number of latent variables.

  z_dim_new = z_dim - len(str2vec(args.remove_dims))
  # print ("-------------",z_dim_new)
  A = np.zeros(shape=(z_dim_new,z_dim)).astype(np.float32)
  count=0

  for i in range(z_dim):
    if i in str2vec(args.remove_dims):
      continue
    else:
      # print (count)
      A[count][i] = 1
      count += 1 

  #set the selector matrix
  if(args.remove_dims==''):
    A = np.identity(args.z_dim, dtype=np.float32)
    print("Full Tranasfer")

  transfer_dims = args.remove_dims
  if(transfer_dims==''):
    transfer_dims = "full"
  dataset = args.dataset
  imsize  = args.imsize
  
  if args.transfer==True:
    if(args.target_task=="denoising"):
      target_checkpoint_dir = "./checkpoints/A->D_transfer_mnist/"+transfer_dims
      source_checkpoint_dir = "./checkpoints/autoencoding_mnist"
      out_dir = "out/A->D_transfer/"+transfer_dims
      target_checkpoint_path = target_checkpoint_dir+"/"+"A->D_transfer_"+transfer_dims+'.ckpt'
    elif(args.target_task=="autoencoding"):
      target_checkpoint_dir = "./checkpoints/D->A_transfer/"+transfer_dims
      source_checkpoint_dir = "./checkpoints/denoising_mnist"
      out_dir = "out/D->A_transfer/"+transfer_dims
      target_checkpoint_path=target_checkpoint_dir+"/"+"D->A_transfer_"+transfer_dims+'.ckpt'
          
    ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
  else:
    target_checkpoint_dir="./checkpoints/"+args.task
    out_dir="out/"+args.task
    ckpt=None
    target_checkpoint_path=target_checkpoint_dir+"/"+args.task+'_model.ckpt'

  if not os.path.exists(source_checkpoint_dir):
    os.makedirs(source_checkpoint_dir)
  if not os.path.exists(target_checkpoint_dir):
    os.makedirs(target_checkpoint_dir)

    # A = A[:,0:8]
  vae = ConvVAE(args=args, A=A.T)
  n_samples = train_set.shape[0]

  # load previously trained model if appilcable
  if ckpt and args.restore:
    vae.load_model(source_checkpoint_dir)

  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_xs = sample_minibatch()
      if((args.target_task=="denoising" and args.transfer==True) or args.task=="denoising" ):
        noisy_batch = []
        for j in range(batch_size):
          noisy_batch.append(add_noise(batch_xs[j]))
        x = np.array(noisy_batch)
        y = batch_xs
      elif((args.target_task=="autoencoding" and args.transfer==True) or args.task=="autoencoding"):
        x = batch_xs
        y = batch_xs

      # Fit training using batch data
      vae_cost, reconstr_loss, kl_loss, disc_loss = vae.partial_fit(x,y)
      avg_cost += vae_cost / n_samples * batch_size

      # Display logs per epoch step
      if i % display_step == 0:
          print "Epoch:", '%04d' % (epoch+1), \
            "batch:", '%04d' % (i), \
            "vae_cost:" , "{:.6f}".format(vae_cost), \
            "reconstr_loss =", "{:.6f}".format(reconstr_loss), \
            "kl_loss =", "{:.6f}".format(kl_loss), \
              "disc_loss =", "{:.6f}".format(disc_loss)	

    #save reconstructions
    #recons = vae.reconstruct(batch_xs)

    if not os.path.exists(out_dir):
    	os.makedirs(out_dir)

    if((args.target_task=="denoising" and args.transfer==True) or args.task=="denoising" ):
      noisy_batch = []
      for j in range(batch_size):
        noisy_batch.append(add_noise(batch_xs[j]))
      x = np.array(noisy_batch)
      y = batch_xs
    elif((args.target_task=="autoencoding" and args.transfer==True) or args.task=="autoencoding"):
      x = batch_xs
      y = batch_xs

    recons = vae.reconstruct(x)
    save_images(x, image_manifold_size(x.shape[0]),out_dir+'/epoch_{}_actual.png'.format(str(epoch+1)))
    save_images(recons, image_manifold_size(recons.shape[0]),out_dir+'/epoch_{}_recons.png'.format(str(epoch+1)))
    # Display logs per epoch step
    print "Epoch:", '%04d' % (epoch+1), \
          "cost=", "{:.6f}".format(avg_cost)

  # save model one last time, under zero label to denote finish.
  vae.save_model(target_checkpoint_path, 0)
  return vae

def test(args):  
  z_dim = args.z_dim # number of latent variables.

  transfer_dims = args.remove_dims
  if(transfer_dims==''):
    transfer_dims = "full"

  dataset = args.dataset
  imsize  = args.imsize
  test_y_set = np.load("test_samples_without_noise.npy")

  if args.transfer==True:
    if(args.target_task=="denoising"):
      target_checkpoint_dir = "./checkpoints/A->D_transfer_mnist/"+transfer_dims
      source_checkpoint_dir = "./checkpoints/autoencoding"
      out_dir = "out/A->D_transfer/"+transfer_dims
      target_checkpoint_path = target_checkpoint_dir+"/"+"A->D_transfer_"+transfer_dims+'.ckpt'
      test_set = np.load("test_samples_without_noise.npy")
    elif(args.target_task=="autoencoding"):
      target_checkpoint_dir = "./checkpoints/D->A_transfer/"+transfer_dims
      source_checkpoint_dir = "./checkpoints/denoising"
      out_dir = "out/D->A_transfer/"+transfer_dims
      target_checkpoint_path=target_checkpoint_dir+"/"+"D->A_transfer_"+transfer_dims+'.ckpt'
      test_set = np.load("test_samples.npy")    
  else:
    if (args.task=="autoencoding"):
      test_set  = test_y_set
    target_checkpoint_dir="./checkpoints/"+args.task
    out_dir="out/"+args.task

  if(args.transfer):
    test_results_file="out/transfer.txt"
  else:
    test_results_file = "out/tasks.txt"

  z_dim_new = args.z_dim - len(str2vec(args.remove_dims))

  A = np.zeros(shape=(z_dim_new,args.z_dim)).astype(np.float32)
  count=0
  print z_dim
  for i in range(z_dim):
    if i in str2vec(args.remove_dims):
      continue
    else:
      # print (count)
      A[count][i] = 1
      count += 1 

  vae = ConvVAE(args=args, A=A.T)
  print(target_checkpoint_dir)
  ckpt = tf.train.get_checkpoint_state(target_checkpoint_dir)
  if ckpt:
    vae.load_model(target_checkpoint_dir, load_transfer=True)
  print ("Loaded model")
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  avg_recon_cost = []
  avg_vae_cost=[]

  n_batches=int(test_set.shape[0]/args.batch_size)
  print("Number of batches are %d"%n_batches)
  for i in range(n_batches):
    x_batch=[]
    y_batch=[]
    if(args.transfer==True):
      if(args.target_task=="denoising"):
        x_batch=test_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
      elif (args.target_task=="autoencoding"):
        x_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
    else:
      if(args.task=="denoising"):
        x_batch=test_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
      elif (args.task=="autoencoding"):
        x_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]

    n_samples = test_set.shape[0]
    vae_cost, reconstr_loss = vae.test_loss(x_batch,y_batch)
    x_prime=vae.reconstruct(x_batch)
    save_images(np.array(x_prime), image_manifold_size(len(x_prime)),out_dir+'/test_{}_recons.png'.format(str(i+1)))
    save_images(np.array(y_batch), image_manifold_size(len(y_batch)),out_dir+'/test_{}_actual.png'.format(str(i+1)))

    avg_recon_cost.append(reconstr_loss)
    avg_vae_cost.append(vae_cost)
    print "vae cost : ", vae_cost
    print "reconstruction loss : ",reconstr_loss

  print "average vae_cost:" , "{:.6f}".format(np.mean(vae_cost)), \
                "average reconstr_loss =", "{:.6f}".format(np.mean(reconstr_loss)) 
  with open(test_results_file,"a") as text_file:
    if args.transfer==True:
      text_file.write("Removed dims :"+args.remove_dims+"\n")
      text_file.write("source task :"+args.source_task+"\n")
      text_file.write("target task :"+args.target_task+"\n")
    else:
      text_file.write("task :"+args.task+"\n")

    text_file.write("recon loss : "+str(np.mean(reconstr_loss))+"\n")
    text_file.write("vae cost : "+str(np.mean(vae_cost))+"\n")
    text_file.write("---------------------------------"+"\n")
    text_file.close() 


def MI(args):
  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  z_dim = args.z_dim # number of latent variables.
  z_dim_new = z_dim - len(str2vec(args.mi_dims))

  transfer_dims = args.remove_dims

  if(transfer_dims==''):
    transfer_dims = "full"

  test_y_set = np.load("test_samples_without_noise.npy")

  if args.transfer==True:
    if(args.target_task=="denoising"):
      #target_checkpoint_dir = "./checkpoints/A->D_transfer_mnist/"+transfer_dims
      source_checkpoint_dir = "./checkpoints/autoencoding_mnist"
      out_dir = "out/A->D_transfer/"+transfer_dims
      #target_checkpoint_path = target_checkpoint_dir+"/"+"A->D_transfer_"+transfer_dims+'.ckpt'
      test_set = np.load("test_samples_without_noise.npy")
    elif(args.target_task=="autoencoding"):
      #target_checkpoint_dir = "./checkpoints/D->A_transfer/"+transfer_dims
      source_checkpoint_dir = "./checkpoints/denoising_mnist"
      out_dir = "out/D->A_transfer/"+transfer_dims
      #target_checkpoint_path=target_checkpoint_dir+"/"+"D->A_transfer_"+transfer_dims+'.ckpt'
      test_set = np.load("test_samples.npy")    
  else:
    if (args.task=="autoencoding"):
      test_set  = test_y_set
    #target_checkpoint_dir="./checkpoints/"+args.task
    out_dir="out/"+args.task

  #define the selector matrix 
  A = np.zeros(shape=(z_dim_new,z_dim),dtype=np.float32)
  count = 0
  for i in range(z_dim):
    if i in str2vec(args.mi_dims):
      continue
    else:
      A[count][i] = 1
      count += 1 

  dataset = args.dataset
  imsize  = args.imsize
  checkpoint_dir="./checkpoints/A->D_transfer_mnist"
  test_results_file="./out/MI_values.txt"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  vae = ConvVAE(args=args,A=A.T)

  n_samples = test_set.shape[0]
  num_batches = int(n_samples/args.batch_size)
  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
  if ckpt and args.restore:
    vae.load_model(source_checkpoint_dir)


  mi_checkpoint_dir = "./checkpoints/MI/"+args.task+"/"
  if not os.path.exists(mi_checkpoint_dir):
    os.makedirs(mi_checkpoint_dir)
  save = 1
  if(save):
    vae.saver_mi(mi_checkpoint_dir,0)
  vae.load_model(mi_checkpoint_dir, load_mi=True)
  print("Loaded MI")
  #train MI
  for epoch in range(training_epochs):
    mi_loss_avg=[]
    for i in range(num_batches):
      x_batch=[]
      y_batch=[]
      if(args.transfer==True):
        if(args.target_task=="denoising"):
          x_batch=test_set[i*args.batch_size:(i+1)*args.batch_size]
          y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
        elif (args.target_task=="autoencoding"):
          x_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
          y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
      else:
        if(args.task=="denoising"):
          x_batch=test_set[i*args.batch_size:(i+1)*args.batch_size]
          y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
        elif (args.task=="autoencoding"):
          x_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
          y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]

      vae.train_mi(x_batch, y_batch)
      mi_loss=vae.mi_value(x_batch, y_batch)
      mi_loss_avg.append(mi_loss)
    print ("mi loss for epoch: ",epoch,"is ",np.mean(mi_loss_avg))

  #Test MI
  mi_loss_avg = []
  for i in range(num_batches):
    x_batch=[]
    y_batch=[]
    if(args.transfer==True):
      if(args.target_task=="denoising"):
        x_batch=test_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
      elif (args.target_task=="autoencoding"):
        x_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
    else:
      if(args.task=="denoising"):
        x_batch=test_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
      elif (args.task=="autoencoding"):
        x_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]
        y_batch=test_y_set[i*args.batch_size:(i+1)*args.batch_size]

    mi_loss=vae.mi_value(x_batch,y_batch)
    mi_loss_avg.append(mi_loss)
  print ("Final Mi is : ",np.mean(mi_loss_avg) )
  
  with open(test_results_file,"a") as text_file:
  # if args.transfer==True:
   text_file.write("dims-selected :"+str(args.mi_dims)+"  ")
   text_file.write("MI value: "+str(np.mean(mi_loss_avg))+"\n")
# return (np.mean(mi_loss_avg))
    

if __name__ == '__main__':
  main()
