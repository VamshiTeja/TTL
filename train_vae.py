import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle
import pickle
#import cv2
from tensorflow.examples.tutorials.mnist import input_data

from model.conv_vae import ConvVAE

#from utils.loader import load_batched_data
from utils.tools import *
from utils.loader import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = np.reshape(mnist.train.images, [-1, 28, 28, 1])[:-2000]
test_set  = np.reshape(mnist.train.images, [-1, 28, 28, 1])[-2000:]
#test_set = np.load("./test_samples.npy")
#train_data = (train_set+1.0) / 2.0  # normalization; range: -1 ~ 1
#dataset_zip = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding='bytes')
#imgs = dataset_zip['imgs']

#train_set = np.reshape(imgs, [-1,64,64,1])

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
  parser.add_argument('--display_step', type=int, default=10,
                     help='display step')
  parser.add_argument('--checkpoint_step', type=int, default=5,
                     help='checkpoint step')
  parser.add_argument('--task', type=str, default="autoencoding",
                     help='task name(denoising or autoencoding)')
  parser.add_argument('--batch_size', type=int, default=256,
                     help='batch size')
  parser.add_argument('--z_dim', type=int, default=20,
                     help='z dim')
  parser.add_argument('--learning_rate', type=float, default=0.0001,
                     help='learning rate')
  parser.add_argument('--dataset', type=str, default='mnist',
                      help='dataset')
  parser.add_argument('--imsize', type=int, default=28,
                      help='imsize')
  parser.add_argument('--num_channels', type=int, default=1,
                      help='num_channels')
  parser.add_argument('--restore',type=int, default=1, help='restore')
  parser.add_argument('--train_or_test',type=str, default='train', help='train_or_test')
  args = parser.parse_args()
  if(args.train_or_test=='train'):
    return train(args)
  else:
    return test(args)

def train(args):

  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  z_dim = args.z_dim # number of latent variables.

  dataset = args.dataset
  imsize  = args.imsize
  if(args.task=="denoising"):
    checkpoint_dir = "./checkpoints/denoising_mnist"
  elif(args.task=="autoencoding"):
    checkpoint_dir = "./checkpoints/autoencoding_mnist"

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  vae = ConvVAE(args=args)

  n_samples = train_set.shape[0]

  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and args.restore:
    vae.load_model(checkpoint_dir)

  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = sample_minibatch(batch_size=args.batch_size)

        if(args.task=="denoising"):
          noisy_batch = []
          for j in range(batch_size):
            noisy_batch.append(make_noisy(batch_xs[j]))
          x = noisy_batch
          y = batch_xs
        elif(args.task=="autoencoding"):
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

    if(args.task=="denoising"):
      out_dir = "out/denoising"
    elif(args.task=="autoencoding_mnist"):
      out_dir = "out/autoencoding_mnist"

    if not os.path.exists(out_dir):
    	os.makedirs(out_dir)

    if(args.task=="denoising"):
      noisy_batch = []
      for j in range(batch_size):
        noisy_batch.append(add_noise(batch_xs[j]))
      x = np.array(noisy_batch)
      y = batch_xs
    elif(args.task=="autoencoding"):
      x = batch_xs
      y = batch_xs

    recons = vae.reconstruct(x)
    save_images(x, image_manifold_size(x.shape[0]),out_dir+'/epoch_{}_actual.png'.format(str(epoch+1)))
    save_images(recons, image_manifold_size(recons.shape[0]),out_dir+'/epoch_{}_recons.png'.format(str(epoch+1)))
    # Display logs per epoch step
    print "Epoch:", '%04d' % (epoch+1), \
          "cost=", "{:.6f}".format(avg_cost)

    # save model
    if epoch > 0 and epoch % checkpoint_step == 0:
      checkpoint_path = os.path.join(checkpoint_dir, args.task+'_model.ckpt')
      vae.save_model(checkpoint_path, epoch)
      print "model saved to {}".format(checkpoint_path)

  # save model one last time, under zero label to denote finish.
  vae.save_model(checkpoint_path, 0)

  return vae

def test(args):

  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  z_dim = args.z_dim # number of latent variables.

  dataset = args.dataset
  imsize  = args.imsize
  if(args.task=="denoising"):
    checkpoint_dir = "./checkpoints/denoising_mnist"
    out_dir = "./out/denoising_mnist/test"
  elif(args.task=="autoencoding"):
    checkpoint_dir = "./checkpoints/autoencoding_mnist"
    out_dir = "./out/autoencoding_mnist/test"

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  vae = ConvVAE(args=args)

  n_samples = test_set.shape[0]

  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt:
    vae.load_model(checkpoint_dir)

  # Loop over all batches
  x = test_set
  
  if(args.task=="denoising"):
    noisy_x = []
    for j in range(n_samples):
      noisy_x.append(make_noisy(x[j]))
   
    y = x
    x = np.array(noisy_x)
  elif(args.task=="autoencoding"):
    x = x
    y = x

  # Fit training using batch data
  vae_cost, reconstr_loss = vae.test_loss(x,y)
  x_recons = vae.reconstruct(x)

  avg_vae_cost = np.mean(vae_cost)
  avg_recons_loss = np.mean(reconstr_loss)

  print("Average vae cost: %f, Average recons cost: %f"%(avg_vae_cost,avg_recons_loss))

  test_results_file = "./out/task_test_results.txt"
  with open(test_results_file,"a") as text_file:
    text_file.write("task :"+args.task+"\n")

    text_file.write("recon loss : "+str(np.mean(reconstr_loss))+"\n")
    text_file.write("vae cost : "+str(np.mean(vae_cost))+"\n")
    text_file.write("---------------------------------"+"\n")
    text_file.close() 

  save_images(x[0:256,:,:], image_manifold_size(x[0:256,:,:].shape[0]),out_dir+'/test_actual.png')
  save_images(x_recons[0:256,:,:], image_manifold_size(x_recons[0:256,:,:].shape[0]),out_dir+'/test_recons.png')




if __name__ == '__main__':
  main()
