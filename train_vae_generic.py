import numpy as np
import tensorflow as tf
import argparse
import time
import os
import cPickle
import pickle
import itertools
from model.conv_vae_generic import ConvVAE
from utils.tools import *
from utils.loader import *

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
  parser.add_argument('--training_epochs', type=int, default=50,
                     help='training epochs')
  parser.add_argument('--display_step', type=int, default=10,
                     help='display step')
  parser.add_argument('--checkpoint_step', type=int, default=5,
                     help='checkpoint step')
  parser.add_argument('--task', type=str, default="autoencoding",#[denoising, autoencoding,depth_euclidean,edge_texture,normal]
                     help='task name')
  parser.add_argument('--batch_size', type=int, default=64,
                     help='batch size')
  parser.add_argument('--z_dim', type=int, default=20,
                     help='z dim')
  parser.add_argument('--learning_rate', type=float, default=0.0001,
                     help='learning rate')
  parser.add_argument('--dataset', type=str, default='mnist',
                      help='dataset')
  parser.add_argument('--imsize', type=int, default=128,
                      help='imsize')
  # parser.add_argument('--num_channels', type=int, default=1,
  #                     help='num_channels')
  parser.add_argument('--restore',type=int, default=0, help='restore')
  parser.add_argument('--mode',type=str, default='train', help='train_or_test')
  parser.add_argument('--gpu',type=str,default='1')
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  if(args.mode=='train'):
    return train(args)
  elif (args.mode=='test'):
    return test(args)
  elif (args.mode=='edge_detection'):
    return edge_detection_kl(args)
    
  # elif (args.mode=='mi'):
  #   return 

def train(args):

  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  z_dim = args.z_dim # number of latent variables.
  imsize  = args.imsize
  if args.task=="denoising":
    X = np.load("/DATA1/taskonomy-resized/noisy_rgb.npy").astype(np.float32)
    Y = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
  elif args.task=="autoencoding":
    X = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    Y = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
  else:
    Y = np.load("/DATA1/taskonomy-resized/"+args.task+".npy").astype(np.float32)
    X = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
  if len(Y.shape)==3:
    num_channels=1
    Y=np.reshape(Y,[-1,Y.shape[1],Y.shape[2],1])
  else:
    num_channels=Y.shape[3]
  max_y = np.max(Y)
  min_y = np.min(Y)
  max_x = np.max(X)
  min_x = np.min(X)
  Y=(Y-np.min(Y))/(np.max(Y)-np.min(Y))
  X=(X-np.min(X))/(np.max(X)-np.min(X))
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  train_set= X[:7568],Y[:7568]
#   test_set = X[7568:],Y[7568:]
  vae = ConvVAE(args=args,num_channels=num_channels)

  n_samples = train_set[0].shape[0]

  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and args.restore:
    vae.load_model(checkpoint_dir)
    print("Loaded Model from %s", checkpoint_dir)

  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    indices=np.arange(train_set[0].shape[0]).astype(np.int64)
    np.random.shuffle(indices)
    for i in range(total_batch):
      x = train_set[0][i*batch_size:(i+1)*batch_size]
      y = train_set[1][i*batch_size:(i+1)*batch_size]
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
          "disc_loss =", "{:.6f}".format(disc_loss)	," for "+args.task

    out_dir="out/"+args.task
    if not os.path.exists(out_dir):
    	os.makedirs(out_dir)
    i=0
    x = train_set[0][i*batch_size:(i+1)*batch_size]
    y = train_set[1][i*batch_size:(i+1)*batch_size]
    recons = vae.reconstruct(x)
    save_images(y, image_manifold_size(y.shape[0]),out_dir+'/epoch_{}_actual.png'.format(str(epoch+1)))
    save_images(recons, image_manifold_size(recons.shape[0]),out_dir+'/epoch_{}_recons.png'.format(str(epoch+1)))
    # Display logs per epoch step
    if epoch > 0 and epoch % checkpoint_step == 0:
      checkpoint_path = os.path.join(checkpoint_dir, args.task+'_model.ckpt')
      vae.save_model(checkpoint_path, epoch)
      print "model saved to {}".format(checkpoint_path)
    print "Epoch:", '%04d' % (epoch+1), \
          "cost=", "{:.6f}".format(avg_cost)
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
  checkpoint_dir="./checkpoints/"+args.task
  out_dir = "./out/"+args.task+"/test"
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  if args.task=="denoising":
    X = np.load("/DATA1/taskonomy-resized/noisy_rgb.npy").astype(np.float32)
    Y = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
  elif args.task=="autoencoding":
    X = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    Y = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
  else:
    Y = np.load("/DATA1/taskonomy-resized/"+args.task+".npy").astype(np.float32)
    X = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
#   train_set= X[:7568],Y[:7568]
  
  if len(Y.shape)==3:
    num_channels=1
    Y=np.reshape(Y,[-1,Y.shape[1],Y.shape[2],1])
  else:
    num_channels=Y.shape[3]
  max_y = np.max(Y)
  min_y = np.min(Y)
  min_x = np.min(X)
  max_x = np.max(X)
  Y=(Y-np.min(Y))/(np.max(Y)-np.min(Y))
  X=(X-np.min(X))/(np.max(X)-np.min(X))
  test_set = X[7568:],Y[7568:]
  test_set = test_set[0][:1856],test_set[1][:1856]
  vae = ConvVAE(args=args,num_channels=num_channels)

  n_samples = test_set[0].shape[0]

  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt:
    vae.load_model(checkpoint_dir)
    print "Load the trained Model"

  # Loop over all batches
  x = test_set[0]
  y = test_set[1]
  vae_loss_list=[]
  recon_loss_list=[]
  # Fit training using batch data
  print "n smaples = ",n_samples
  # for batch in range(n_samples):
  #   try:
  #     x_batch=x[batch*args.batch_size:(batch+1)*args.batch_size]
  #     y_batch=y[batch*args.batch_size:(batch+1)*args.batch_size]
  #     # print (x_batch.shape)
  #     vae_cost, reconstr_loss = vae.test_loss(x_batch,y_batch)
  #     # print (reconstr_loss)
  #     vae_loss_list.append(vae_cost)
  #     recon_loss_list.append(reconstr_loss)
  #     print batch+1
  #   except:
  #     pass
  # print (x_batch.shape)
  x_batch=x[:args.batch_size]
  y_batch=y[:args.batch_size]
  x_recons = vae.reconstruct(x_batch)
  print ("reconstruction done")
  # avg_vae_cost = np.mean(vae_cost)
  # avg_recons_loss = np.mean(reconstr_loss)

  # print("Average vae cost: %f, Average recons cost: %f"%(avg_vae_cost,avg_recons_loss))

  # test_results_file = "./out/task_test_results.txt"
  # with open(test_results_file,"a") as text_file:
  #   text_file.write("task :"+args.task+"\n")

  #   text_file.write("recon loss : "+str(np.mean(reconstr_loss))+"\n")
  #   text_file.write("vae cost : "+str(np.mean(vae_cost))+"\n")
  #   text_file.write("---------------------------------"+"\n")
  #   text_file.close() 

  save_images(y_batch, image_manifold_size(y_batch.shape[0]),out_dir+'/test_actual.png')
  save_images((x_recons), image_manifold_size(x_recons.shape[0]),out_dir+'/test_recons.png')

def kl_estimator(args,z):
  with tf.variable_scope("kl_discriminator", reuse=False,is_train=True):
    disc_1 = tf.layers.dense(inputs=z, units=1024, activation=tf.nn.leaky_relu, name="disc_kl_1",trainable=is_train)
    disc_2 = tf.layers.dense(inputs=disc_1, units=1024, activation=tf.nn.leaky_relu, name="disc_kl_2",trainable=is_train)
    disc_3 = tf.layers.dense(inputs=disc_2, units=1024, activation=tf.nn.leaky_relu, name="disc_kl_3",trainable=is_train)
    disc_4 = tf.layers.dense(inputs=disc_3, units=1024, activation=tf.nn.leaky_relu, name="disc_kl_4",trainable=is_train)
    disc_5 = tf.layers.dense(inputs=disc_4, units=1024, activation=tf.nn.leaky_relu, name="disc_kl_5",trainable=is_train)
    disc_6 = tf.layers.dense(inputs=disc_5, units=1024, activation=tf.nn.leaky_relu, name="disc_kl_6",trainable=is_train)
    logits = tf.layers.dense(inputs=disc_6, units=2, name="disc_kl_logits",trainable=is_train)
    probabilities = tf.nn.softmax(logits)
  return logits, probabilities

def edge_detection(args):
  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  z_dim = args.z_dim # number of latent variables.
  imsize  = args.imsize
  possible_combs=list(itertools.combinations(["denoising", "autoencoding","depth_euclidean","edge_texture","normal"],2))
  z_1=tf.placeholder(tf.float32, [None, args.z_dim])
  z_2=tf.placeholder(tf.float32, [None, args.z_dim])
  vae = ConvVAE(args=args,num_channels=num_channels)
  z1_latents=[]
  z2_latents=[]
  for comb in possible_combs:
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    if comb[0]=="denoising":
      X = np.load("/DATA1/taskonomy-resized/noisy_rgb.npy").astype(np.float32)
      Y = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    elif comb[0]=="autoencoding":
      X = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
      Y = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    else:
      Y = np.load("/DATA1/taskonomy-resized/"+comb[0]+".npy").astype(np.float32)
      X = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
    if len(Y.shape)==3:
      num_channels=1
      Y=np.reshape(Y,[-1,Y.shape[1],Y.shape[2],1])
    else:
      num_channels=Y.shape[3]
    max_y = np.max(Y)
    min_y = np.min(Y)
    min_x = np.min(X)
    max_x = np.max(X)
    Y=(Y-np.min(Y))/(np.max(Y)-np.min(Y))
    X=(X-np.min(X))/(np.max(X)-np.min(X))
    test_set = X[7568:],Y[7568:]
    test_set = test_set[0][:1856],test_set[1][:1856]
    checkpoint_dir="./checkpoints/"+comb[0]
    vae.load_model(checkpoint_dir)
    for i in range(len(X)):
      z1_latents.append(vae.transform(X[i]))
    checkpoint_dir="./checkpoints/"+comb[1]
    vae.load_model(checkpoint_dir)
    for i in range(len(X)):
      z2_latents.append(vae.transform(X[i]))
    reuse_var=False  
    # out_dir = "./out/"+args.task+"/test"
    logits_real, probs_real = kl_discriminator(z_1,reuse=reuse_var)
    logits_permuted, probs_permuted = kl_discriminator(z_2,reuse=True)
    loss = tf.reduce_mean(tf.log(tf.softmax(logits_real)[0])-tf.log(tf.softmax(logits_real)[1]))
    t_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='kl_discriminator')
    optimizer=tf.train.AdamOptimizer(0.001).minimize(loss, var_list=t_vars)
    init=tf.global_variables_initializer()
    avg_kld=[]
    with tf.Session() as sess:
      for j in range(2):
        kld_vals=[]
        sess.run(init)
        for epoch in range(15):
          for batch in test_set.shape[0]//args.batch_size:
            batch_z1=z1_latents[batch*args.batch_size:(batch+1)*args.batch_size]
            batch_z2=z2_latents[batch*args.batch_size:(batch+1)*args.batch_size]
            _,kld=sess.run([optimizer,loss],{z_1:batch_z1,z_2:batch_z2})
        for i in range(len(z1_latents)):
          kld_vals.append(sess.run(loss,{z_1:z1_latents[i].reshape((1,args.z_dim)),z_2:z2_latents[i].reshape((1,args.z_dim))}))
        avg_kld.append(np.mean(kld_vals))
        b=z1_latents
        z1_latents=z2_latents
        z2_latents=b
    with open("kl_results.txt",'a') as kl_file:
      kl_file.write("Task 1: %s, Task 2: %s"%(comb[1],comb[0]))
      kl_file.write("1->2 kl loss: %f"%avg_kld[0])
      kl_file.write("2->1 kl loss: %f"%avg_kld[1])
      kl_file.write("################################################")
      kl_file.close()
    

          
        
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
