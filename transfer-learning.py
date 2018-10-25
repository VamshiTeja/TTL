# transfer learning script

import itertools
import numpy as np
import tensorflow as tf
import argparse
import time
import os
import cPickle
import pickle
from tqdm import tqdm
from model.transfer_learning_vae import FactorVAE
from utils.tools import *
from utils.loader import *
from config import cfg, cfg_from_file

mnist = open("./datasets/mnist.pickle", "rb")
mnist_dict = pickle.load(mnist)
cifar10 = open("./datasets/cifar.pickle", "rb")
cifar_dict = pickle.load(cifar10)
svhn = open("./datasets/svhn.pickle", "rb")
svhn_dict = pickle.load(svhn)


def sample_minibatch(dataset, batch_size=256):
    indices = np.random.choice(range(len(dataset)), batch_size, replace=False)
    return indices


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_epochs', type=int, default=30, help='training epochs')
    # parser.add_argument('--display_step', type=int, default=20, help='display step')
    # parser.add_argument('--checkpoint_step', type=int, default=5, help='checkpoint step')
    # parser.add_argument('--task', type=str, default="autoencoding", help='autoencoding|classification')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--z_dim', type=int, default=10, help='latent variable dimensionality')
    # parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    # parser.add_argument('--dataset', type=str, default='svhn', help='mnist|svhn|cifar10')
    # parser.add_argument('--restore', type=int, default=0, help='restore')
    # parser.add_argument('--transfer', type=bool, default=True, help="")
    # parser.add_argument('--source_task', type=str, default="autoencoding")
    # parser.add_argument('--target_task', type=str, default="classification")
    # parser.add_argument('--mode', type=str, default="train", help="train|test|mi")  # test or train or edge-detection
    # parser.add_argument('--remove_dims', type=str, default='')
    # parser.add_argument('--gpu', type=str, default='s0')

    parser.add_argument("--cfg", dest='cfg_file', default='./config/transfer-learning.yml', type=str, help="An optional config file"
                                                                                              " to be loaded")
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    if (cfg.mode == "train"):
        return train(cfg)
    elif (cfg.mode == "test"):
        return test(cfg)
    elif (cfg.mode == "mi"):
        return MI(cfg)


def train(cfg):
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    display_step = cfg.display_step
    checkpoint_step = cfg.checkpoint_step  # save training results every check point step
    z_dim = cfg.z_dim  # number of latent variables.

    # set the selector matrix
    A = np.zeros(shape=(z_dim, z_dim)).astype(np.float32)
    count = 0
    if (cfg.remove_dims != ''):
        for i in range(z_dim):
            if i in str2vec(cfg.remove_dims):
                continue
            else:
                A[count][i] = 1
                count += 1

    transfer_dims = cfg.remove_dims
    if (transfer_dims == ''):
        A = np.identity(cfg.z_dim, dtype=np.float32)
        transfer_dims = "full"
        print("Full latent from encoder to decoder")

    if cfg.transfer == False:
        print ("no tranfer")
        print ("currently training the task : " + cfg.task)
        if cfg.task == "autoencoding":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_train"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_train"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_train"]

            dataset_x = dataset_x.astype(np.float32) / 255.0
            dataset_y = dataset_x
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = num_channels_x
        elif cfg.task == "classification":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_train"]
                dataset_y = mnist_dict["y_train"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_train"]
                dataset_y = cifar_dict["y_train"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_train"]
                dataset_y = svhn_dict["y_train"]
            dataset_x = dataset_x.astype(np.float32) / 255.0
            a = np.zeros((dataset_x.shape[0], 10))
            a[np.arange(dataset_x.shape[0]), dataset_y] = 1
            dataset_y = a
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = None
        target_checkpoint_dir = "./checkpoints/source_tasks/" + cfg.dataset + "/" + str(cfg.z_dim) + "/" + cfg.task
        out_dir = "./out/source_tasks/" + cfg.dataset + "/" + str(cfg.z_dim) + "/" + cfg.task
        if not os.path.exists(target_checkpoint_dir):
            os.makedirs(target_checkpoint_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        target_checkpoint_path = target_checkpoint_dir + "/" + cfg.task + "_" + str(cfg.z_dim) + '_model.ckpt'
        if not (len(os.listdir(target_checkpoint_dir))):
            ckpt = None
        else:
            ckpt = tf.train.latest_checkpoint(target_checkpoint_dir)
        # print (target_checkpoint_path)
    else:
        print ("source task is " + cfg.source_task)
        print ("target task is " + cfg.target_task)
        print("Removed dimensions: %s" % cfg.remove_dims)
        if cfg.source_task == "autoencoding":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_val"]
                dataset_y = mnist_dict["y_val"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_val"]
                dataset_y = cifar_dict["y_val"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_val"]
                dataset_y = svhn_dict["y_val"]
            a = np.zeros((dataset_x.shape[0], 10))
            dataset_x = dataset_x.astype(np.float32) / 255.0
            a[np.arange(dataset_x.shape[0]), dataset_y] = 1
            dataset_y = a
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = None
        elif cfg.source_task == "classification":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_val"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_val"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_val"]
            dataset_x = dataset_x.astype(np.float32) / 255.0
            dataset_y = dataset_x
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = num_channels_x
        target_checkpoint_dir = "./checkpoints/transfers/" + cfg.dataset + "/" + str(
            cfg.z_dim) + "/" + cfg.source_task + "->" + cfg.target_task + "/" + transfer_dims
        source_checkpoint_dir = "./checkpoints/source_tasks/" + cfg.dataset + "/" + str(
            cfg.z_dim) + "/" + cfg.source_task
        out_dir = "./out/transfers/" + cfg.dataset + "/" + str(
            cfg.z_dim) + "/" + cfg.source_task + "->" + cfg.target_task + str(cfg.z_dim) + "_" + transfer_dims
        target_checkpoint_path = target_checkpoint_dir + "/" + cfg.source_task + "->" + cfg.target_task + str(
            cfg.z_dim) + "_" + transfer_dims + '.ckpt'
        ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
        target_decoder_checkpoint_dir =   "./checkpoints/transfers/" + cfg.dataset + "/" + str(
            cfg.z_dim) + "/" + cfg.target_task + "_decoder"
        target_decoder_checkpoint_path = target_decoder_checkpoint_dir + "/" + cfg.target_task + "_decoder_" + str(cfg.z_dim) + '.ckpt'
        target_decoder_ckpt = tf.train.get_checkpoint_state(target_decoder_checkpoint_dir)

    imsize = dataset_x[0].shape[1]
    if not os.path.exists(target_decoder_checkpoint_dir):
        os.makedirs(target_decoder_checkpoint_dir)
    if not os.path.exists(target_checkpoint_dir):
        os.makedirs(target_checkpoint_dir)

    #Freeze encoder while transfer
    if(cfg.transfer):
        isTrain_Enc = False
    else:
        isTrain_Enc = True

    vae = FactorVAE(args=cfg, A=A.T, num_channels_x=num_channels_x, num_channels_y=num_channels_y, imsize=imsize,
                    isTrain_Enc=isTrain_Enc)

    train_set = (dataset_x, dataset_y)
    n_samples = train_set[0].shape[0]
    print ("X shape ", dataset_x.shape)
    print ("y shape ", dataset_y.shape)
    start_epoch = 0
    if (cfg.transfer == True):
        restore = True
        if ckpt:
            vae.load_model(source_checkpoint_dir, load_transfer=True)
        #loads target task static decoder for uniform experimentation
        if target_decoder_ckpt:
            vae.load_decoder(target_decoder_checkpoint_dir)
            print("Loaded Target Task static decoder(same init) from %s"%target_decoder_checkpoint_path)
        else:
            vae.save_decoder(target_decoder_checkpoint_path)
            print("Saved Target Task static decoder to %s"%target_decoder_checkpoint_path)

    else:
        restore = cfg.transfer
        if ckpt:
            vae.load_model(target_checkpoint_dir, load_transfer=False)
            i = -1
            while (i > len(ckpt) * -1):
                if (ckpt[i] == "-"):
                    break
                i -= 1
            start_epoch = int(ckpt[i + 1:])

    # if ckpt and restore:
    #     vae.load_model(source_checkpoint_dir)
    if ckpt == False and restore == True:
        print ("no saved model to load")

    for epoch in range(num_epochs):
        # break
        if (start_epoch):
            if epoch <= start_epoch:
                continue
        # print ("epoch number is---------------------------------------- "+str(epoch))
        # break
        avg_cost = 0.
        reconstr_loss = 0
        kl_loss = 0
        disc_loss = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            indices = sample_minibatch(train_set[0], cfg.batch_size)
            x = train_set[0][indices]
            y = train_set[1][indices]
            vae_tr_loss, recons_cost, kl, disc = vae.partial_fit(x, y)
            avg_cost += vae_tr_loss
            reconstr_loss += recons_cost
            kl_loss += kl
            disc_loss += disc
        avg_cost = (avg_cost / n_samples) * batch_size
        reconstr_loss = (reconstr_loss / n_samples) * batch_size
        kl_loss = (kl_loss / n_samples) * batch_size
        disc_loss = (disc_loss / n_samples) * batch_size
        if num_channels_y:
            print "Epoch:", '%04d' % (epoch + 1), \
                "batch:", '%04d' % (i), \
                "vae_cost:", "{:.6f}".format(vae_tr_loss), \
                "reconstr_loss =", "{:.6f}".format(reconstr_loss), \
                "kl_loss =", "{:.6f}".format(kl), \
                "disc_loss =", "{:.6f}".format(disc)
        else:
            print "Epoch:", '%04d' % (epoch + 1), \
                "batch:", '%04d' % (i), \
                "vae_cost:", "{:.6f}".format(vae_tr_loss), \
                "avg_accuracy =", "{:.6f}".format(reconstr_loss), \
                "kl_loss =", "{:.6f}".format(kl), \
                "disc_loss =", "{:.6f}".format(disc)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        recons = vae.reconstruct(x)
        if num_channels_y:
            if (cfg.batch_size >= 256):
                save_images(y[0:256], image_manifold_size(256), out_dir + '/epoch_{}_actual.png'.format(str(epoch + 1)))
                save_images(recons[0:256], image_manifold_size(256),
                            out_dir + '/epoch_{}_recons.png'.format(str(epoch + 1)))
            else:
                save_images(y, image_manifold_size(y.shape[0]), out_dir + '/epoch_{}_actual.png'.format(str(epoch + 1)))
                save_images(recons, image_manifold_size(recons.shape[0]),
                            out_dir + '/epoch_{}_recons.png'.format(str(epoch + 1)))
        else:
            with open(out_dir + "/classification.txt", "a") as text_file:
                # text_file.write("Removed dims :"+args.remove_dims+"\n")
                if epoch == 0:
                    if cfg.transfer:
                        text_file.write("source task :" + cfg.task + "\n")
                        text_file.write("target task :" + cfg.task + "\n")
                    else:
                        text_file.write("source task :" + cfg.source_task + "\n")
                        text_file.write("target task :" + cfg.target_task + "\n")
                text_file.write("accuracy of epoch " + str(epoch) + " = " + str(reconstr_loss) + "\n")
                text_file.close()
                if cfg.transfer == False:
                    print ("current task : " + cfg.task)
                else:
                    print ("source task :" + cfg.source_task + " target task: " + cfg.target_task)
        print (target_checkpoint_path)
    vae.save_model(target_checkpoint_path, num_epochs)
    print ("model saved")
    # break
    # return vae


def test(cfg):

    #load eps
    eps = np.load("./datasets/eps_10.npy")

    z_dim = cfg.z_dim  # number of latent variables.

    # set the selector matrix
    A = np.zeros(shape=(z_dim, z_dim)).astype(np.float32)
    count = 0
    if (cfg.remove_dims != ''):
        for i in range(z_dim):
            if i in str2vec(cfg.remove_dims):
                continue
            else:
                A[count][i] = 1
                count += 1

    transfer_dims = cfg.remove_dims
    if (transfer_dims == ''):
        A = np.identity(cfg.z_dim, dtype=np.float32)
        transfer_dims = "full"
        print("Full Latent from encoder to decoder")

    if cfg.transfer == False:
        print ("No tranfer")
        print ("Currently Training the task : " + cfg.task)
        if cfg.task == "Autoencoding":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_test"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_test"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_test"]

            dataset_x = dataset_x.astype(np.float32) / 255.0
            dataset_y = dataset_x
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = num_channels_x
        elif cfg.task == "classification":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_test"]
                dataset_y = mnist_dict["y_test"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_test"]
                dataset_y = cifar_dict["y_test"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_test"]
                dataset_y = svhn_dict["y_test"]
            dataset_x = dataset_x.astype(np.float32) / 255.0
            a = np.zeros((dataset_x.shape[0], 10))
            a[np.arange(dataset_x.shape[0]), dataset_y] = 1
            dataset_y = a
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = None
        target_checkpoint_dir = "./checkpoints/source_tasks/" + cfg.dataset + "/" + str(cfg.z_dim) + "/" + cfg.task
        out_dir = "./out/source_tasks/" + cfg.dataset + "/" + str(cfg.z_dim) + "/" + cfg.task
        log_dir = "./logging/source_tasks/" + cfg.dataset +"/" + str(cfg.z_dim) + "/"+ cfg.task+"/"
        test_results_file = log_dir + cfg.task +"_test.txt"

        if not os.path.exists(target_checkpoint_dir):
            os.makedirs(target_checkpoint_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        target_checkpoint_path = target_checkpoint_dir + "/" + cfg.task + "_" + str(cfg.z_dim) + '_model.ckpt'
        if not (len(os.listdir(target_checkpoint_dir))):
            ckpt = None
        else:
            ckpt = tf.train.latest_checkpoint(target_checkpoint_dir)
        # print (target_checkpoint_path)
    else:
        print ("source task is " + cfg.source_task)
        print ("target task is " + cfg.target_task)
        if(cfg.remove_dims!=""):
            print("Removed dimensions: %s" % cfg.remove_dims)
        if cfg.source_task == "autoencoding":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_test"]
                dataset_y = mnist_dict["y_test"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_test"]
                dataset_y = cifar_dict["y_test"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_test"]
                dataset_y = svhn_dict["y_test"]
            a = np.zeros((dataset_x.shape[0], 10))
            dataset_x = dataset_x.astype(np.float32) / 255.0
            a[np.arange(dataset_x.shape[0]), dataset_y] = 1
            dataset_y = a
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = None
        elif cfg.source_task == "classification":
            if cfg.dataset == "mnist":
                dataset_x = mnist_dict["X_test"]
            elif cfg.dataset == "cifar10":
                dataset_x = cifar_dict["X_test"]
            elif cfg.dataset == "svhn":
                dataset_x = svhn_dict["X_test"]
            dataset_x = dataset_x.astype(np.float32) / 255.0
            dataset_y = dataset_x
            num_channels_x = dataset_x[0].shape[-1]
            num_channels_y = num_channels_x

        target_checkpoint_dir = "./checkpoints/transfers/" + cfg.dataset + "/" + str(
            cfg.z_dim) + "/" + cfg.source_task + "->" + cfg.target_task + "/" + transfer_dims
        source_checkpoint_dir = "./checkpoints/source_tasks/" + cfg.dataset + "/" + str(
            cfg.z_dim) + "/" + cfg.source_task
        out_dir = "./out/transfers/" + cfg.dataset + "/" + str(
            cfg.z_dim) + "/" + cfg.source_task + "->" + cfg.target_task + str(cfg.z_dim) + "_" + transfer_dims
        target_checkpoint_path = target_checkpoint_dir + "/" + cfg.source_task + "->" + cfg.target_task + str(
            cfg.z_dim) + "_" + transfer_dims + '.ckpt'
        ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
        log_dir = "./logging/transfers/"+ cfg.dataset + "/" + str(cfg.z_dim) + "/" + cfg.source_task + "->" + cfg.target_task + "/" + transfer_dims +"/"
        test_results_file = log_dir +cfg.source_task + "->" + cfg.target_task+"_test.txt"


    imsize = dataset_x[0].shape[1]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(target_checkpoint_dir):
        os.makedirs(target_checkpoint_dir)
    if not os.path.exists("./logging"):
        os.makedirs("./logging")

    #Freeze encoder while transfer
    if(cfg.transfer):
        isTrain_Enc = False
    else:
        isTrain_Enc = True

    vae = FactorVAE(args=cfg, A=A.T, num_channels_x=num_channels_x, num_channels_y=num_channels_y, imsize=imsize, isTrain_Enc=isTrain_Enc)

    ckpt = tf.train.get_checkpoint_state(target_checkpoint_dir)

    if ckpt:
        vae.load_model(target_checkpoint_dir)
        print ("Loaded Trained Model")

    recons_cost_per_run = []
    vae_cost_per_run = []
    avg_recon_cost = []
    avg_vae_cost = []

    n_batches = int(dataset_y.shape[0] / cfg.batch_size_test)
    print("Number of Test batches are %d" % n_batches)
    for run in tqdm(range(cfg.num_test_runs)):
        for i in range(n_batches):
            x = dataset_x[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            y = dataset_y[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            # n_samples = test_set.shape[0]
            vae_cost, reconstr_loss, kl_div = vae.test_loss(x, y,eps[i])  # vae.partial_fit(x_batch,y_batch)
            avg_recon_cost.append(reconstr_loss)
            avg_vae_cost.append(vae_cost)
        recons_cost_per_run.append(np.mean(avg_recon_cost))
        vae_cost_per_run.append(np.mean(avg_vae_cost))

    #for saving test images
    i = 0
    x = dataset_x[i * cfg.batch_size:(i + 1) * cfg.batch_size]
    y = dataset_y[i * cfg.batch_size:(i + 1) * cfg.batch_size]
    recons = vae.reconstruct(x)
    if num_channels_y:
        if (cfg.batch_size >= 256):
            save_images(y[0:256], image_manifold_size(256), log_dir + '/epoch_{}_actual.png'.format(str(epoch + 1)))
            save_images(recons[0:256], image_manifold_size(256),
                        log_dir +"/recons/"+ '/test.png')
        else:
            save_images(y, image_manifold_size(y.shape[0]), out_dir + '/epoch_{}_actual.png'.format(str(epoch + 1)))
            save_images(recons, image_manifold_size(recons.shape[0]),
                        log_dir + "/recons/" + '/test.png')

        with open(test_results_file,"a") as text_file:
            if cfg.transfer:
                text_file.write("source task :" + cfg.source_task + "\n")
                text_file.write("target task :" + cfg.target_task + "\n")
            else:
                text_file.write("Task :" + cfg.task + "\n")

            text_file.write("Average Reconstruction Error : " + str(np.mean(recons_cost_per_run)) + "\n")
            text_file.close()
            if cfg.transfer == False:
                print ("current task : " + cfg.task)
            else:
                if(cfg.remove_dims==''):
                    print("Transfer Full Latent")
                else:
                    print ("Source task :" + cfg.source_task + " Target task: " + cfg.target_task+" Remove dimensions: "+cfg.remove_dims+"\n")
            print("Average Reconstruction Error : " + str(np.mean(recons_cost_per_run)) + "\n")
    else:
        with open(test_results_file, "a") as text_file:
            # text_file.write("Removed dims :"+args.remove_dims+"\n")

            if cfg.transfer:
                text_file.write("source task :" + cfg.source_task + "\n")
                text_file.write("target task :" + cfg.target_task + "\n")
                if(cfg.remove_dims!=""):
                    text_file.write("Remove dimensions: "+cfg.remove_dims)
            else:
                text_file.write("Task :" + cfg.task + "\n")

            text_file.write("Test Accuracy : " + str(np.mean(recons_cost_per_run)) + "\n")
            text_file.close()
            if cfg.transfer == False:
                print ("current task : " + cfg.task)
            else:
                if (cfg.remove_dims == ''):
                    print("Transfer Full Latent")
                else:
                    print ("Source   task :" + cfg.source_task + " Target task: " + cfg.target_task + " Remove dimensions: " + cfg.remove_dims+"\n")
            print("Test Accuracy : " + str(np.mean(recons_cost_per_run)) + "\n")


def MI(args):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    display_step = args.display_step
    checkpoint_step = args.checkpoint_step  # save training results every check point step
    z_dim = args.z_dim  # number of latent variables.

    # set the selector matrix
    A = np.zeros(shape=(z_dim, z_dim)).astype(np.float32)
    count = 0
    if (args.remove_dims != ''):
        for i in range(z_dim):
            if i in str2vec(args.remove_dims):
                continue
            else:
                A[count][i] = 1
                count += 1

    if (args.remove_dims == ''):
        A = np.identity(args.z_dim, dtype=np.float32)
        print("Full Transfer")

    transfer_dims = args.remove_dims
    if (transfer_dims == ''):
        transfer_dims = "full"

    if args.dataset == "mnist":
        dataset_x = np.reshape(mnist.train.images, [-1, 28, 28, 1])
        dataset_y = np.reshape(mnist.train.images, [-1, 28, 28, 1])
    elif args.dataset == "taskonomy":
        dataset_x = np.load("./preprocess/autoencoding.npy")
        if (args.transfer == True):
            if args.target_task == "denoising":
                dataset_x = np.load("./preprocess/denoising.npy")
                dataset_y = np.load("./preprocess/autoencoding.npy")
            else:
                dataset_y = np.load("./preprocess/" + args.target_task + ".npy")
        else:
            if args.task == "denoising":
                dataset_x = np.load("./preprocess/denoising.npy")
                dataset_y = np.load("./preprocess/autoencoding.npy")
            else:
                dataset_y = np.load("./preprocess/" + args.task + ".npy")

    dataset_x = dataset_x[int(0.8 * len(dataset_x)):]
    dataset_y = dataset_y[int(0.8 * len(dataset_y)):]
    dataset_x = dataset_x.astype(np.float32)
    dataset_y = dataset_y.astype(np.float32)
    min_x = np.min(dataset_x)
    max_x = np.max(dataset_x)
    min_y = np.min(dataset_y)
    max_y = np.max(dataset_y)
    dataset_x = (dataset_x - min_x) / (max_x - min_x)
    dataset_y = (dataset_y - min_y) / (max_y - min_y)
    imsize = dataset_y[0].shape[1]
    num_channels_x = dataset_x[0].shape[-1]
    num_channels_y = dataset_y[0].shape[-1]

    # set all directories for saving checkpoints and saving results and logs
    if (args.transfer == True):
        target_checkpoint_dir = "./checkpoints/transfers/" + str(
            args.z_dim) + "/" + args.source_task + "->" + args.target_task + "/" + transfer_dims
        source_checkpoint_dir = "./checkpoints/source_tasks/" + str(args.z_dim) + "/" + args.source_task
        out_dir = "./out/transfers/" + str(args.z_dim) + "/" + args.source_task + "->" + args.target_task + str(
            args.z_dim) + "_" + transfer_dims
        target_checkpoint_path = target_checkpoint_dir + "/" + args.source_task + "->" + args.target_task + str(
            args.z_dim) + "_" + transfer_dims + '.ckpt'
        ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
    else:
        target_checkpoint_dir = "./checkpoints/source_tasks/" + str(args.z_dim) + "/" + args.task
        out_dir = "./out/source_tasks/" + str(args.z_dim) + "/" + args.task
        ckpt = None
        target_checkpoint_path = target_checkpoint_dir + "/" + args.task + "_" + str(args.z_dim) + '_model.ckpt'

    mi_checkpoint_dir = "./checkpoints/mi/" + str(args.z_dim) + "/"
    mi_checkpoint_path = mi_checkpoint_dir + "mi_params.ckpt"

    if (args.transfer):
        mi_results_file = "logging/mi/" + str(
            args.z_dim) + "/" + args.source_task + "->" + args.target_task + "_transfer.txt"
    else:
        mi_results_file = "logging/source_tasks/" + str(args.z_dim) + "/" + args.task

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(mi_checkpoint_dir):
        os.makedirs(mi_checkpoint_dir)

    vae = FactorVAE(args=args, A=A.T, imsize=imsize, num_channels_x=num_channels_x, num_channels_y=num_channels_y)

    ckpt = tf.train.get_checkpoint_state(source_checkpoint_dir)
    if ckpt:
        vae.load_model(source_checkpoint_dir, load_transfer=True)
        print ("Loaded model")

    ckpt = tf.train.get_checkpoint_state(mi_checkpoint_dir)
    if ckpt:
        vae.load_model(mi_checkpoint_dir, load_mi=True)
        print ("Loaded model")

    mi_loss_avg = []
    for epoch in range(30):
        # break
        np.random.shuffle(indices)
        test_set = dataset_x[indices]
        test_y_set = dataset_y[indices]

        for i in range(test_set.shape[0] // batch_size):
            # batch_x_corrupt = test_set[i*batch_size:(i+1)*batch_size]
            batch_x = test_set[i * batch_size:(i + 1) * batch_size]
            batch_y = test_y_set[i * batch_size:(i + 1) * batch_size]
            vae.train_mi(batch_x, batch_y)
            mi_loss = vae.mi_value(batch_x, batch_y)
            mi_loss_avg.append(mi_loss)
            print ("mi loss for epoch: ", epoch, "is ", np.mean(mi_loss_avg))

    mi_loss_avg = []
    for i in range(test_set.shape[0] // args.batch_size):
        # batch_x_corrupt = test_set[i*args.batch_size:(i+1)*args.batch_size]
        batch_x = test_set[i * batch_size:(i + 1) * batch_size]
        batch_y = test_y_set[i * batch_size:(i + 1) * batch_size]
        vae.train_mi(batch_x, batch_y)
        mi_loss = vae.mi_value(batch_x, batch_y)
        mi_loss_avg.append(mi_loss)
        print ("final mi is : ", np.mean(mi_loss_avg))

    with open(mi_results_file, "a") as text_file:
        # if args.transfer==True:
        text_file.write("dims-selected :" + str(args.remove_dims) + "\n")
        text_file.write("MI value: " + str(np.mean(mi_loss_avg)) + "\n")
    # return (np.mean(mi_loss_avg))


def edge_detection_kl(args):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    display_step = args.display_step
    z_dim = args.z_dim  # number of latent variables.
    imsize = args.imsize
    possible_combs = list(itertools.combinations(["autoencoding", "normal"], 2))
    z_1 = tf.placeholder(tf.float32, [None, args.z_dim])
    z_2 = tf.placeholder(tf.float32, [None, args.z_dim])
    print (possible_combs)
    z1_latents = []
    z2_latents = []
    for comb in possible_combs:
        print comb[0], "------------------"
        print comb[1], "------------------"
        if comb[0] == "denoising":
            X1 = np.load("/DATA1/taskonomy-resized/noisy_rgb.npy").astype(np.float32)
            Y1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
        elif comb[0] == "autoencoding":
            X1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
            Y1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
        else:
            Y1 = np.load("/DATA1/taskonomy-resized/" + comb[0] + ".npy").astype(np.float32)
            X1 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)

        if len(Y1.shape) == 3:
            num_channels = 1
            Y1 = np.reshape(Y1, [-1, Y1.shape[1], Y1.shape[2], 1])
        else:
            num_channels = Y1.shape[3]
        vae = FactorVAE(args=args, num_channels=num_channels)
        if comb[1] == "denoising":
            X2 = np.load("/DATA1/taskonomy-resized/noisy_rgb.npy").astype(np.float32)
            Y2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
        elif comb[1] == "autoencoding":
            X2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
            Y2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)
        else:
            Y2 = np.load("/DATA1/taskonomy-resized/" + comb[1] + ".npy").astype(np.float32)
            X2 = np.load("/DATA1/taskonomy-resized/rgb.npy").astype(np.float32)

        if len(Y2.shape) == 3:
            num_channels = 1
            Y2 = np.reshape(Y2, [-1, Y2.shape[1], Y2.shape[2], 1])
        else:
            num_channels = Y2.shape[3]
        Y1 = (Y1 - np.min(Y1)) / (np.max(Y1) - np.min(Y1))
        X1 = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1))
        Y2 = (Y2 - np.min(Y2)) / (np.max(Y2) - np.min(Y2))
        X2 = (X2 - np.min(X2)) / (np.max(X2) - np.min(X2))
        checkpoint_dir = "./checkpoints/" + comb[0]
        vae.load_model(checkpoint_dir)
        mu1 = []
        cov1 = []
        mu2 = []
        cov2 = []
        for i in range(len(X1[7568:])):
            mu, cov = vae.get_distribution(X1[i].reshape((1, X1.shape[1], X1.shape[2], X1.shape[3])))
            mu1.append(mu)
            cov1.append(cov)
        checkpoint_dir = "./checkpoints/" + comb[1]
        vae.load_model(checkpoint_dir)
        for i in range(len(X2[7568:])):
            mu, cov = vae.get_distribution(X2[i].reshape((1, X2.shape[1], X2.shape[2], X2.shape[3])))
            mu2.append(mu)
            cov2.append(cov)
        mu1, mu2, cov1, cov2 = np.mean(mu1, axis=0), np.mean(mu2, axis=0), np.mean(cov1, axis=0), np.mean(cov2, axis=0)
        kl1 = kl_div(mu1, cov1, mu2, cov2)
        kl2 = kl_div(mu2, cov2, mu1, cov1)
        print ("comb = ", comb)
        print ("kl1 = ", kl1)
        print ("kl2 = ", kl2)
        with open("kl_results.txt", 'a') as kl_file:
            kl_file.write("Task 1: %s, Task 2: %s" % (comb[0], comb[1]))
            kl_file.write("1->2 kl loss: %f" % kl1)
            kl_file.write("2->1 kl loss: %f" % kl2)
            kl_file.write("################################################")
            kl_file.close()


if __name__ == '__main__':
    main()