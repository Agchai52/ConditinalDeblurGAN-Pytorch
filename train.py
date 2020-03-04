from __future__ import print_function  # help to use print() in python 2.x
import os
from math import log10
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import *
from network import *
from Dataset import DeblurDataset

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def train(args):
    print('===> Loading datasets')
    f_train = open("./dataset/AidedDeblur/train_instance_names.txt", "r")
    f_test = open("./dataset/AidedDeblur/test_instance_names.txt", "r")

    train_data = f_train.readlines()
    test_data = f_test.readlines()

    f_train.close()
    f_test.close()

    train_data = [line.rstrip() for line in train_data]
    test_data = [line.rstrip() for line in test_data]

    train_data_loader = DataLoader(DeblurDataset(train_data, args), batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(DeblurDataset(test_data, args, False), batch_size=args.batch_size, shuffle=False)

    print('===> Building models')
    net_G = Generator(args)
    net_D = Discriminator(args)
    net_D.apply(weights_init)
    net_G.apply(weights_init)

    print(net_G)
    print(net_D)

    print('===> Setting up loss functions')
    criterion_L2 = nn.MSELoss()
    criterion_GAN = GANLoss()
    criterion_DarkChannel = DarkChannelLoss()
    criterion_Gradient = GradientLoss()

    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    counter = 0
    G_losses = []
    D_losses = []
    L2_losses = []
    Grad_losses = []
    DarkCha_losses = []

    print('===> Training')
    for epoch in range(args.epoch):
        for iteration, batch in enumerate(train_data_loader, 1):
            real_A, real_B, img_name = batch[0], batch[1], batch[2]
            fake_B = net_G(real_A)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizer_D.zero_grad()

            # train with fake
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = net_D(fake_AB.detach())
            loss_d_fake = criterion_GAN(pred_fake, False)

            # train with real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = net_D(real_AB)
            loss_d_real = criterion_GAN(pred_real, True)

            # combine d loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            optimizer_G.zero_grad()

            # G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = net_D(fake_AB)
            loss_g_gan = criterion_GAN(pred_fake, True)

            # G(A) = B
            loss_g_l2 = criterion_L2(fake_B, real_B) * args.L1_lambda
            loss_g_darkCh = criterion_DarkChannel(fake_B, real_B) * args.dark_channel_lambda
            loss_g_grad = criterion_Gradient(fake_B, real_B) * args.L1_lambda

            loss_g = loss_g_gan \
                     + (loss_g_l2 + loss_g_grad)  \
                     + loss_g_darkCh

            loss_g.backward()
            optimizer_G.step()
            counter += 1

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_L2: {:.4f} Loss_Grad: {:.4f} Loss_Dark: {:.4f}".format(
            epoch, iteration, len(train_data_loader), loss_d.item(), loss_g.item(), loss_g_l2, loss_g_grad, loss_g_darkCh))

            G_losses.append(loss_g.item())
            D_losses.append(loss_d.item())
            L2_losses.append(loss_g_l2.item())
            Grad_losses.append(loss_g_grad.item())
            DarkCha_losses.append(loss_g_darkCh.item())

            if (counter % 500 == 1) or ((epoch == args.epoch - 1) and (iteration == len(train_data_loader) - 1)):
                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")
                if not os.path.exists(os.path.join("checkpoint", args.dataset_name + "/netG")):
                    os.mkdir(os.path.join("checkpoint", args.dataset_name + "/netG"))
                    os.mkdir(os.path.join("checkpoint", args.dataset_name + "/netD"))
                net_G_save_path = "checkpoint/{}/netG/G_model_step_{}.pth".format(args.dataset_name, counter)
                net_D_save_path = "checkpoint/{}/netD/D_model_step_{}.pth".format(args.dataset_name, counter)
                torch.save(net_G, net_G_save_path)
                torch.save(net_D, net_D_save_path)
                print("Checkpoint saved to {}".format("checkpoint/" + args.dataset_name))

        all_psnr = []
        for batch in test_data_loader:
            real_A, real_B, img_name = batch[0], batch[1], batch[2]
            pred_B = net_G(real_A)
            if epoch == args.epoch - 1 and img_name[0][-2:] == '01':
                img_B = pred_B.detach().squeeze(0)
                save_img(img_B, '{}/test_'.format(args.test_dir) + img_name[0])
            real_B = (real_B + 1.0) / 2.0
            pred_B = (pred_B + 1.0) / 2.0
            mse = criterion_L2(pred_B, real_B)
            psnr = 10 * log10(1 / mse.item())
            all_psnr.append(psnr)

        print("===> Avg. PSNR: {:.4f} dB".format(sum(all_psnr) / len(test_data_loader)))

    print("===> Saving Losses")
    plt.figure()
    plt.plot(D_losses[0:-1:100], 'r-', label='d_loss')
    plt.xlabel("iteration*100")
    plt.ylabel("Error")
    #plt.xlim(xmin=-5, xmax=300)  # xmax=300
    #plt.ylim(ymin=0, ymax=60)  # ymax=60
    plt.title("Discriminator Loss")
    plt.savefig("plot_d_loss.jpg")

    plt.figure()
    plt.plot(G_losses[0:-1:100], 'g-', label='g_loss')
    plt.xlabel("iteration*100")
    plt.ylabel("Error")
    #plt.xlim(xmin=-5, xmax=300)
    #plt.ylim(ymin=0, ymax=60)
    plt.title("Generator Loss")
    plt.savefig("plot_g_loss.jpg")

    plt.figure()
    plt.plot(L2_losses[0:-1:100], 'b-', label='l2_loss')
    plt.plot(Grad_losses[0:-1:100], 'g-', label='grad_loss')
    plt.plot(DarkCha_losses[0:-1:100], 'r-', label='dc_loss')
    plt.xlabel("iteration*100")
    plt.ylabel("Error")
    #plt.xlim(xmin=-5, xmax=480)
    #plt.ylim(ymin=0, ymax=16)
    plt.title("DarkChannel Loss")
    plt.savefig("plot_3g_losses.jpg")
    #plt.show()

    print("===> Training finished")






