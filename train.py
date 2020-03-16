from __future__ import print_function  # help to use print() in python 2.x
import os
from math import log10
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import *
from network import *
from Dataset import DeblurDataset


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

    device = torch.device('cuda:{}'.format(args.gpu) if (torch.cuda.is_available() and args.gpu > 0) else "cpu")

    print('===> Building models')
    net_g_path = "checkpoint/{}/netG".format(args.dataset_name)
    net_d_path = "checkpoint/{}/netD".format(args.dataset_name)
    if not find_latest_model(net_g_path) or not find_latest_model(net_d_path):
        print(" [!] Load failed...")
        net_G = Generator(args).to(device)
        net_D = Discriminator(args, device).to(device)

        net_D.apply(weights_init)
        net_G.apply(weights_init)
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        model_path_D = find_latest_model(net_d_path)
        net_G = torch.load(model_path_G).to(device)
        net_D = torch.load(model_path_D).to(device)

    print(net_G)
    print(net_D)

    print('===> Setting up loss functions')
    criterion_L2 = nn.MSELoss().to(device)
    criterion_GAN = GANLoss().to(device)
    criterion_DarkChannel = DarkChannelLoss().to(device)
    criterion_Gradient = GradientLoss(device=device).to(device)

    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)

    #lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epoch, args.epoch_start, args.epoch_decay).step)
    #lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(args.epoch, args.epoch_start, args.epoch_decay).step)

    params = net_G.parameters()
    counter = 0
    PSNR_average = []
    #D_loss = []
    #G_loss = []

    loss_record = "loss_record.txt"
    psnr_record = "psnr_record.txt"
    ddg_record = "ddg_record.txt"
    print('===> Training')
    for epoch in range(args.epoch):
        cur_d1 = []
        cur_d2 = []
        cur_g = []
        for iteration, batch in enumerate(train_data_loader, 1):
            real_A, real_B, img_name = batch[0].to(device), batch[1].to(device), batch[2]
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
            loss_d = (loss_d_fake + loss_d_real)

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
            loss_g_grad = criterion_Gradient(fake_B, real_B)

            loss_g = loss_g_gan \
                     + (loss_g_l2 + loss_g_grad) \
                     + loss_g_darkCh

            loss_g.backward(retain_graph=True)
            optimizer_G.step()
#
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
            loss_g_grad = criterion_Gradient(fake_B, real_B)

            loss_g = loss_g_gan \
                     + (loss_g_l2 + loss_g_grad) \
                     + loss_g_darkCh

            loss_g.backward()
            optimizer_G.step()

            #if counter % 100 == 1:
            #    print('Current Learning rate is:')
            #    print(lr_scheduler_G.get_lr())
            counter += 1

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_GAN: {:.4f} Loss_L2: {:.4f} Loss_Grad: {:.4f} Loss_Dark: {:.4f}".format(
            epoch, iteration, len(train_data_loader), loss_d.item(), loss_g.item(), loss_g_gan.item(), loss_g_l2.item(), loss_g_grad.item(), loss_g_darkCh.item()))

            cur_d1.append(loss_d_fake.item())
            cur_d2.append(loss_d_real.item())
            cur_g.append(loss_g_gan.item())

            # To record losses in a .txt file
            losses_dg = [loss_d.item(), loss_g.item(), loss_g_gan.item(), loss_g_l2.item(), loss_g_grad.item(), loss_g_darkCh.item()]
            losses_dg_str = " ".join(str(v) for v in losses_dg)

            with open(loss_record, 'a+') as file:
                file.writelines(losses_dg_str + "\n")

            if (counter % 500 == 1) or ((epoch == args.epoch - 1) and (iteration == len(train_data_loader) - 1)):
                net_G_save_path = "checkpoint/{}/netG/G_model_epoch_{}.pth".format(args.dataset_name, epoch)
                net_D_save_path = "checkpoint/{}/netD/D_model_epoch_{}.pth".format(args.dataset_name, epoch)
                torch.save(net_G, net_G_save_path)
                torch.save(net_D, net_D_save_path)
                print("Checkpoint saved to {}".format("checkpoint/" + args.dataset_name))

        # Update Learning rate
        #lr_scheduler_G.step()
        #lr_scheduler_D.step()

        ddg = [sum(cur_d1) / len(train_data_loader), sum(cur_d2) / len(train_data_loader),
               sum(cur_g) / len(train_data_loader)]
        ddg_str = " ".join(str(v) for v in ddg)
        with open(ddg_record, 'a+') as file:
            file.writelines(ddg_str + "\n")

        all_psnr = []
        for batch in test_data_loader:
            real_A, real_B, img_name = batch[0].to(device), batch[1].to(device), batch[2]
            pred_B = net_G(real_A)
            if epoch == args.epoch - 1 and img_name[0][-2:] == '01':
                img_B = pred_B.detach().squeeze(0).cpu()
                save_img(img_B, '{}/test_'.format(args.test_dir) + img_name[0])
            real_B = (real_B + 1.0) / 2.0
            pred_B = (pred_B + 1.0) / 2.0
            mse = criterion_L2(pred_B, real_B)
            psnr = 10 * log10(1 / mse.item())
            if epoch == args.epoch - 1 and img_name[0][-2:] == '01':
                print('test_{}: PSNR = {} dB'.format(img_name[0], psnr))
            all_psnr.append(psnr)
        PSNR_average.append(sum(all_psnr) / len(test_data_loader))
        with open(psnr_record, 'a+') as file:
            file.writelines(str(sum(all_psnr) / len(test_data_loader)) + "\n")
        print("===> Avg. PSNR: {:.4f} dB".format(sum(all_psnr) / len(test_data_loader)))

    print("===> Average Validation PSNR for each epoch")
    print(PSNR_average)

    print("===> Saving Losses")
    plot_losses()
    print("===> Training finished")






