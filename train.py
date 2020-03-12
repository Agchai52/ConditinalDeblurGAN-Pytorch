from __future__ import print_function  # help to use print() in python 2.x
import os
import itertools
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
    net_g_bs_path = "checkpoint/{}/netG_B2S".format(args.dataset_name)
    net_g_sb_path = "checkpoint/{}/netG_S2B".format(args.dataset_name)
    net_d_s_path = "checkpoint/{}/netD_S".format(args.dataset_name)
    net_d_b_path = "checkpoint/{}/netD_B".format(args.dataset_name)
    if not find_latest_model(net_g_bs_path) or not find_latest_model(net_d_s_path):
        print(" [!] Load failed...")
        netG_B2S = Generator(args).to(device)
        netG_S2B = Generator(args).to(device)
        netD_B = Discriminator(args, device).to(device)
        netD_S = Discriminator(args, device).to(device)

        netG_B2S.apply(weights_init)
        netG_S2B.apply(weights_init)
        netD_B.apply(weights_init)
        netD_S.apply(weights_init)

    else:
        print(" [*] Load SUCCESS")
        model_path_G_BS = find_latest_model(net_g_bs_path)
        model_path_G_SB = find_latest_model(net_g_sb_path)
        model_path_D_B = find_latest_model(net_d_b_path)
        model_path_D_S = find_latest_model(net_d_s_path)
        netG_B2S = torch.load(model_path_G_BS).to(device)
        netG_S2B = torch.load(model_path_G_SB).to(device)
        netD_B = torch.load(model_path_D_B).to(device)
        netD_S = torch.load(model_path_D_S).to(device)

    print(netG_B2S)
    print(netD_B)

    print('===> Setting up loss functions')
    # Losses
    criterion_L2 = nn.MSELoss().to(device)
    criterion_GAN = GANLoss().to(device)
    criterion_Cycle = torch.nn.L1Loss()
    criterion_DarkChannel = DarkChannelLoss().to(device)
    criterion_Gradient = GradientLoss(device=device).to(device)

    # Optimizers
    optimizer_G = optim.Adam(itertools.chain(netG_B2S.parameters(), netG_S2B.parameters()), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    optimizer_D_S = optim.Adam(netD_S.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)

    #lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epoch, args.epoch_start, args.epoch_decay).step)
    #lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(args.epoch, args.epoch_start, args.epoch_decay).step)

    #params = net_G.parameters()
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
            fake_B = netG_B2S(real_A)
            fake_A = netG_S2B(real_B)

            ############################
            # (1) Update D_S network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizer_D_S.zero_grad()

            # train with fake
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake_B = netD_S(fake_AB.detach())
            loss_d_s_fake = criterion_GAN(pred_fake_B, False)

            # train with real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real_B = netD_S(real_AB)
            loss_d_s_real = criterion_GAN(pred_real_B, True)

            # combine d loss
            loss_d_s = (loss_d_s_fake + loss_d_s_real)

            loss_d_s.backward()
            optimizer_D_S.step()

            ############################
            # (2) Update D_B network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizer_D_B.zero_grad()

            # train with fake
            fake_BA = torch.cat((real_B, fake_A), 1)
            pred_fake_A = netD_S(fake_BA.detach())
            loss_d_b_fake = criterion_GAN(pred_fake_A, False)

            # train with real
            real_BA = torch.cat((real_B, real_A), 1)
            pred_real_A = netD_S(real_BA)
            loss_d_b_real = criterion_GAN(pred_real_A, True)

            # combine d loss
            loss_d_b = (loss_d_b_fake + loss_d_b_real)

            loss_d_b.backward()
            optimizer_D_B.step()

            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            optimizer_G.zero_grad()

            # G_B2S(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake_B = net_D(fake_AB)
            loss_g_gan_bs = criterion_GAN(pred_fake_B, True)

            # G_B2S(A) should fake the discriminator
            fake_BA = torch.cat((real_B, fake_A), 1)
            pred_fake_A = net_D(fake_BA)
            loss_g_gan_sb = criterion_GAN(pred_fake_A, True)

            # Cycle
            recovered_A = netG_S2B(fake_B)
            recovered_B = netG_B2S(fake_A)

            # G(A) = B
            loss_g_gan = loss_g_gan_sb + loss_g_gan_bs
            loss_cycle = criterion_Cycle(recovered_A, real_A) + criterion_Cycle(recovered_B, real_B)
            loss_g_l2 = (criterion_L2(fake_B, real_B) + criterion_L2(fake_A, real_A)) * args.L1_lambda
            loss_g_darkCh = (criterion_DarkChannel(fake_B, real_B) + criterion_DarkChannel(fake_A, real_A)) * args.dark_channel_lambda
            loss_g_grad = criterion_Gradient(fake_B, real_B) + criterion_Gradient(fake_A, real_A)

            loss_g = loss_g_gan \
                     + (loss_g_l2 + loss_g_grad) \
                     + loss_g_darkCh \
                     + loss_cycle

            loss_g.backward(retain_graph=True)
            optimizer_G.step()
#
            ############################
            # (4) Update G network: maximize log(D(G(z)))
            ###########################
            optimizer_G.zero_grad()

            # G_B2S(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake_B = net_D(fake_AB)
            loss_g_gan_bs = criterion_GAN(pred_fake_B, True)

            # G_B2S(A) should fake the discriminator
            fake_BA = torch.cat((real_B, fake_A), 1)
            pred_fake_A = net_D(fake_BA)
            loss_g_gan_sb = criterion_GAN(pred_fake_A, True)

            # Cycle
            recovered_A = netG_S2B(fake_B)
            recovered_B = netG_B2S(fake_A)

            # G(A) = B
            loss_g_gan = loss_g_gan_sb + loss_g_gan_bs
            loss_cycle = criterion_Cycle(recovered_A, real_A) + criterion_Cycle(recovered_B, real_B)
            loss_g_l2 = (criterion_L2(fake_B, real_B) + criterion_L2(fake_A, real_A)) * args.L1_lambda
            loss_g_darkCh = (criterion_DarkChannel(fake_B, real_B) + criterion_DarkChannel(fake_A,
                                                                                           real_A)) * args.dark_channel_lambda
            loss_g_grad = criterion_Gradient(fake_B, real_B) + criterion_Gradient(fake_A, real_A)

            loss_g = loss_g_gan \
                     + (loss_g_l2 + loss_g_grad) \
                     + loss_g_darkCh \
                     + loss_cycle

            loss_g.backward()
            optimizer_G.step()

            #if counter % 100 == 1:
            #    print('Current Learning rate is:')
            #    print(lr_scheduler_G.get_lr())
            counter += 1

            print("===> Epoch[{}]({}/{}): Loss_DB: {:.4f} Loss_DS: {:.4f} Loss_G: {:.4f} Loss_GAN: {:.4f} Loss_L2: {:.4f} Loss_Grad: {:.4f} Loss_Dark: {:.4f}".format(
            epoch, iteration, len(train_data_loader), loss_d_b.item(), loss_d_b.item(), loss_g.item(), loss_g_gan.item(), loss_g_l2.item(), loss_g_grad.item(), loss_g_darkCh.item()))
            cur_d1.append(loss_d_s_fake.item())
            cur_d2.append(loss_d_s_real.item())
            cur_g.append(loss_g_gan_bs.item())

            # To record losses in a .txt file
            losses_dg = [loss_d_b.item(), loss_g.item(), loss_g_gan.item(), loss_g_l2.item(), loss_g_grad.item(), loss_g_darkCh.item()]
            losses_dg_str = " ".join(str(v) for v in losses_dg)

            with open(loss_record, 'a+') as file:
                file.writelines(losses_dg_str + "\n")

            if (counter % 500 == 1) or ((epoch == args.epoch - 1) and (iteration == len(train_data_loader) - 1)):
                net_g_bs_save_path = net_g_bs_path + "/G_model_epoch_{}.pth".format(epoch)
                net_g_sb_save_path = net_g_sb_path + "/G_model_epoch_{}.pth".format(epoch)
                net_d_s_save_path = net_d_s_path + "/B_model_epoch_{}.pth".format(epoch)
                net_d_b_save_path = net_d_b_path + "/B_model_epoch_{}.pth".format(epoch)

                torch.save(netG_B2S, net_g_bs_save_path)
                torch.save(netG_S2B, net_g_sb_save_path)
                torch.save(netD_S, net_d_s_save_path)
                torch.save(netD_B, net_d_b_save_path)
                print("Checkpoint saved to {}".format("checkpoint/" + args.dataset_name))

        # Update Learning rate
        #lr_scheduler_G.step()
        #lr_scheduler_D.step()

        ddg = [sum(cur_d1) / len(train_data_loader), sum(cur_d2) / len(train_data_loader), sum(cur_g) / len(train_data_loader)]
        ddg_str = " ".join(str(v) for v in ddg)
        with open(ddg_record, 'a+') as file:
            file.writelines(ddg_str + "\n")

        all_psnr = []
        for batch in test_data_loader:
            real_A, real_B, img_name = batch[0].to(device), batch[1].to(device), batch[2]
            pred_B = netG_B2S(real_A)
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






