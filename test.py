from __future__ import print_function
import os
import time
import torch
import torchvision.transforms as transforms
from Dataset import DeblurDataset
from torch.utils.data import DataLoader
from utils import *
from skimage.measure import compare_ssim as ssim


def test(args):
    device = torch.device('cuda:{}'.format(args.gpu) if (torch.cuda.is_available() and args.gpu > 0) else "cpu")
    print("====> Loading model")
    net_g_path = "checkpoint/{}/netG".format(args.dataset_name)
    if not find_latest_model(net_g_path):
        print(" [!] Load failed...")
        raise Exception('No model to load for testing!')
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        net_G = torch.load(model_path_G).to(device)
        print(model_path_G)

    print("====> Loading data")
    ############################
    # For AidedDeblur dataset
    ###########################
    f_test = open("./dataset/AidedDeblur/test_instance_names.txt", "r")
    test_data = f_test.readlines()
    test_data = [line.rstrip() for line in test_data]
    f_test.close()
    test_data_loader = DataLoader(DeblurDataset(test_data, args, False), batch_size=args.batch_size, shuffle=False)

    ############################
    # For Other datasets
    ###########################
    # image_dir = "dataset/{}/test/a/".format(args.dataset_name)
    # image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
    # transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # transform = transforms.Compose(transform_list)
    # for image_name in image_filenames:

    all_psnr = []
    all_ssim = []
    start_time = time.time()
    for batch in test_data_loader:
        real_A, real_B, img_name = batch[0].to(device), batch[1].to(device), batch[2]
        pred_B = net_G(real_A)
        if img_name[0][-2:] == '01':
            img_B = pred_B.detach().squeeze(0).cpu()
            if not os.path.exists("result"):
                os.makedirs("result")
            save_img(img_B, 'result/test_' + img_name[0])
        real_B = real_B.squeeze(0).permute(1, 2, 0).cpu()
        pred_B = pred_B.detach().squeeze(0).permute(1, 2, 0).cpu()
        real_B = real_B.float().numpy()
        pred_B = pred_B.float().numpy()
        real_B = (real_B + 1.0) / 2.0
        pred_B = (pred_B + 1.0) / 2.0
        cur_psnr = psnr(real_B, pred_B)
        cur_ssim = ssim(real_B, pred_B, gaussian_weights=True, multichannel=True, use_sample_covariance=False)
        all_psnr.append(cur_psnr)
        all_ssim.append(cur_ssim)
        if img_name[0][-2:] == '01':
            print('test_{}: PSNR = {} dB'.format(img_name[0], psnr))
        #print("Image {}, PSNR = {}, SSIM = {}".format(img_name[0], cur_psnr, cur_ssim))

    total_time = time.time() - start_time
    ave_psnr = sum(all_psnr)/len(test_data_loader)
    ave_ssim = sum(all_ssim)/len(test_data_loader)
    ave_time = total_time/len(test_data_loader)
    print("Average PSNR = {}, SSIM = {}, Processing time = {}".format(ave_psnr, ave_ssim, ave_time))



