from __future__ import division
import os
import numpy as np
import math
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename+'.png')


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def find_latest_model(net_path):
    file_list = os.listdir(net_path)
    model_names = [int(f[14:-4]) for f in file_list if ".pth" in f]
    if len(model_names) == 0:
        return False
    else:
        iter_num = max(model_names)
        if net_path[-2] == '2':
            return os.path.join(net_path, "G_model_epoch_{}.pth".format(iter_num))
        elif net_path[-2] == '_':
            return os.path.join(net_path, "D_model_epoch_{}.pth".format(iter_num))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def plot_losses():
    loss_record = "loss_record.txt"
    psnr_record = "psnr_record.txt"
    ddg_record = "ddg_record.txt"

    losses_dg = np.loadtxt(loss_record)
    psnr_ave = np.loadtxt(psnr_record)
    ddg_ave = np.loadtxt(ddg_record)

    plt.figure()
    plt.plot(losses_dg[0:-1:100, 0], 'r-', label='d_loss')
    plt.xlabel("iteration*100")
    plt.ylabel("Error")
    #plt.xlim(xmin=-5, xmax=300)  # xmax=300
    #plt.ylim(ymin=0, ymax=60)  # ymax=60
    plt.title("Discriminator Loss")
    plt.savefig("plot_d_loss.jpg")

    plt.figure()
    plt.plot(losses_dg[0:-1:100, 1], 'g-', label='g_loss')
    plt.xlabel("iteration*100")
    plt.ylabel("Error")
    #plt.xlim(xmin=-5, xmax=300)
    #plt.ylim(ymin=0, ymax=60)
    plt.title("Generator Loss")
    plt.savefig("plot_g_loss.jpg")

    plt.figure()
    plt.plot(losses_dg[0:-1:100, 3], 'b--', label='l2_loss')
    plt.plot(losses_dg[0:-1:100, 4], 'g:', label='grad_loss')
    plt.plot(losses_dg[0:-1:100, 5], 'r-', label='dc_loss')
    plt.plot(losses_dg[0:-1:100, 6], 'y-', label='cycle_loss')
    plt.plot(losses_dg[0:-1:100, 2], 'k-', label='gan_loss')
    plt.xlabel("iteration*100")
    plt.ylabel("Error")
    plt.legend()
    # plt.xlim(xmin=-5, xmax=480)
    # plt.ylim(ymin=0, ymax=16)
    plt.title("L2_Grad_DarkChan Loss")
    plt.savefig("plot_4g_losses.jpg")
    # plt.show()

    plt.figure()
    plt.plot(psnr_ave, 'r-')
    plt.xlabel("epochs")
    plt.ylabel("Average PSNR")
    # plt.xlim(xmin=-5, xmax=300)  # xmax=300
    # plt.ylim(ymin=0, ymax=60)  # ymax=60
    plt.title("Validation PSNR")
    plt.savefig("plot_psnr_loss.jpg")

    plt.figure()
    plt.plot(ddg_ave[:, 0], 'b-', label='d_fake')
    plt.plot(ddg_ave[:, 1], 'r-', label='d_real')
    plt.plot(ddg_ave[:, 2], 'g-', label='gan')
    plt.xlabel("epochs")
    plt.ylabel("Average loss")
    plt.legend()
    # plt.xlim(xmin=-5, xmax=300)  # xmax=300
    # plt.ylim(ymin=0, ymax=60)  # ymax=60
    plt.title("D1_D2_G PSNR")
    plt.savefig("plot_ddg_loss.jpg")

#plot_losses()
