import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.input_nc = args.input_nc
        self.ngf = args.ngf
        # Encoder
        self.e1 = nn.Conv2d(self.input_nc, self.ngf, kernel_size=5, stride=2, padding=3, padding_mode='circular')
        self.e2 = Encoder(self.ngf, self.ngf * 2)
        self.e3 = Encoder(self.ngf * 2, self.ngf * 4)
        self.e4 = Encoder(self.ngf * 4, self.ngf * 8)
        self.e5 = Encoder(self.ngf * 8, self.ngf * 8, stride=1)

        # Decoder
        self.d0 = Decoder(self.ngf * 8, self.ngf * 8, stride=1)
        self.d1 = Decoder(self.ngf * 8 * 2, self.ngf * 8, stride=1)
        self.d2 = Decoder(self.ngf * 8 * 2, self.ngf * 4)
        self.d3 = Decoder(self.ngf * 4 * 2, self.ngf * 2)
        self.d4 = Decoder(self.ngf * 2 * 2, self.ngf * 1)
        self.d5 = nn.ConvTranspose2d(self.ngf * 2, self.input_nc, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.d6 = nn.Tanh()

    def forward(self, img):
        """
        Layer Sizes after Conv2d for input img size (3, 256, 256) self.ngf = 64
        Encoder:
        0. (3, 256, 256)
        1. (ngf, 128, 128)
        2. (2*ngf, 64, 64)
        3. (4*ngf, 32, 32)
        4. (8*ngf, 16, 16)
        5. (8*ngf, 16, 16)
        6. (8*ngf, 16, 16)
        7. (8*ngf, 16, 16)
        8. (8*ngf, 16, 16)

        Decoder with skip connection
        1. (2*8*ngf, 16, 16)
        2. (2*8*ngf, 16, 16)
        3. (2*8*ngf, 16, 16)
        4. (2*8*ngf, 16, 16)
        5. (2*4*ngf, 32, 32)
        6. (2*2*ngf, 64, 64)
        7. (2*1*ngf, 128, 12
        8. (3, 256, 256)
        """
        e_layer1 = self.e1(img)
        e_layer2 = self.e2(e_layer1)
        e_layer3 = self.e3(e_layer2)
        e_layer4 = self.e4(e_layer3)
        e_layer5 = self.e5(e_layer4)
        e_layer6 = self.e5(e_layer5)
        e_layer7 = self.e5(e_layer6)
        e_layer8 = self.e5(e_layer7)

        d_layer1 = self.d0(e_layer8)
        d_layer1 = torch.cat([d_layer1, e_layer7], 1)

        d_layer2 = self.d1(d_layer1)
        d_layer2 = torch.cat([d_layer2, e_layer6], 1)

        d_layer3 = self.d1(d_layer2)
        d_layer3 = torch.cat([d_layer3, e_layer5], 1)

        d_layer4 = self.d1(d_layer3)
        d_layer4 = torch.cat([d_layer4, e_layer4], 1)

        d_layer5 = self.d2(d_layer4)
        d_layer5 = torch.cat([d_layer5, e_layer3], 1)

        d_layer6 = self.d3(d_layer5)
        d_layer6 = torch.cat([d_layer6, e_layer2], 1)

        d_layer7 = self.d4(d_layer6)
        d_layer7 = torch.cat([d_layer7, e_layer1], 1)

        d_layer8 = self.d5(d_layer7)
        d_layer9 = self.d6(d_layer8)

        # print(img.shape)
        # print(e_layer1.shape)
        # print(e_layer2.shape)
        # print(e_layer3.shape)
        # print(e_layer4.shape)
        # print(e_layer5.shape)
        # print(e_layer6.shape)
        # print(e_layer7.shape)
        # print(e_layer8.shape)
        # print("Decoder")
        # print(d_layer1.shape)
        # print(d_layer2.shape)
        # print(d_layer3.shape)
        # print(d_layer4.shape)
        # print(d_layer5.shape)
        # print(d_layer6.shape)
        # print(d_layer7.shape)
        # print(d_layer8.shape)
        return d_layer9


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.input_nc = args.input_nc * 2
        self.ndf = args.ndf
        self.net = [nn.Conv2d(self.input_nc, self.ndf, kernel_size=5, stride=2, padding=3, padding_mode='circular'),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=5, stride=2, padding=3, padding_mode='circular'),
                    nn.BatchNorm2d(self.ndf * 2, momentum=0.9),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=5, stride=2, padding=3, padding_mode='circular'),
                    nn.BatchNorm2d(self.ndf * 4, momentum=0.9),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=5, stride=1, padding=3, padding_mode='circular'),
                    nn.BatchNorm2d(self.ndf * 8, momentum=0.9),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.ndf * 8, 1, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
                    nn.Sigmoid()]

        self.model = nn.Sequential(*self.net)

    def forward(self, img):
        return self.model(img)


class Encoder(nn.Module):
    def __init__(self, c_in, c_out, k_size=5, stride=2, pad=0):
        super(Encoder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if stride == 2:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.ReflectionPad2d((1, 2, 1, 2)),
                nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad, padding_mode='circular'),
                nn.BatchNorm2d(self.c_out, momentum=0.9))
        else:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.ReflectionPad2d((2, 2, 2, 2)),
                nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                          padding_mode='circular'),
                nn.BatchNorm2d(self.c_out, momentum=0.9))

    def forward(self, maps):
        return self.model(maps)


class Decoder(nn.Module):
    def __init__(self, c_in, c_out, k_size=5, stride=2, pad=2):
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if stride == 2:
            self.model = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad, output_padding=1),
                nn.BatchNorm2d(self.c_out, momentum=0.9),
                nn.Dropout2d(0.5))
        else:
            self.model = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                                   output_padding=0),
                nn.BatchNorm2d(self.c_out, momentum=0.9),
                nn.Dropout2d(0.5))

    def forward(self, maps):
        return self.model(maps)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.BCELoss()

    def get_target_tensor(self, image, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(image)

    def __call__(self, img, target_is_real):
        target_tensor = self.get_target_tensor(img, target_is_real)
        return self.loss(img, target_tensor)


class DarkChannelLoss(nn.Module):
    def __init__(self, kernel_size=15):
        super(DarkChannelLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)

    def forward(self, x):
        # x : (B, 3, H, W), in [-1, 1]
        x = (x + 1.0) / 2.0
        H, W = x.size()[2], x.size()[3]

        # Minimum among three channels
        x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        x = self.unfold(x)  # (B, k*k, H*W)
        x = x.unsqueeze(1)  # (B, 1, k*k, H*W)

        # Minimum in (k, k) patch
        dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
        x = dark_map.view(-1, 1, H, W)

        # Count Zeros
        #y0 = torch.zeros_like(x)
        #y1 = torch.ones_like(x)
        #x = torch.where(x < 0.1, y0, y1)
        #x = torch.sum(x)
        #x = int(H * W - x)
        return x.clamp(min=0.0, max=0.1)

    def __call__(self, real, fake):
        real_map = self.forward(real)
        fake_map = self.forward(fake)
        return self.loss(real_map, fake_map)


class GradientLoss(nn.Module):
    def __init__(self, kernel_size=3, device="cpu"):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)
        self.device = device

    def forward(self, x):
        """
        Sobel Filter
        :param x:
        :return: dh, dv
        """
        # x : (B, 3, H, W)
        x = (x + 1.0) / 2.0

        # Compute a gray-scale image by averaging
        x = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        # weight :
        filter_h = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).expand(1, 1, 3, 3)
        filter_v = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).expand(1, 1, 3, 3)

        filter_h = filter_h.flip(-1).flip(-2)
        filter_v = filter_v.flip(-1).flip(-2)

        filter_h = filter_h.to(self.device)
        filter_v = filter_v.to(self.device)
        # Convolution
        gradient_h = F.conv2d(x, filter_h)  # .to(device)
        gradient_v = F.conv2d(x, filter_v)  # .to(device)

        return gradient_h, gradient_v

    def __call__(self, real, fake):
        real_grad_h, real_grad_v = self.forward(real)
        fake_grad_h, fake_grad_v = self.forward(fake)

        return self.loss(real_grad_h, fake_grad_h) + self.loss(real_grad_v, fake_grad_v)


