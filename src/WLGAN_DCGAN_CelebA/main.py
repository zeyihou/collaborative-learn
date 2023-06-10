import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dcgan import Generator, Discriminator # can replace with other GAN architectures
from calibration import LRcalibrator
from collaborative_learn import collaborative_learn

from inception_score import inception_score
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--load-g', required=True, help='path for the generator file')
parser.add_argument('--load-d', required=True, help='path for the discriminator file')
parser.add_argument('--image-size', type=int, default=64, help='image size (input/output size for the discriminator/generator')
parser.add_argument('--ndf', type=int, default=64, help='num features in discriminator')
parser.add_argument('--ngf', type=int, default=64, help='num features in generator')
parser.add_argument('--nz', type=int, default=100, help='latent space dimensions')
parser.add_argument('--nc', type=int, default=3, help='number of image channels')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus')
parser.add_argument('--batch-size', type=int, default=100, help='batch size for sampling')
parser.add_argument('--num-images', type=int, default=50000, help='total numbers of required samples')
parser.add_argument('--calibrate', action='store_true', help='whether to calibrate the discriminator scores (if true, use LR, else, use id mapping)')
parser.add_argument('--clen', type=int, default=640, help='length of each Markov chain')
parser.add_argument('--tau', type=float, default=0.1, help='Langevin step size in L2MC')
parser.add_argument('--eta', type=float, default=0.3162, help='scale of white noise (default to sqrt(tau))')
opt = parser.parse_args()

# load model
device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
netG = Generator(opt.ngpu, opt.nz, opt.ngf, opt.nc).to(device)
netD = Discriminator(opt.ngpu, opt.ndf, opt.nc).to(device)
netG.load_state_dict(torch.load(opt.load_g, map_location=device))
netD.load_state_dict(torch.load(opt.load_d, map_location=device))
print('model loaded')
torch.set_grad_enabled(False)

# discriminator calibration
if opt.calibrate: # use an LR calibrator 
    dataset = datasets.ImageFolder(root=opt.dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    calibrator = LRcalibrator(netG, netD, data_loader, device, nz=opt.nz)
else:
    calibrator = torch.nn.Identity() # no calibration

print('start collaborative_learn sampling')
accepted_samples = []
for i in tqdm(range(0, opt.num_images, opt.batch_size)):
    samples = collaborative_learn(netG, netD, calibrator, device, opt.nz, opt.batch_size, opt.clen, opt.tau, opt.eta)
    accepted_samples.append(samples.cpu())

#conpute inception_score
IS =inception_score( accepted_samples, cuda=True, batch_size=32, resize=True, splits=1 )
print("collaborative_learn inception score:{}".format(IS))