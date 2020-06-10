from collections import OrderedDict

import inspect
from nnAudio import Spectrogram
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scipyfft
import soundfile as sf
import torch

from mlp64 import experiment
from mlp64.st import *
from tqdm import tqdm

librosa.fft.set_fftlib(scipyfft)

cuda = True if torch.cuda.is_available() else False

# Set parameters here
plot = True
# Saves model parameters to the experiment folder, can be quite large
save_model = False
# Overwrites old experiment folder if this is true. If you want to make sure to save all experiments, set this to false!
# Perhaps if true check before computing
overwrite = True

sr = 16000
n_fft = 512
hop_length = 128

n_layers = 6
n_filters = 128
# kernel_size = 2  # [(11, 11)]
#kernel_size = [(11, 5), (3, 35), (5, 5), (11, 11), (19, 19), (25, 25)]  # (11, 11)
kernel_size = [(256, 2), (53, 3), (3, 51), (5, 5), (11, 11), (19, 19), (27, 27)]  # (11, 11)
# padding = [(50, 1), (26, 1), (5, 2), (1, 1), (2, 2), (5, 5), (9, 9), (13, 13)]#[int(x / 2) for x in kernel_size]  # (5, 5)
padding = [(0, 0)] * 8
bn = False
stride = 1

lr = 0.5
num_epochs = 20
# style_weights = [1 / (2 ** i) for i in range(n_layers)]
style_weights = [1 for i in range(n_layers)]
ac_weights = [1 for i in range(1, n_layers + 1)]
ac_bounds = (mstobin(200, hop_length), mstobin(3000, hop_length))
# ac_bounds = (0, mstobin(2000, hop_length))
ac_epochs = int(num_epochs)
freq_weight = 1
freq_epochs = int(num_epochs)
business = 0.5 #torch.Tensor([1] * 100 + [0] * 157).cuda()

noise_stats = False

style_inst = "windchimes"

# Try window function in stft
window = torch.hann_window(n_fft)
# cqt = Spectrogram.CQT1992v2(sr=16000, hop_length=512, fmin=32.7, n_bins=84, bins_per_octave=12,
#                             norm=1, window='hann', center=True, pad_mode='reflect', trainable=False)
# transform = lambda x: cqt(x)[None, :]
# transform = lambda x: ri_spec(x, n_fft, hop_length, window)
transform = lambda x: stft_torch(x, n_fft, hop_length, window)

source, line = inspect.findsource(transform)
transform_string = source[line]

experiment_name = "chimes_own_128_business05"
notes = "See params"

# Add new parameters to this dictionary!
params_dict = OrderedDict(
    {"num_epochs": num_epochs, "n_layers": n_layers, "kernel_size": kernel_size, "padding": padding,
     "bn": bn, "n_filters": n_filters, "n_fft": n_fft, "hop_length": hop_length, "stride": stride, "lr": lr,
     "style_weight": style_weights, "noise_stats": noise_stats, "transform": transform_string})

# bpo = 60
# n_bins = 400
# cqt = lambda y: np.abs(librosa.cqt(y, sr=16000, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bpo))
# ms = lambda y: librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length)


s_file = "baselines/" + style_inst + ".wav"
style, _ = librosa.load(s_file, sr)
device = torch.cuda.current_device() if torch.cuda.device_count() >= 1 else torch.device("cpu")
# Figure out PyTorch device handling
style_f = transform(torch.from_numpy(style))  # .to(device))
print("Style (transformed shape):", style_f.shape)

n_fft_bins = style_f.shape[2]
n_frames = style_f.shape[3]

# Plot original spectrogram
if plot:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    if style_f.shape[1] > 1:
        im = ax1.imshow(stft_torch(torch.from_numpy(style), n_fft, hop_length, window).squeeze())
    else:
        im = ax1.imshow(style_f.cpu().squeeze())
    plt.colorbar(im, ax=ax1, fraction=0.047 * style_f.shape[2] / style_f.shape[3], pad=0.04)
    ax1.set_title("Style Spectrogram")
    ax2.plot(style)
    plt.show()

model = RandomCNN2d(n_channels_in=style_f.shape[1], n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers,
                    bias=False, bn=bn, padding=padding, stride=stride, activation=nn.ReLU(inplace=True))
# sets requires_grad to False - check for BN though, that may not work
model.fix_params()

if cuda:
    style_f = style_f.cuda()
    model = model.cuda()
    window = window.cuda()

loss_layers = model.layer_dict.keys()
targets = model(style_f, loss_layers)

print("style_weights:", style_weights)
style_loss = [TextureLoss(target.detach(), weight=w) for target, w in zip(targets, style_weights)]
# ac_loss = [AutoLoss(target.detach(), weight=w) for target, w in zip(targets, ac_weights)]
# ac_loss = AutoLoss(torch.from_numpy(style).cuda(), weight=0.00001, bounds=[3200, 32000])
ac_loss = AutoLoss(style_f, weight=ac_weights[0], bounds=ac_bounds)  # , bounds=[3200, 32000])
ac_schedule = [ac_weights[0]] * ac_epochs + [0] * (num_epochs - ac_epochs)
freq_loss = FreqLoss(style_f, weight=freq_weight, business=business)
freq_schedule = [freq_weight] * freq_epochs + [0] * (num_epochs - freq_epochs)

# Add other losses/weights here once they come
all_losses = [style_loss]
# all_losses = [style_loss, ac_loss]

# noise = create_noise(style_f, noise_stats=noise_stats)
noise = torch.randn(style.shape, device="cuda" if cuda else None, requires_grad=True)

optimizer = torch.optim.LBFGS([noise], lr=lr)  # , betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = torch.optim.LBFGS([noise], lr=lr)  # , betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
# Multiplying by this will set the weights to 0. Alternatively we can set directly, but then we need to get the weights
# out of the declaration above and interpolate here.

losses = []
loss_outer = np.inf
with tqdm(total=num_epochs) as pbar:
    for epoch in range(1, num_epochs + 1):
        ac_loss.weight = ac_schedule[epoch - 1]
        freq_loss.weight = freq_schedule[epoch - 1]
        
        
        def closure():
            optimizer.zero_grad()
            spec = transform(noise)
            out = model(spec, loss_layers)
            layer_losses = [loss(x) for losses in all_losses for x, loss in zip(out, losses)]
            loss = sum(layer_losses) + ac_loss(spec) + freq_loss(spec)
            loss.backward()
            global loss_outer
            loss_outer = loss.item()
            return loss
        
        
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        # im = ax1.imshow(transform(noise).detach().cpu().squeeze())
        # plt.colorbar(im, ax=ax1, fraction=0.047 * style_f.shape[0] / style_f.shape[1], pad=0.04)
        # ax1.set_title("Output Spectrogram")
        # ax2.plot(noise.detach().cpu())
        # plt.show()
        optimizer.step(closure)
        lr_scheduler.step(loss_outer)
        pbar.set_description("loss: {:.4f}".format(loss_outer))
        pbar.update()
        losses.append(loss_outer)

# Can change this to ri to later plot RI spectrograms
out = noise.detach().cpu().numpy()
gen_spectrum = stft_torch(noise, n_fft, hop_length, window).detach().cpu().squeeze()

if plot:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    im = ax1.imshow(gen_spectrum)
    plt.colorbar(im, ax=ax1, fraction=0.047 * style_f.shape[0] / style_f.shape[1], pad=0.04)
    ax1.set_title("Spectrogram " + experiment_name)
    ax2.plot(out)
    plt.show()
    # plt.figure()
    # plt.imshow(transform(noise).detach().cpu().squeeze())
    # plt.show()

sf.write("experiments/output/" + experiment_name + ".wav", out, samplerate=sr)

experiment.log(model, experiment_name, gen_spectrum, out, losses, notes, "experiments", overwrite=overwrite, sr=sr,
               save_model=save_model, show_plot=False, **params_dict)
