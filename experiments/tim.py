import inspect
from collections import OrderedDict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scipyfft
import sounddevice as sd
import soundfile as sf
import torch
from mlp64.st import *
from mlp64 import experiment, models
from tqdm import tqdm

sd.default.samplerate = 16000
librosa.fft.set_fftlib(scipyfft)

cuda = True if torch.cuda.is_available() else False

# Set parameters here
plot = True
# Saves model parameters to the experiment folder, can be quite large
save_model = False

sr = 16000
n_fft = 512
hop_length = 64

n_layers = 6
n_filters = 512
kernel_size = 2  # [2 ** i + 1 for i in range(1, n_layers + 1)]
padding = 0
bn = False
stride = 1
lr = 1
num_epochs = 200
div_stop_epoch = int(num_epochs / 2)

ac_bounds = (mstobin(200, hop_length), mstobin(2000, hop_length))
style_weights = [1 for i in range(n_layers)]
ac_weights = [1e3 for i in range(1, n_layers + 1)]
div_weights = [1e-4 for i in range(1, n_layers + 1)]

noise_stats = False

style_inst = "birds"
transform = lambda y: stft(y, n_fft, hop_length)
source, line = inspect.findsource(transform)
transform_string = source[line]

experiment_name = "ts_class_in1d"
notes = "Using the classifier trained on NSynth, with instance normalisation."

# Add new parameters to this dictionary!
params_dict = OrderedDict(
    {"num_epochs": num_epochs, "n_layers": n_layers, "kernel_size": kernel_size, "padding": padding,
     "bn": bn, "n_filters": n_filters, "n_fft": n_fft, "hop_length": hop_length, "stride": stride, "lr": lr,
     "div_stop_epoch": div_stop_epoch, "style_weight": style_weights, "ac_weights": ac_weights,
     "div_weights": div_weights, "noise_stats": noise_stats, "transform": transform_string})

# bpo = 60
# n_bins = 400
# cqt = lambda y: np.abs(librosa.cqt(y, sr=16000, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bpo))
# ms = lambda y: librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
transform = lambda y: stft(y, n_fft, hop_length)
# C = 10
# transform = lambda y: np.log1p(C * np.abs(librosa.stft(y, n_fft, hop_length))) / np.log1p(C)


s_file = "baselines/" + style_inst + ".wav"
style, _ = librosa.load(s_file, sr)

style_f = transform(style)

n_fft_bins = style_f.shape[0]
n_frames = style_f.shape[1]

# Plot original spectrogram
if plot:
    plt.imshow(style_f)
    plt.colorbar()
    plt.title("Style Spectrogram")
    plt.show()

model = models.resnet18(num_classes=11, n_fft_bins=257, classify=False)
exp = experiment.Experiment(model, "experiments/test", 40, None, None, continue_from_epoch=-2)
model = exp.model
model.fix_params()

style_torch = torch.from_numpy(style_f[None, :, :])
print(style_torch.shape)

if cuda:
    style_torch = style_torch.cuda()
    model = model.cuda()

loss_layers = ["0", "1", "2", "3", "4"]
targets = model(style_torch, loss_layers)

print("style_weights:", style_weights)
print("ac_weights:", ac_weights)
print("div_weights:", div_weights)
# The grams are computed inside the loss function now
style_loss = [StyleLoss(target.detach(), weight=w) for target, w in zip(targets, style_weights)]
# Think about changing bounds for multilayers because of receptive field
ac_loss = [AutoLoss(targets[0].detach(), weight=1000, bounds=(30, 100))]
# ac_loss = [AutoLoss(target.detach(), weight=w) for target, w in zip(targets, ac_weights)]
# Likewise, step needs to be decreased to work for different receptive fields
div_loss = [DivLoss(targets[0].detach(), weight=0.1, step=25)]
# div_loss = [DivLoss(target.detach(), weight=w) for target, w in zip(targets, div_weights)]

#     plt.show()
#     plt.imshow(style_loss[1].target.cpu().numpy().squeeze())
#     plt.title("Layer 1 Gram")
#     plt.show()

# Add other losses/weights here once they come
all_losses = [style_loss]  # , ac_loss, div_loss]  # + others

# Try scaling noise with output mean/std
# Also scale output
noise = torch.rand(style_torch.shape, device="cuda" if cuda else None, requires_grad=True)

optimizer = torch.optim.LBFGS([noise], lr=lr)  # , betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00002)
divloss_schedule = [1] * div_stop_epoch + [0] * (num_epochs - div_stop_epoch)

losses = []
loss_outer = np.inf
with tqdm(total=num_epochs) as pbar:
    for epoch in range(1, num_epochs + 1):
        def closure():
            optimizer.zero_grad()
            out = model(noise, loss_layers)
            layer_losses = [loss(x) for losses in all_losses for x, loss in zip(out, losses)]
            loss = sum(layer_losses)
            loss.backward()
            global loss_outer
            loss_outer = loss.item()
            return loss
        
        
        optimizer.step(closure)
        for loss in div_loss:
            loss.weight *= divloss_schedule[epoch - 1]
        pbar.set_description("loss: {:.4f}".format(loss_outer))
        pbar.update()
        losses.append(loss_outer)

gen_spectrum = noise.cpu().data.numpy().squeeze()
out = griffin_lim(gen_spectrum, hop_length)

experiment.log(model, experiment_name, gen_spectrum, out, losses, notes, "experiments", overwrite=True, sr=sr,
               save_model=save_model, **params_dict)
