from collections import OrderedDict
import inspect
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from time import time

from mlp64 import experiment
from mlp64 import resnet2d
from mlp64.st import *
from tqdm import tqdm

loss_outer = loss_g = loss_ac = loss_f = 0


def run_baseline(style_inst, random_cnn=True, decay=True, net_type="2D", opt=torch.optim.LBFGS,
                 sr=16000, plot=False, n_fft=512, hop_length=128, transform="STFT",
                 n_layers=6, n_filters=512, kernel_size=11, padding=0, bn=False,
                 stride=1, lr=1, num_epochs=20, ac_bounds=None, style_weights=None,
                 ac_weight=0, ac_on="spec", freq_weight=0, synth_waveform=True,
                 noise_stats=False, experiment_name="default", notes="", experiment_folder="experiments"):
    cuda = True if torch.cuda.is_available() else False
    
    save_model = False
    overwrite = True
    
    if ac_bounds == None:
        ac_bounds = (mstobin(200, hop_length), mstobin(2000, hop_length))
    
    if style_weights is None:
        if decay:
            style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        else:
            style_weights = [1 for i in range(n_layers)]
    
    window = torch.hann_window(n_fft)
    
    ac_epochs = int(num_epochs / 4)
    freq_weight = freq_weight
    freq_epochs = int(num_epochs / 4)
    
    if transform == "RI":
        transform = lambda x: ri_spec(x, n_fft, hop_length, window)
    else:
        if net_type == "2D":
            transform = lambda x: stft_torch(x, n_fft, hop_length, window)
        else:
            transform = lambda x: stft_torch(x, n_fft, hop_length, window, out_dims=3)
    
    source, line = inspect.findsource(transform)
    transform_string = source[line]
    
    params_dict = OrderedDict(
        {"sound": style_inst, "num_epochs": num_epochs, "n_layers": n_layers, "kernel_size": kernel_size,
         "padding": padding, "bn": bn, "n_filters": n_filters, "n_fft": n_fft, "hop_length": hop_length,
         "stride": stride, "lr": lr, "style_weight": style_weights, "ac_weight": ac_weight, "ac_epochs": ac_epochs,
         "ac_bounds": ac_bounds, "freq_weight": freq_weight, "freq_epochs": freq_epochs, "noise_stats": noise_stats,
         "transform": transform_string, "synth_waveform": synth_waveform})
    
    s_file = "baselines/" + style_inst + ".wav"
    style, _ = librosa.load(s_file, sr)
    style_f = transform(torch.from_numpy(style))
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(style_f)
        plt.colorbar(fraction=0.047 * style_f.shape[0] / style_f.shape[1], pad=0.04)
        plt.title("Style Spectrogram")
        plt.show()
    
    if random_cnn:
        if net_type == "2D":
            model = RandomCNN2d(n_channels_in=style_f.shape[1], n_filters=n_filters, kernel_size=kernel_size,
                                n_layers=n_layers, bias=False, bn=bn, padding=padding, stride=stride,
                                activation=nn.ReLU(inplace=True))
        else:
            model = RandomCNN1d(n_fft_bins=style_f.shape[1], n_filters=n_filters, kernel_size=kernel_size,
                                n_layers=n_layers, bias=False, bn=bn, padding=padding, stride=stride,
                                activation=nn.ReLU(inplace=True))
        
        loss_layers = model.layer_dict.keys()
    
    else:
        model = resnet2d.ResNet(resnet2d.BasicBlock, [1, 1, 1, 1], num_classes=33, norm_layer=nn.InstanceNorm2d,
                                classify=False)
        exp = experiment.Experiment(model, "experiments/classifier_instancenorm2d", 40, None, None,
                                    continue_from_epoch=-2)
        model = exp.model
        loss_layers = ["0", "1", "2", "3", "4"]
    
    model.fix_params()
    
    if cuda:
        style_f = style_f.cuda()
        model = model.cuda()
        window = window.cuda()
    
    targets = model(style_f, loss_layers)
    
    if net_type == "2D":
        style_loss = [TextureLoss(target.detach(), weight=w) for target, w in zip(targets, style_weights)]
    else:
        style_loss = [StyleLoss(target.detach(), weight=w) for target, w in zip(targets, style_weights)]
    
    if ac_on == "spec":
        ac_loss = AutoLoss(style_f, weight=ac_weight, bounds=ac_bounds, on="spec")
    else:
        ac_loss = [AutoLoss(target.detach(), weight=ac_weight, bounds=ac_bounds, on="fmaps") for target in targets]
    
    ac_schedule = [ac_weight] * ac_epochs + [0] * (num_epochs - ac_epochs)
    
    freq_loss = FreqLoss(style_f, weight=freq_weight, dim=3 if net_type == "2D" else 2)
    freq_schedule = [freq_weight] * freq_epochs + [0] * (num_epochs - freq_epochs)
    
    if ac_on == "spec":
        all_losses = [style_loss]  # + others
    else:
        all_losses = [style_loss, ac_loss]  # + others
    
    if synth_waveform:
        noise = torch.randn(style.shape, device="cuda" if cuda else None, requires_grad=True)
    else:
        noise = create_noise(style_f, noise_stats)
    
    optimizer = opt([noise], lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    
    global loss_outer, loss_ac, loss_f, loss_g
    losses = []
    start = time()
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(1, num_epochs + 1):
            if ac_on == "fmaps":
                for l in ac_loss:
                    l.weight = ac_schedule[epoch - 1]
            else:
                ac_loss.weight = ac_schedule[epoch - 1]
            freq_loss.weight = freq_schedule[epoch - 1]
            
            def closure():
                optimizer.zero_grad()
                
                if synth_waveform:
                    spec = transform(noise)
                else:
                    spec = noise
                out = model(spec, loss_layers)
                
                layer_losses = [loss(x) for losses in all_losses for x, loss in zip(out, losses)]
                fl = freq_loss(spec)
                if ac_on == "spec":
                    acl = ac_loss(spec)
                    loss = sum(layer_losses) + acl + fl
                else:
                    loss = sum(layer_losses) + fl
                loss.backward()
                
                global loss_outer, loss_g, loss_ac, loss_f
                loss_outer = loss.item()
                loss_g = sum(layer_losses).item()
                if ac_on == "spec":
                    loss_ac = acl.item() if isinstance(acl, torch.Tensor) else acl
                loss_f = fl.item() if isinstance(fl, torch.Tensor) else fl
                return loss
            
            optimizer.step(closure)
            lr_scheduler.step(loss_outer)
            pbar.set_description("loss: {:.4f}".format(loss_outer))
            pbar.update()
            losses.append([loss_outer, loss_g, loss_ac, loss_f])
    
    if synth_waveform:
        out = noise.detach().cpu().numpy()
        gen_spectrum = stft_torch(noise, n_fft, hop_length, window).detach().cpu().squeeze()
    else:
        out = griffin_lim(noise.detach().cpu().numpy().squeeze(), hop_length)
        gen_spectrum = noise.detach().cpu().squeeze()
    # Add SNR and possibly other metrics
    contrast = webercontrast(gen_spectrum).item()
    SNR = snr(gen_spectrum).item()
    distance = (torch.norm(style_f.cpu().squeeze() - gen_spectrum) / torch.norm(style_f.cpu().squeeze())).item()
    elapsed = time() - start
    metrics = {"snr": SNR, "contrast": contrast, "distance": distance, "time_elapsed": elapsed}
    
    experiment.log(model, experiment_name, gen_spectrum, out, losses, notes, experiment_folder, metrics=metrics,
                   overwrite=overwrite, sr=sr, save_model=save_model, show_plot=plot, time=elapsed, **params_dict)
    return 0
