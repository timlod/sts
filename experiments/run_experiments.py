## Automatically runs experiments (using different sound textures and hyperparameters)

import itertools
from pathlib import Path

from experiments import baseline_train
from mlp64.st import *

# Static variables
librosa.fft.set_fftlib(scipyfft)
cuda = True if torch.cuda.is_available() else False
sr = 16000

experiment_folder = Path("experiments_init2")
experiment_folder.mkdir(exist_ok=True)
# Experiment variables
style_inst = ["windchimes"]
random_cnn = [True]
net_type = "2D"
# Needs to be False for non-pretrained
decay_weights = [False]
autocorrelation_weights = [0]
ac_bounds = [(200, 2000)]
frequency_weights = [1, 0]
n_fft = [512]
hop_length = [128]
n_layers = [6]
n_filters = [128]
# kernel_size = [[(int(k / 2), 2), (3, 11), (3, 23), (5, 5), (11, 11), (19, 19), (27, 27)] for k in n_fft]
kernel_size = [[(128, 2), (51, 3), (5, 5), (11, 11), (21, 21), (3, 31), (27, 27)]]  # for k in n_fft]
# kernel_size = [2]
# padding = [[(int(x[0]/2), int(x[1]/2)) for x in kernel_size[0]]]
padding = [[(0, 0)] * 8]
bn = [False]
stride = [1]
synth_waveform = [False, True]
ac_on = ["fmaps"]
noise_stats = [False, True]

transform = ["STFT"]
opt = torch.optim.LBFGS

lr = [0.4]
num_epochs = [40]

all_params = list(itertools.product(style_inst, random_cnn, decay_weights,
                                    autocorrelation_weights, frequency_weights, n_fft,
                                    hop_length, n_layers, n_filters, kernel_size,
                                    padding, bn, stride, lr, num_epochs, ac_bounds, synth_waveform, ac_on, noise_stats,
                                    transform))
# print(all_params)
i = 0
print("No. experiments:", len(all_params))
for params in all_params:
    print(params)
    cur_style_inst, cur_random, cur_decay, cur_auto, cur_freq, cur_n_fft, cur_hop_length, cur_n_layers, cur_n_filters, \
    cur_kernel_size, cur_padding, cur_bn, cur_stride, cur_lr, cur_num_epochs, cur_acbounds, cur_synth, cur_ac_on, \
    cur_noise_stats, cur_transform = params
    
    exp_name = cur_style_inst
    exp_name += "_rnd" if cur_random else "resnet"
    exp_name += ("_" + cur_transform + "_" + net_type) if cur_random else ""
    exp_name += "_decay" if cur_decay else ""
    exp_name += "_wav" if cur_synth else "_spec"
    exp_name += "_ac" + str(cur_auto) + (
        "-" + cur_ac_on + "-" + "-".join([str(x) for x in cur_acbounds]) if (cur_auto != 0) else "")
    exp_name += "_freq" + str(cur_freq)
    exp_name += "_nfft" + str(cur_n_fft)
    exp_name += "_hop" + str(cur_hop_length)
    exp_name += "_nl" + str(cur_n_layers)
    exp_name += "_nf" + str(cur_n_filters)
    exp_name += "_ep" + str(cur_num_epochs)
    exp_name += "_ns" if cur_noise_stats else ""
    exp_name = exp_name.replace('.', 'p')
    
    print("Running experiment no. {}, {}".format(i, exp_name))
    baseline_train.run_baseline(style_inst=cur_style_inst, random_cnn=cur_random, transform=cur_transform,
                                decay=cur_decay, ac_weight=cur_auto, net_type=net_type, opt=opt,
                                freq_weight=cur_freq, n_fft=cur_n_fft, ac_on=cur_ac_on,
                                hop_length=cur_hop_length, n_layers=cur_n_layers,
                                n_filters=cur_n_filters, kernel_size=cur_kernel_size,
                                padding=cur_padding, bn=cur_bn, stride=cur_stride, noise_stats=cur_noise_stats,
                                lr=cur_lr, num_epochs=cur_num_epochs, synth_waveform=cur_synth,
                                experiment_name=exp_name, experiment_folder=experiment_folder)
    i += 1
