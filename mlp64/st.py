import librosa
import scipy.fft as scipyfft

librosa.fft.set_fftlib(scipyfft)
import numpy as np
import torch
import torch.nn as nn
import torchaudio as ta
from scipy.signal import correlate2d
from typing import Union

cuda = True if torch.cuda.is_available() else False


# IDEA: use random multiscale network, with filtersizes representing pitch/octave

def stft_params(sr, frame_ms, hop_ms, signal_length):
    # Librosa stft uses signal_length/hop_length to determine number of frames, doesn't subtract frame_length first
    # Always truncate towards 0
    n_fft = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    num_frames = int(np.ceil(signal_length / hop_length))
    num_bins = int(1 + n_fft / 2)
    if num_bins * num_frames < signal_length:
        print("Reconstruction using n_fft: {}, hop_length: {} will be lossy,"
              " their product needs to be larger than the length of the signal."
              " Choose larger frames or shorter hops.").format(n_fft, hop_length)
    return n_fft, hop_length, num_bins, num_frames


def mstobin(ms, hop_length, sr=16000):
    ms_samples = ms / 1000 * sr
    return int(ms_samples / hop_length)


def stft(x, n_fft, hop_length):
    x = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    return np.log1p(np.abs(x))


def stft_torch(x: torch.Tensor, n_fft, hop_length, window, out_dims=4):
    x = ta.functional.complex_norm(torch.stft(x, n_fft, hop_length, window=window))
    if out_dims == 4:
        return torch.log1p(x)[None, None, :, :]
    else:
        return torch.log1p(x)[None, :, :]


def ri_spec(x: torch.Tensor, n_fft, hop_length, window):
    # Make sure to be in logarithmic scale
    x = torch.stft(x, n_fft, hop_length, window=window)
    x = x.permute(2, 0, 1)
    x = (2 * torch.sigmoid(x)) - 1
    return x[None, :]


def griffin_lim(x, hop_length, n_fft_bins=None, n_iter=500):
    # dim1 is the first dimension of the content in fourier space - should be equal in our case
    if n_fft_bins is None:
        n_fft_bins = x.shape[0]
    out_fft = np.zeros_like(x)
    out_fft[:n_fft_bins, :] = np.expm1(x)
    
    # This code does phase reconstruction
    p = 2 * np.pi * np.random.random_sample(out_fft.shape) - np.pi
    for i in range(n_iter):
        x = out_fft * np.exp(1j * p)
        x = librosa.istft(x, hop_length=hop_length)
    
    return x


def create_noise(style_f, noise_stats):
    
    if noise_stats:
        n_frames = style_f.shape[3]
        n_fft_bins = style_f.shape[2]
        
        mean_f = style_f.mean(3).cpu()
        std_f = style_f.std(3).cpu()
        mean_f = np.tile(mean_f, n_frames).reshape(n_frames, n_fft_bins)
        std_f = np.tile(std_f, n_frames).reshape(n_frames, n_fft_bins)
        
        mean_f = torch.from_numpy(mean_f[None, :, :]).transpose(1, 2)
        std_f = torch.from_numpy(std_f[None, :, :]).transpose(1, 2)
        
        mean_f = mean_f.cuda()
        std_f = std_f.cuda()
        
        noise = torch.zeros(style_f.shape, device="cuda" if cuda else None)
        noise = ((noise + mean_f)).requires_grad_()
    else:
        noise = torch.rand(style_f.shape, device="cuda" if cuda else None, requires_grad=True)
    
    return noise


class RandomCNN2d(nn.Module):
    
    def __init__(self, n_channels_in=1, n_filters=512, kernel_size=2, activation=nn.ReLU(), n_layers=1, bias=False,
                 stride=1,
                 padding=0, bn=False, stack=False):
        super(RandomCNN2d, self).__init__()
        
        if kernel_size == 2 and (n_layers == 1):
            kernel_size = [(11, 11)]
            padding = [(0, 0)]
        # If kernel_size is a tuple (1 layer) or list of tuples, all good
        if isinstance(kernel_size, tuple) and (n_layers == 1):
            kernel_size = [kernel_size]
            padding = [padding]
        elif isinstance(kernel_size, list) and isinstance(padding, list):
            if isinstance(kernel_size[0], int):
                kernel_size = [(x, x) for x in kernel_size]
            if isinstance(padding[0], int):
                padding = [(x, x) for x in padding]
            pass
        # Otherwise, for more layers, use powers of 2 for each additional layer
        elif n_layers > 1:
            kernel_size = [(2 ** i + 1, 2 ** i + 1) for i in range(1, n_layers + 1)]
            padding = [(int(2 ** i / 2), int(2 ** i / 2)) for i in range(1, n_layers + 1)]
        
        self.bias = bias
        self.n_layers = n_layers
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.bn = bn
        self.stack = stack
        self.layer_dict = nn.ModuleDict()
        self.bonus_dict = nn.ModuleDict()
        
        for i in range(n_layers):
            # Check strides etc.
            # Try BatchNorm
            if self.bn:
                self.bonus_dict["bn{}".format(i)] = nn.BatchNorm2d(
                    n_channels_in if (i == 0) or not stack else n_filters)
            
            self.layer_dict["conv{}".format(i)] = nn.Conv2d(n_channels_in, n_filters, kernel_size=kernel_size[i],
                                                            stride=self.stride,
                                                            padding=self.padding[i],
                                                            bias=self.bias)
            # weight = torch.rand(self.layer_dict["conv{}".format(i)].weight.data.shape) * 0.1 - 0.05
            # self.layer_dict["conv{}".format(i)].weight = torch.nn.Parameter(weight, requires_grad=False)
    
    def forward(self, x, out_keys):
        out = {}
        if self.bn:
            for layer, bonus in zip(self.layer_dict, self.bonus_dict):
                x = self.bonus_dict[bonus](x)
                out[layer] = self.activation(self.layer_dict[layer].forward(x))
                x = out[layer]
        else:
            for layer in self.layer_dict:
                out[layer] = self.activation(self.layer_dict[layer].forward(x))
                if self.stack:
                    x = out[layer]
        return [out[key] for key in out_keys]
    
    def fix_params(self):
        for conv in self.layer_dict.items():
            conv[1].weight.requires_grad = False


class RandomCNN1d(nn.Module):
    # Need to make requires_grad False outside using fix_params()
    def __init__(self, n_fft_bins, n_filters=512, kernel_size=2, activation=nn.ReLU(), n_layers=1, bias=False, stride=1,
                 padding=0, bn=False, stack=False):
        super(RandomCNN1d, self).__init__()
        
        # If kernel_size is a tuple (1 layer) or list of tuples, all good
        if isinstance(kernel_size, int) and (n_layers == 1):
            kernel_size = [kernel_size]
            padding = [padding]
        elif isinstance(kernel_size, list) and isinstance(padding, list):
            pass
        # Otherwise, for more layers, use powers of 2 for each additional layer
        elif n_layers > 1:
            kernel_size = [2 ** i + 1 for i in range(1, n_layers + 1)]
            padding = [int(2 ** i / 2) for i in range(1, n_layers + 1)]
        
        self.bias = bias
        self.n_layers = n_layers
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.bn = bn
        self.stack = stack
        self.layer_dict = nn.ModuleDict()
        self.bonus_dict = nn.ModuleDict()
        
        for i in range(n_layers):
            # Check strides etc.
            # Try BatchNorm
            if self.bn:
                self.bonus_dict["bn{}".format(i)] = nn.BatchNorm1d(n_fft_bins if (i == 0) or not stack else n_filters)
            
            self.layer_dict["conv{}".format(i)] = nn.Conv1d(n_fft_bins if (i == 0) or not stack else n_filters,
                                                            n_filters, kernel_size=kernel_size[i], stride=self.stride,
                                                            padding=self.padding[i],
                                                            bias=self.bias)
    
    def forward(self, x, out_keys):
        out = {}
        if self.bn:
            for layer, bonus in zip(self.layer_dict, self.bonus_dict):
                x = self.bonus_dict[bonus](x)
                out[layer] = self.activation(self.layer_dict[layer].forward(x))
                x = out[layer]
        else:
            for layer in self.layer_dict:
                out[layer] = self.activation(self.layer_dict[layer].forward(x))
                if self.stack:
                    x = out[layer]
        return [out[key] for key in out_keys]
    
    def fix_params(self):
        for conv in self.layer_dict.items():
            conv[1].weight.requires_grad = False


class GramMatrix(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.bmm(x, x.transpose(1, 2)) / x.shape[2]


class TimeGram(nn.Module):
    def forward(self, x: torch.Tensor):
        x = x.transpose(2, 1)
        return torch.matmul(x, x.transpose(2, 3)).transpose(1, 3)


class StyleLoss(nn.Module):
    # Changed to true Frobenius norm (adds sqrt)
    def __init__(self, target, weight=1, criterion=nn.MSELoss):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.gram = GramMatrix()
        self.target = self.gram(target)
        self.criterion = criterion()
    
    def forward(self, x):
        if self.weight == 0:
            return 0.
        G = self.gram(x)
        return self.weight * torch.norm(self.target - G) / torch.norm(self.target)


class TextureLoss(nn.Module):
    def __init__(self, target, weight):
        super(TextureLoss, self).__init__()
        self.weight = weight
        self.gram = TimeGram()
        self.target = self.gram(target)
    
    def forward(self, x):
        if self.weight == 0:
            return 0.
        return self.weight * torch.norm(self.target - self.gram(x)) / torch.norm(self.target)


class AutoLoss(nn.Module):
    # Currently summing over 2D channels
    def __init__(self, target, weight=1, bounds=(0, np.inf), on="fmaps"):
        super(AutoLoss, self).__init__()
        self.bounds = [*bounds]
        self.bounds[1] = min(self.bounds[1], target.shape[-1])
        self.weight = weight
        self.on = on
        if on == "fmaps":
            self.target = autocorr(target.sum(1))[..., self.bounds[0]:self.bounds[1]]
        else:
            self.target = autocorr(target)[..., self.bounds[0]:self.bounds[1]]
    
    def forward(self, x):
        if self.weight == 0:
            return 0.
        if self.on == "fmaps":
            A = autocorr(x.sum(1))[..., self.bounds[0]:self.bounds[1]]
        else:
            A = autocorr(x)[..., self.bounds[0]:self.bounds[1]]
        return self.weight * torch.norm(A - self.target) / torch.norm(self.target)


def autocorr(x):
    halfn = x.shape[-1] / 2
    if np.mod(x.shape[-1], 2) == 0:
        pad = [int(halfn), int(halfn) - 1]
    else:
        pad = [int(halfn), int(halfn)]
    x = torch.rfft(torch.nn.functional.pad(x, pad), 1, onesided=True)
    # For 1D audio
    # x = torch.rfft(x, 1, onesided=True)
    # Complex # times its conjugate == a**2+b**2, which is the magnitude squared
    # Need to stack with zeros as irfft requires complex input, while the complex norm returns a real number
    x = torch.irfft(torch.stack((ta.functional.complex_norm(x, 2), torch.zeros(x.shape[:-1]).cuda()), -1), 1,
                    onesided=True)
    return x[..., :int(halfn * 2)]


class DivLoss(nn.Module):
    def __init__(self, target, weight, step=50):
        super(DivLoss, self).__init__()
        self.weight = weight
        self.target = target
        # self.criterion = nn.MSELoss()
        self.step = step
    
    def forward(self, x):
        numerator = torch.sum(self.target ** 2)
        l = []
        for i in range(self.step, x.shape[-1] - self.step, self.step):
            a = numerator / torch.sum((self.target[..., i - self.step:i] - x[..., i:i + self.step]) ** 2)
            if a != float("inf"):
                l.append(a)
        return self.weight * torch.max(torch.stack(l))


class FreqLoss(nn.Module):
    def __init__(self, target, weight, dim=3, business=1):
        super(FreqLoss, self).__init__()
        self.weight = weight
        self.dim = dim
        self.target = target.sum(dim) * business
    
    def forward(self, x):
        if self.weight == 0:
            return 0.
        return self.weight * torch.norm(x.sum(self.dim) - self.target) / torch.norm(self.target)


class SNRLoss(nn.Module):
    # Don't use this. Doesn't work atm
    def __init__(self, target, weight):
        super(SNRLoss, self).__init__()
        self.weight = weight
        # target = norm01(target)
        self.target_contrast = (target.max() - target.mean()) / target.mean()
    
    def forward(self, x):
        if self.weight == 0:
            return 0
        # x = norm01(x)
        # lq = quantile(x, 0.5)
        # uq = quantile(x, 0.99)
        lq = x.mean()
        uq = x.max()
        contrast = (uq - lq) / (lq + 1e-6)
        # sd = (self.target_contrast - contrast) ** 2 / self.target_contrast
        # print("C: {:5f}, L: {:5f}, l: {:5f}, u: {:5f}".format(
        #     contrast.item(), torch.exp(-contrast * 1/self.weight).item(), lq.item(), uq.item()))
        return torch.exp(-contrast * 1 / self.weight)


def norm01(x):
    return (x - x.min()) / handle_zeros(x.max() - x.min())


def handle_zeros(x):
    if x == .0:
        return 1.
    else:
        return x


def quantile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values
    return result


def webercontrast(x):
    return (x.max() - x.median()) / x.median()


def snr(x):
    if x.median().item() == 0.:
        return torch.Tensor([9999])
    x = torch.expm1(x)
    return 10 * torch.log10(x.max() / x.median())


def angle(x):
    a, b = x[..., 0], x[..., 1]
    theta = torch.atan(b / a)
    theta[a < 0] += np.pi
    theta = torch.fmod(theta, 2 * np.pi)
    return theta


def phase(x):
    a, b = x[..., 0], x[..., 1]
    theta = torch.atan(b / a)
    theta[a < 0] += np.pi
    return theta


def diff(x, axis):
    """Take the finite difference of a tensor along an axis.
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.shape
    
    begin_back = [0 for unused_s in range(len(shape))]
    #     print("begin_back",begin_back)
    begin_front = [0 for unused_s in range(len(shape))]
    
    begin_front[axis] = 1
    #     print("begin_front",begin_front)
    
    size = list(shape)
    size[axis] -= 1
    #     print("size",size)
    slice_front = x[begin_front[0]:begin_front[0] + size[0], begin_front[1]:begin_front[1] + size[1]]
    slice_back = x[begin_back[0]:begin_back[0] + size[0], begin_back[1]:begin_back[1] + size[1]]
    
    #     slice_front = tf.slice(x, begin_front, size)
    #     slice_back = tf.slice(x, begin_back, size)
    #     print("slice_front",slice_front)
    #     print(slice_front.shape)
    #     print("slice_back",slice_back)
    
    d = slice_front - slice_back
    return d


def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    ddmod = torch.fmod(dd + np.pi, 2.0 * np.pi) - np.pi
    
    idx = np.logical_and(np.equal(ddmod, -np.pi), np.greater(dd, 0))
    
    ddmod = torch.where(idx, torch.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    
    idx = np.less(np.abs(dd), discont)
    
    ddmod = torch.where(idx, torch.zeros_like(ddmod), dd)
    ph_cumsum = torch.cumsum(ph_correct, dim=axis)
    
    shape = p.shape
    shape[axis] = 1
    ph_cumsum = torch.cat([torch.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    return p + ph_cumsum


def autocorr_loss(F_synth, F_target, bounds=[15, 195]):
    # Bounds determine the lag of the autocorrelation we consider
    # The last dimension is time - and every spectrogram bin corresponds with a frame-size
    # The Antognini paper discards AC outside of (200ms, 2s)
    bounds[1] = min(bounds[1], F_synth.shape[-1])
    ac_synth = autocorr(F_synth)[..., bounds[0]:bounds[1]]
    ac_target = autocorr(F_target)[..., bounds[0]:bounds[1]]
    return torch.sum((ac_synth - ac_target) ** 2) / (torch.sum(ac_target) ** 2)


def div_loss(F_synth, F_target, step=50):
    # step size is 50 frames
    # TODO: try multiplying this with autocorrelation to get some kind of shift-invariant diversity loss
    numerator = torch.sum(F_target ** 2)
    l = []
    for i in range(step, F_synth.shape[-1] - step, step):
        l.append(numerator / torch.sum((F_synth[..., i - step:i] - F_target[..., i:i + step]) ** 2))
    return max(l)
