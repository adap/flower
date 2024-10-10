import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image


class RandomTimeResample(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input):
        data_length = input.size(-1)
        factor = torch.ones(1).uniform_(self.factor[0], self.factor[1]).item()
        new_data_length = round(data_length * factor)
        input = input.unsqueeze(0).unsqueeze(-1)
        input = torch.nn.functional.interpolate(input, [new_data_length, 1], mode='bilinear')
        input = input.squeeze(0).squeeze(-1)
        return input


class CenterCropPad(torch.nn.Module):
    def __init__(self, data_length):
        super().__init__()
        self.data_length = data_length

    def forward(self, input):
        data_length = input.size(-1)
        if data_length < self.data_length:
            pad = self.data_length - data_length
            input = torch.nn.functional.pad(input, [pad // 2, pad - (pad // 2)])
        elif data_length > self.data_length:
            input = input[..., data_length // 2 - self.data_length // 2: data_length // 2 + self.data_length // 2]
        return input


class RandomTimeShift(torch.nn.Module):
    def __init__(self, shift_time, sr=16000):
        super().__init__()
        self.shift_time = shift_time
        self.sr = sr
        shift = round(self.sr * self.shift_time)
        self.padding = (shift, shift)

    def forward(self, input):
        data_length = input.size(-1)
        input = F.pad(input, self.padding)
        left = torch.randint(0, self.padding[0], size=(1,))
        input = input[..., left: left + data_length]
        return input


class RandomBackgroundNoise(torch.nn.Module):
    def __init__(self, background_noise, freq=0.8, volume=0.1):
        super().__init__()
        self.background_noise = background_noise
        self.freq = freq
        self.volume = volume

    def forward(self, input):
        u = torch.rand((1,)).item()
        if u < self.freq:
            length = input.size(-1)
            volume = torch.ones(1).uniform_(0, self.volume).item()
            path_idx = torch.randint(0, len(self.background_noise), size=(1,)).item()
            background_noise = self.background_noise[path_idx]
            left = torch.randint(0, background_noise.size(-1) - length + 1, size=(1,))
            background_noise = background_noise[..., left: left + length]
            input = input + volume * background_noise
            input = torch.clamp(input, -1, 1)
        return input


class RandomPitchShift(torch.nn.Module):
    def __init__(self, shift_pitch):
        super().__init__()
        self.shift_pitch = shift_pitch
        self.padding = (0, 0, shift_pitch, shift_pitch)

    def forward(self, input):
        data_length = input.size(-2)
        input = F.pad(input, self.padding)
        left = torch.randint(0, self.shift_pitch, size=(1,))
        input = input[..., left: left + data_length, :]
        return input


class ComplextoPower(torch.nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, input):
        input = input.abs().pow(self.power)
        return input


class SpectoMFCC(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=40, dct_type=2, norm='ortho', log_mels=False, melkwargs=None):
        super().__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported: {}'.format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB('power', self.top_db)
        melkwargs = melkwargs or {}
        self.mel_scale = torchaudio.transforms.MelScale(sample_rate=self.sample_rate, **melkwargs)
        if self.n_mfcc > self.mel_scale.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        dct_mat = torchaudio.functional.create_dct(self.n_mfcc, self.mel_scale.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_mels = log_mels

    def forward(self, input):
        mel_specgram = self.mel_scale(input)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc


class SpectoImage(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        high = input.amax(dim=(-2, -1))
        low = input.amin(dim=(-2, -1))
        input.sub_(low).div_(max(high - low, 1e-5))
        input = input.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).numpy()
        input = Image.fromarray(input)
        return input
