import sys
import torch
from models.stft import STFT


class Denoiser(torch.nn.Module):
    def __init__(
        self, model, filter_length=1024, n_overlap=4, win_length=1024, mode="normal"
    ):
        super(Denoiser, self).__init__()
        self.stft = STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length,
        )
        if mode == "zeros":
            mel_input = torch.zeros((1, 80, 88))
        elif mode == "normal":
            mel_input = torch.randn((1, 80, 88))
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            ort_inputs = {model.get_inputs()[0].name: mel_input.float().cpu().numpy()}
            ort_outs = hifigan.run(None, ort_inputs)
            bias_audio = torch.FloatTensor(ort_outs[0]).view(1, -1)
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
