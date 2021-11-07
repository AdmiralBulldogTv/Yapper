import io

import onnxruntime
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence

from models.fast.model import load_model as load_fast_model
from models.fast.text import cmudict
from models.fast.text.text_processing import TextProcessing

from models.tacotron2.params import SAMPLING_RATE

from buffer import buffer
from scipy.io.wavfile import write

DEFAULT_PACE = 1
DEFAULT_PITCH_SHIFT = 0


class FastHifiTTS:
    def __init__(self):
        self.warmup_done = False
        self.ready = False
        self.speaker = ""

    def update_model(
        self,
        fast_path: str,
        onnx_path: str,
        cmudict_path: str,
        speaker: str,
        warm_up: bool = False,
        gpu: bool = False,
        pace: float = DEFAULT_PACE,
        pitch_shift: int = DEFAULT_PITCH_SHIFT,
        p_arpabet: float = 1,
        period: bool = False,
        start: int = 0,
        volume: int = 1,
        energy_conditioning: bool = False
    ):
        self.ready = False

        self.fast = load_fast_model(
            fast_path, gpu=gpu, p_arpabet=p_arpabet, cmudict_path=cmudict_path, energy_conditioning=energy_conditioning
        )
        self.onnx = onnxruntime.InferenceSession(onnx_path)
        self.gpu = gpu
        self.device = torch.device("cuda" if self.gpu else "cpu")
        self.speaker = speaker
        self.period = period
        self.pace = pace
        self.pitch_shift = pitch_shift
        self.volume = volume

        # TODO: allow for different types of pace and pitch transformations at inference time
        self.default_pitch_function = create_shift_pitch(self.pitch_shift)
        self.gen_kw = {
            "pace": self.pace,
            "speaker": self.speaker,
            "pitch_tgt": None,
            "pitch_transform": self.default_pitch_function,
        }

        if p_arpabet > 0.0:
            cmudict.initialize(cmudict_path, keep_ambiguous=True)

        self.tp = TextProcessing(
            "english_basic",
            ["english_cleaners"],
            p_arpabet=p_arpabet,
            handle_arpabet_ambiguous="first",
        )
        self.start = [self.tp.encode_text("A A")[1]] * start
        if warm_up:
            self.warm_up()

        self.ready = True

    def warm_up(self):
        if self.warmup_done:
            return

        with buffer() as out:
            self.generate("I am yapping so hard right now", out)

    def generate(self, text: str, out: io.IOBase) -> float:
        if not self.ready:
            raise Exception("Not ready!")

        self.warmup_done = True

        if self.period:
            text = text + "."

        text = torch.LongTensor(self.start + self.tp.encode_text(text))
        self.tp.sequence_to_text(text.numpy())
        sequence = pad_sequence([text], batch_first=True)

        with torch.no_grad():
            mel_output, *_ = self.fast.infer(sequence, **self.gen_kw)

            gen_output = self.onnx.run(None, {"input": to_numpy(mel_output)})
            audio = np.squeeze(gen_output[0], 0)[0] * self.volume
            write(out, SAMPLING_RATE, audio)
        return audio.shape[0] / SAMPLING_RATE

    def get_speaker(self) -> str:
        if not self.ready:
            raise Exception("Not ready!")

        return self.speaker


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def create_shift_pitch(shift):
    def shift_pitch(pitch, pitch_lens, mean, std):
        return pitch + shift / std

    return shift_pitch
