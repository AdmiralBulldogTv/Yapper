import onnxruntime
import torch
import numpy as np
import io
import re

from models.tacotron2.model import load_model as load_taco_model
from models.tacotron2.text import text_to_sequence
from models.tacotron2.params import SAMPLING_RATE

from scipy.io.wavfile import write
from buffer import buffer

DEFAULT_GATE_THRESHOLD = 0.1


class TaHifiTTS:
    def __init__(self):
        self.warmup_done = False
        self.ready = False
        self.speaker = ""

    def warm_up(self):
        if self.warmup_done:
            return

        with buffer() as out:
            self.generate("I am yapping so hard right now", out)

    def generate(self, text: str, out: io.IOBase) -> float:
        if not self.ready:
            raise Exception("Not ready!")

        text = clean_text(text)

        self.warmup_done = True
        if self.period:
            text = text + "."
        sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]

        if self.gpu:
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        else:
            sequence = torch.autograd.Variable(torch.from_numpy(sequence))

        _, mel_output, _, _ = self.taco.inference(sequence)

        with torch.no_grad():
            if self.gpu:
                mel_output = mel_output.float()
            else:
                mel_output = mel_output.float().data.cpu()
            gen_output = self.onnx.run(
                None, {"input": mel_output.detach().cpu().numpy()}
            )
            audio = np.squeeze(gen_output[0], 0)[0]
            write(out, SAMPLING_RATE, audio)
        return audio.shape[0] / SAMPLING_RATE

    def update_model(
        self,
        taco_path: str,
        onnx_path: str,
        speaker: str,
        warm_up: bool = False,
        gpu: bool = False,
        gate_threshold: float = DEFAULT_GATE_THRESHOLD,
        period: bool = False,
    ):
        self.ready = False

        self.taco = load_taco_model(taco_path, gpu=gpu, gate_threshold=gate_threshold)
        self.onnx = onnxruntime.InferenceSession(onnx_path)
        self.gpu = gpu
        self.speaker = speaker
        self.gate_threshold = gate_threshold
        self.period = period

        if warm_up:
            self.warm_up()

        self.ready = True

    def get_speaker(self) -> str:
        if not self.ready:
            raise Exception("Not ready!")

        return self.speaker


pattern = re.compile("[^a-z |-]")


def clean_text(text: str) -> str:
    return pattern.sub("", text.strip().lower())
