# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvReLUNorm
from .text import cmudict
from .alignment import b_mas, mas_width1
from .attention import ConvAttention
from .transformer import FFTransformer


def regulate_len(
    durations, enc_out, pace: float = 1.0, mel_max_len: Optional[int] = None
):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (
        torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)
    ).float()
    pitch_nelems = (
        torch.gather(pitch_nonzero_cums, 2, dce)
        - torch.gather(pitch_nonzero_cums, 2, dcs)
    ).float()

    pitch_avg = torch.where(
        pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems
    )
    return pitch_avg


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(
        self, input_size, filter_size, kernel_size, dropout, n_layers=2, n_predictions=1
    ):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(
            *[
                ConvReLUNorm(
                    input_size if i == 0 else filter_size,
                    filter_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out


class Fast(nn.Module):
    def __init__(
        self,
        n_mel_channels=80,
        n_symbols=148,
        padding_idx=0,
        symbols_embedding_dim=384,
        in_fft_n_layers=6,
        in_fft_n_heads=1,
        in_fft_d_head=64,
        in_fft_conv1d_kernel_size=3,
        in_fft_conv1d_filter_size=1536,
        in_fft_output_size=384,
        p_in_fft_dropout=0.1,
        p_in_fft_dropatt=0.1,
        p_in_fft_dropemb=0,
        out_fft_n_layers=6,
        out_fft_n_heads=1,
        out_fft_d_head=64,
        out_fft_conv1d_kernel_size=3,
        out_fft_conv1d_filter_size=1536,
        out_fft_output_size=384,
        p_out_fft_dropout=0.1,
        p_out_fft_dropatt=0.1,
        p_out_fft_dropemb=0,
        dur_predictor_kernel_size=3,
        dur_predictor_filter_size=256,
        p_dur_predictor_dropout=0.1,
        dur_predictor_n_layers=2,
        pitch_predictor_kernel_size=3,
        pitch_predictor_filter_size=256,
        p_pitch_predictor_dropout=0.1,
        pitch_predictor_n_layers=2,
        pitch_embedding_kernel_size=3,
        energy_conditioning=False,
        energy_predictor_kernel_size=3,
        energy_predictor_filter_size=256,
        p_energy_predictor_dropout=0.1,
        energy_predictor_n_layers=2,
        energy_embedding_kernel_size=3,
        n_speakers=1,
        speaker_emb_weight=1.0,
        pitch_conditioning_formants=1,
    ):
        super(Fast, self).__init__()

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers,
            n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx,
        )

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout,
            n_layers=dur_predictor_n_layers,
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers,
            n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim,
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=pitch_predictor_filter_size,
            kernel_size=pitch_predictor_kernel_size,
            dropout=p_pitch_predictor_dropout,
            n_layers=pitch_predictor_n_layers,
            n_predictions=pitch_conditioning_formants,
        )

        self.pitch_emb = nn.Conv1d(
            pitch_conditioning_formants,
            symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2),
        )

        # Store values precomputed for training data within the model
        self.register_buffer("pitch_mean", torch.zeros(1))
        self.register_buffer("pitch_std", torch.zeros(1))

        self.energy_conditioning = energy_conditioning
        if energy_conditioning:
            self.energy_predictor = TemporalPredictor(
                in_fft_output_size,
                filter_size=energy_predictor_filter_size,
                kernel_size=energy_predictor_kernel_size,
                dropout=p_energy_predictor_dropout,
                n_layers=energy_predictor_n_layers,
                n_predictions=1,
            )

            self.energy_emb = nn.Conv1d(
                1,
                symbols_embedding_dim,
                kernel_size=energy_embedding_kernel_size,
                padding=int((energy_embedding_kernel_size - 1) / 2),
            )

        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)

        self.attention = ConvAttention(
            n_mel_channels,
            0,
            symbols_embedding_dim,
            use_query_proj=True,
            align_query_enc_type="3xconv",
        )

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas_width1(
                    attn_cpu[ind, 0, : out_lens[ind], : in_lens[ind]]
                )
                attn_out[ind, 0, : out_lens[ind], : in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device()
                )
        return attn_out

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(
                attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1
            )
        return torch.from_numpy(attn_out).to(attn.get_device())

    def infer(
        self,
        inputs,
        pace=1.0,
        dur_tgt=None,
        pitch_tgt=None,
        energy_tgt=None,
        pitch_transform=None,
        max_duration=75,
        speaker=0,
    ):

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Pitch over chars
        pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)

        if pitch_transform is not None:
            if self.pitch_std[0] == 0.0:
                # XXX LJSpeech-1.1 defaults
                mean, std = 218.14, 67.24
            else:
                mean, std = self.pitch_mean[0], self.pitch_std[0]
            pitch_pred = pitch_transform(
                pitch_pred, enc_mask.sum(dim=(1, 2)), mean, std
            )
        if pitch_tgt is None:
            pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
        else:
            pitch_emb = self.pitch_emb(pitch_tgt).transpose(1, 2)

        enc_out = enc_out + pitch_emb

        # Predict energy
        if self.energy_conditioning:

            if energy_tgt is None:
                energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)
                energy_emb = self.energy_emb(energy_pred.unsqueeze(1)).transpose(1, 2)
            else:
                energy_emb = self.energy_emb(energy_tgt).transpose(1, 2)

            enc_out = enc_out + energy_emb
        else:
            energy_pred = None

        len_regulated, dec_lens = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt, enc_out, pace, mel_max_len=None
        )

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        return mel_out, dec_lens, dur_pred, pitch_pred, energy_pred


def load_model(model_path, cmudict_path="default_cmudict", p_arpabet=1, gpu=False, energy_conditioning=False):
    if gpu:
        device = torch.device("cuda")
        model = Fast(energy_conditioning=energy_conditioning).cuda()
    else:
        device = torch.device("cpu")
        model = Fast(energy_conditioning=energy_conditioning)

    state_dict = torch.load(model_path, map_location=device)

    if "state_dict" in state_dict:
        sd = state_dict["state_dict"]
        if any(key.startswith("module.") for key in sd):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    if p_arpabet > 0.0:
        cmudict.initialize(cmudict_path, keep_ambiguous=True)

    return model
