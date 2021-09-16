from .text import symbols

SAMPLING_RATE = 22050


def create_params():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    """Create model hyperparameters. Parse nondefault from given string."""

    params = tf.contrib.training.HParams(
        fp16_run=False,
        max_wav_value=32768.0,
        sampling_rate=SAMPLING_RATE,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        n_frames_per_step=1,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.1,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        attention_rnn_dim=1024,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        batch_size=1,
        mask_padding=True,  # set model's padded outputs to padded values
    )

    return params
