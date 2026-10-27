# Vendored from https://github.com/sony/hFT-Transformer (MIT, Copyright (c) 2023 Sony Group Corporation),
# file model/model_spec2midi.py at the commit pinned in ras/models/__init__.py. Shape comments and
# debug prints are removed; the layers are unchanged so the released weights load as-is.
# Paper: Toyama et al., "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer"
# (ISMIR 2023).

import torch
import torch.nn as nn


class Model_SPEC2MIDI(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder_spec2midi = encoder
        self.decoder_spec2midi = decoder

    def forward(self, input_spec):
        return self.decoder_spec2midi(self.encoder_spec2midi(input_spec))


class Encoder_SPEC2MIDI(nn.Module):
    def __init__(self, n_margin, n_frame, n_bin, cnn_channel, cnn_kernel, hid_dim, n_layers, n_heads, pf_dim,
                 dropout, device):
        super().__init__()
        self.device = device
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.cnn_channel = cnn_channel
        self.cnn_kernel = cnn_kernel
        self.hid_dim = hid_dim
        self.conv = nn.Conv2d(1, self.cnn_channel, kernel_size=(1, self.cnn_kernel))
        self.n_proc = n_margin * 2 + 1
        self.cnn_dim = self.cnn_channel * (self.n_proc - (self.cnn_kernel - 1))
        self.tok_embedding_freq = nn.Linear(self.cnn_dim, hid_dim)
        self.pos_embedding_freq = nn.Embedding(n_bin, hid_dim)
        self.layers_freq = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                          for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale_freq = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, spec_in):
        batch_size = spec_in.shape[0]
        spec = spec_in.unfold(2, self.n_proc, 1).permute(0, 2, 1, 3).contiguous()
        spec_cnn = spec.reshape(batch_size * self.n_frame, self.n_bin, self.n_proc).unsqueeze(1)
        spec_cnn = self.conv(spec_cnn).permute(0, 2, 1, 3).contiguous()
        spec_cnn_freq = spec_cnn.reshape(batch_size * self.n_frame, self.n_bin, self.cnn_dim)
        spec_emb_freq = self.tok_embedding_freq(spec_cnn_freq)
        pos_freq = torch.arange(0, self.n_bin).unsqueeze(0).repeat(batch_size * self.n_frame, 1).to(spec_in.device)
        spec_freq = self.dropout((spec_emb_freq * self.scale_freq.to(spec_in.device))
                                 + self.pos_embedding_freq(pos_freq))
        for layer_freq in self.layers_freq:
            spec_freq = layer_freq(spec_freq)
        return spec_freq.reshape(batch_size, self.n_frame, self.n_bin, self.hid_dim)


class Decoder_SPEC2MIDI(nn.Module):
    def __init__(self, n_frame, n_bin, n_note, n_velocity, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.n_note = n_note
        self.n_frame = n_frame
        self.n_velocity = n_velocity
        self.n_bin = n_bin
        self.hid_dim = hid_dim
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding_freq = nn.Embedding(n_note, hid_dim)
        self.layer_zero_freq = DecoderLayer_Zero(hid_dim, n_heads, pf_dim, dropout, device)
        self.layers_freq = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                          for _ in range(n_layers - 1)])
        self.fc_onset_freq = nn.Linear(hid_dim, 1)
        self.fc_offset_freq = nn.Linear(hid_dim, 1)
        self.fc_mpe_freq = nn.Linear(hid_dim, 1)
        self.fc_velocity_freq = nn.Linear(hid_dim, self.n_velocity)
        self.scale_time = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.pos_embedding_time = nn.Embedding(n_frame, hid_dim)
        self.layers_time = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                          for _ in range(n_layers)])
        self.fc_onset_time = nn.Linear(hid_dim, 1)
        self.fc_offset_time = nn.Linear(hid_dim, 1)
        self.fc_mpe_time = nn.Linear(hid_dim, 1)
        self.fc_velocity_time = nn.Linear(hid_dim, self.n_velocity)

    def forward(self, enc_spec):
        b, dev = enc_spec.shape[0], enc_spec.device
        enc_spec = enc_spec.reshape([b * self.n_frame, self.n_bin, self.hid_dim])
        pos_freq = torch.arange(0, self.n_note).unsqueeze(0).repeat(b * self.n_frame, 1).to(dev)
        midi_freq, attention_freq = self.layer_zero_freq(enc_spec, self.pos_embedding_freq(pos_freq))
        for layer_freq in self.layers_freq:
            midi_freq, attention_freq = layer_freq(enc_spec, midi_freq)
        dim = attention_freq.shape
        attention_freq = attention_freq.reshape([b, self.n_frame, dim[1], dim[2], dim[3]])
        shape = [b, self.n_frame, self.n_note]
        onset_freq = self.sigmoid(self.fc_onset_freq(midi_freq).reshape(shape))
        offset_freq = self.sigmoid(self.fc_offset_freq(midi_freq).reshape(shape))
        mpe_freq = self.sigmoid(self.fc_mpe_freq(midi_freq).reshape(shape))
        velocity_freq = self.fc_velocity_freq(midi_freq).reshape(shape + [self.n_velocity])

        midi_time = (midi_freq.reshape([b, self.n_frame, self.n_note, self.hid_dim]).permute(0, 2, 1, 3)
                     .contiguous().reshape([b * self.n_note, self.n_frame, self.hid_dim]))
        pos_time = torch.arange(0, self.n_frame).unsqueeze(0).repeat(b * self.n_note, 1).to(dev)
        midi_time = self.dropout((midi_time * self.scale_time.to(dev)) + self.pos_embedding_time(pos_time))
        for layer_time in self.layers_time:
            midi_time = layer_time(midi_time)
        tshape = [b, self.n_note, self.n_frame]
        onset_time = self.sigmoid(self.fc_onset_time(midi_time).reshape(tshape).permute(0, 2, 1).contiguous())
        offset_time = self.sigmoid(self.fc_offset_time(midi_time).reshape(tshape).permute(0, 2, 1).contiguous())
        mpe_time = self.sigmoid(self.fc_mpe_time(midi_time).reshape(tshape).permute(0, 2, 1).contiguous())
        velocity_time = (self.fc_velocity_time(midi_time).reshape(tshape + [self.n_velocity])
                         .permute(0, 2, 1, 3).contiguous())
        return (onset_freq, offset_freq, mpe_freq, velocity_freq, attention_freq,
                onset_time, offset_time, mpe_time, velocity_time)


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        _src, _ = self.self_attention(src, src, src)
        src = self.layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        return self.layer_norm(src + self.dropout(_src))


class DecoderLayer_Zero(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_src, trg):
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        return self.layer_norm(trg + self.dropout(_trg)), attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_src, trg):
        _trg, _ = self.self_attention(trg, trg, trg)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        return self.layer_norm(trg + self.dropout(_trg)), attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value):
        b = query.shape[0]
        q = self.fc_q(query).view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.fc_k(key).view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.fc_v(value).view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), v).permute(0, 2, 1, 3).contiguous()
        return self.fc_o(x.view(b, -1, self.hid_dim)), attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.dropout(torch.relu(self.fc_1(x))))


def build(params: dict, device: str = "cpu") -> Model_SPEC2MIDI:
    """The released MAESTRO model, from its parameter.json and corpus config."""
    t, c = params["transformer"], params["cnn"]
    enc = Encoder_SPEC2MIDI(32, 128, 256, c["channel"], c["kernel"], t["hid_dim"], t["encoder"]["n_layer"],
                            t["encoder"]["n_head"], t["pf_dim"], 0.0, device)
    dec = Decoder_SPEC2MIDI(128, 256, 88, 128, t["hid_dim"], t["decoder"]["n_layer"], t["decoder"]["n_head"],
                            t["pf_dim"], 0.0, device)
    return Model_SPEC2MIDI(enc, dec)
