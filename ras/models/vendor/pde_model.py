# Vendored from https://github.com/zhanh-he/piano-dynamic-estimation (MIT, Copyright (c) 2025 zhanh-he),
# files pytorch/feature_extractor.py and pytorch/models_audioCNN.py at the commit pinned in
# ras/models/__init__.py. Only the parts inference needs are kept; plotting/CLI code and the unused
# single-task and log-mel paths are removed. The multi-scale CNN encoder is adapted by the upstream
# authors from GuzhengTech99 (https://github.com/LiDCC/GuzhengTech99).
# Paper: "Joint Estimation of Piano Dynamics and Metrical Structure with a Multi-task Multi-Scale
# Network" (ICASSP 2026).
# ruff: noqa: E501, E702

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DYNAMIC_LABELS_5 = ["blank", "pp", "p", "mf", "f", "ff"]


class PsychoFeatureExtractor(nn.Module):
    """Bark-scale specific loudness in sones (Zwicker/Terhardt/Schroeder, after Pampalk's ma_sone)."""

    def __init__(self, sample_rate=22050, fft_size=1024, frames_per_second=50, db_max=96.0):
        super().__init__()
        self.db_max = db_max
        self.fft_size = fft_size
        self.hop_size = int(round(sample_rate / frames_per_second))
        self.window = torch.hann_window(fft_size)
        self.fft_freq = torch.linspace(0, sample_rate / 2, fft_size // 2 + 1)
        bark_upper = torch.tensor([100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
                                   2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])
        self.bark_upper = bark_upper[bark_upper <= sample_rate / 2]
        self.bark_bands = len(self.bark_upper)

    def _outer_ear_weighting(self, power):
        w_adb = torch.zeros_like(self.fft_freq)
        f_khz = self.fft_freq[1:] / 1000
        w_adb[1:] = -3.64 * f_khz ** -0.8 + 6.5 * torch.exp(-0.6 * (f_khz - 3.3) ** 2) - 0.001 * f_khz ** 4
        w = (10 ** (w_adb / 20)) ** 2
        return power * w.to(power.device).view(1, -1, 1)

    def _bark_scaling(self, w_power, device):
        bands, k = [], 0
        for i in range(self.bark_bands):
            idx = torch.arange(k, k + (self.fft_freq[k:] <= self.bark_upper[i]).sum().item(), device=device)
            bands.append(w_power[:, idx, :].sum(dim=1))
            k = idx[-1] + 1
        return torch.stack(bands, dim=1)

    def _schroeder_spreading(self, bark):
        n = self.bark_bands
        b = torch.arange(1, n + 1, device=bark.device).unsqueeze(1) - torch.arange(1, n + 1, device=bark.device).unsqueeze(0) + 0.474
        spread = 10 ** ((15.81 + 7.5 * b - 17.5 * torch.sqrt(1 + b ** 2)) / 10)
        return torch.matmul(spread, bark)

    def forward(self, wav: torch.Tensor):
        wav = wav * (10 ** (self.db_max / 20))
        spec = torch.stft(wav, n_fft=self.fft_size, hop_length=self.hop_size,
                          window=self.window.to(wav.device), return_complex=True)
        power = spec.abs() ** 2 / self.window.sum() ** 2
        bark = self._bark_scaling(self._outer_ear_weighting(power), wav.device)
        bark_db = 10 * torch.log10(torch.clamp(self._schroeder_spreading(bark), min=1.0))
        return torch.where(bark_db >= 40, 2 ** ((bark_db - 40) / 10), (bark_db / 40) ** 2.642)


class block(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, (3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, (3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(out)
        self.sk = nn.Conv2d(inp, out, (1, 1), padding=(0, 0))

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.sk(x)
        return out


class self_att(nn.Module):
    def __init__(self, embeded_dim, num_heads, out):
        super().__init__()
        self.att = nn.MultiheadAttention(embeded_dim, num_heads, batch_first=True)

    def forward(self, x):
        x1 = x.squeeze(-1).transpose(-1, -2)
        res_branch, _ = self.att(x1, x1, x1)
        return torch.add(res_branch.transpose(-1, -2).unsqueeze(-1), x)


class SYMultiScaleAttnFlex(nn.Module):
    """Multi-scale attention CNN with two temporal strides (s1, s2)."""

    def __init__(self, FRE, latent_dim, temporal_strides=None):
        super().__init__()
        inp, size, fs, ps = FRE, 1, (3, 1), (1, 0)
        if temporal_strides is None:
            s1, s2 = 3, 3
        elif isinstance(temporal_strides, Sequence) and not isinstance(temporal_strides, (str, bytes)):
            s1 = int(temporal_strides[0]) if len(temporal_strides) >= 1 else 1
            s2 = int(temporal_strides[1]) if len(temporal_strides) >= 2 else 1
        else:
            s1 = s2 = int(temporal_strides)
        s1, s2 = max(1, s1), max(1, s2)
        self.s1, self.s2 = s1, s2

        self.bn0 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, FRE, (1, size), padding=(0, 0))
        self.maxpool012 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.conv02 = nn.Conv2d(inp, FRE, (1, size), padding=(0, 0))
        self.block11 = block(FRE, FRE * 2)
        self.block12 = block(FRE, FRE * 2)
        self.maxpool112 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.dropout12 = nn.Dropout(p=0.2)
        self.maxpool123 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.dropout123 = nn.Dropout(p=0.2)
        self.us121 = nn.ConvTranspose2d(FRE * 2, FRE * 2, kernel_size=(s1, 1), stride=(s1, 1))
        self.conv21 = nn.Conv2d(FRE * 2, FRE * 2, (1, 2))
        self.conv22 = nn.Conv2d(FRE * 2, FRE * 2, (1, 2))
        self.conv23 = nn.Conv2d(FRE * 2, FRE * 2, (1, 1))
        self.block21 = block(FRE * 2, FRE * 3)
        self.block22 = block(FRE * 2, FRE * 3)
        self.block23 = block(FRE * 2, FRE * 3)
        self.self_att23 = self_att(FRE * 3, 1, 1)
        self.bn23 = nn.BatchNorm2d(FRE * 3)
        self.maxpool212 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.maxpool223 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.dropout22 = nn.Dropout(p=0.2)
        self.dropout23 = nn.Dropout(p=0.2)
        self.us221 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s2, 1), stride=(s2, 1))
        self.us232 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s2, 1), stride=(s2, 1))
        self.conv31 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))
        self.conv32 = nn.Conv2d(FRE * 3, FRE * 3, (1, 3))
        self.conv33 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))
        self.block31 = block(FRE * 3, FRE * 3)
        self.block32 = block(FRE * 3, FRE * 3)
        self.block33 = block(FRE * 3, FRE * 3)
        self.bn31 = nn.BatchNorm2d(FRE * 3); self.relu31 = nn.ReLU(inplace=True)
        self.bn32 = nn.BatchNorm2d(FRE * 3); self.relu32 = nn.ReLU(inplace=True)
        self.bn33 = nn.BatchNorm2d(FRE * 3); self.relu33 = nn.ReLU(inplace=True)
        self.maxpool312 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.dropout312 = nn.Dropout(p=0.2)
        self.us321 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s1, 1), stride=(s1, 1))
        self.us332 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s2, 1), stride=(s2, 1))
        self.self_att = self_att(FRE * 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(FRE * 3)
        self.conv41 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))
        self.conv42 = nn.Conv2d(FRE * 3, FRE * 3, (1, 3))
        self.block41 = block(FRE * 3, FRE * 2)
        self.block42 = block(FRE * 3, FRE * 2)
        self.us421 = nn.ConvTranspose2d(FRE * 2, FRE * 2, kernel_size=(s1, 1), stride=(s1, 1))
        self.conv51 = nn.Conv2d(FRE * 2, FRE * 2, (1, 2))
        self.block51 = block(FRE * 2, FRE)
        self.bn51 = nn.BatchNorm2d(FRE); self.relu51 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(FRE, latent_dim, fs, padding=ps)

    def forward(self, x, Xavg=None, Xstd=None):
        x0 = self.bn0(x)
        x02 = self.block12(self.conv02(self.maxpool012(x0)))
        x11 = self.block11(self.conv1(x0))
        x11_n = torch.cat([x11, self.us121(x02, output_size=x11.shape)], dim=-1)
        x12_n = torch.cat([x02, self.dropout12(self.maxpool112(x11))], dim=-1)
        x13 = self.dropout123(self.maxpool123(x02))
        x21 = self.block21(self.conv21(x11_n))
        x22 = self.block22(self.conv22(x12_n))
        x23 = self.block23(self.conv23(x13))
        x21_n = torch.cat([x21, self.us221(x22, output_size=x21.shape)], dim=-1)
        x22_n = torch.cat([x22, self.dropout22(self.maxpool212(x21)), self.us232(x23, output_size=x22.shape)], dim=-1)
        x23 = self.bn23(self.self_att23(x23))
        x23_n = torch.cat([x23, self.dropout23(self.maxpool223(x22))], dim=-1)
        x31 = self.relu31(self.bn31(self.block31(self.conv31(x21_n))))
        x32 = self.relu32(self.bn32(self.block32(self.conv32(x22_n))))
        x33 = self.bn4(self.self_att(self.relu33(self.bn33(self.block33(self.conv33(x23_n))))))
        x31_n = torch.cat([x31, self.us321(x32, output_size=x31.shape)], dim=-1)
        x32_n = torch.cat([x32, self.us332(x33, output_size=x32.shape), self.dropout312(self.maxpool312(x31))], dim=-1)
        x41 = self.block41(self.conv41(x31_n))
        x42 = self.block42(self.conv42(x32_n))
        x51 = self.relu51(self.bn51(self.block51(self.conv51(torch.cat([x41, self.us421(x42, output_size=x41.shape)], dim=-1)))))
        return self.conv4(x51)


class ExpertConv1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c, c, 3, padding=1), nn.ReLU(), nn.Conv1d(c, c, 3, padding=1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MMoEBlock(nn.Module):
    def __init__(self, c, n_experts=4, n_tasks=4):
        super().__init__()
        self.experts = nn.ModuleList([ExpertConv1D(c) for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(c, n_experts) for _ in range(n_tasks)])

    def forward(self, feat):
        es = torch.stack([e(feat) for e in self.experts], dim=2)
        return [(es * g(feat).softmax(dim=-1).unsqueeze(-1)).sum(dim=2) for g in self.gates]


class MultiTaskCNN(nn.Module):
    """CNN encoder + MMoE with heads for dynamics, beat, downbeat and change point."""

    def __init__(self, cfg: dict):
        super().__init__()
        feat, cnn = cfg["feature"], cfg.get("cnn", {})
        if feat["audio_feature"] != "sone":
            raise ValueError("only the released 'sone' feature checkpoint is supported")
        self.extractor = PsychoFeatureExtractor(feat["sample_rate"], feat["fft_size"], feat["frames_per_second"])
        self.FRE = self.extractor.bark_bands
        latent_dim = int(cnn.get("latent_dim", 96))
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum=0.01)
        self.encoder = SYMultiScaleAttnFlex(self.FRE, latent_dim, temporal_strides=cnn.get("temporal_scale", [3, 3]))
        self.drop = nn.Dropout(float(cnn.get("fc_dropout", 0.2)))
        self.proj_dyn = nn.Linear(latent_dim, latent_dim)
        self.proj_beat = nn.Linear(latent_dim, latent_dim)
        self.proj_db = nn.Linear(latent_dim, latent_dim)
        self.proj_cp = nn.Linear(latent_dim, latent_dim)
        self.use_mmoe = bool(cnn.get("use_mmoe", False))
        if self.use_mmoe:
            self.mmoe = MMoEBlock(latent_dim, n_experts=int(cnn.get("n_experts", 4)), n_tasks=4)
        k = int(feat["dynamic_classes"]) + 1
        self.head_dyn_out = nn.Linear(latent_dim, k)
        self.head_beat_out = nn.Linear(latent_dim, 1)
        self.head_db_out = nn.Linear(latent_dim, 1)
        self.head_cp_out = nn.Linear(latent_dim, 1)

    def forward(self, wav):
        x = self.extractor(wav)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.bn0(x.unsqueeze(3))
        feat = self.drop(self.encoder(x).squeeze(-1).transpose(1, 2))
        h = [F.gelu(p(feat)) for p in (self.proj_dyn, self.proj_beat, self.proj_db, self.proj_cp)]
        if self.use_mmoe:
            h = self.mmoe(feat)
        return {"dynamic_output": self.head_dyn_out(h[0]), "beat_output": self.head_beat_out(h[1]),
                "downbeat_output": self.head_db_out(h[2]), "change_point_output": self.head_cp_out(h[3])}
