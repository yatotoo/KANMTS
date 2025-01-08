import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    # input（32,11,512）
    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape


        # （32,11,512）
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)  # （32,11,512）

        # stochastic pooling

        if self.training:
            # #（32,11,512）对 combined_mean
            ratio = F.softmax(combined_mean, dim=1)  # softmax
            ratio = ratio.permute(0, 2, 1)  # （32,512,11）
            ratio = ratio.reshape(-1, channels)  # （32*512=16384,11）
            indices = torch.multinomial(ratio, 1)  # 16384,1
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)  # （32,1,512）
            combined_mean = torch.gather(combined_mean, 1, indices)  # （32,1,512）
            combined_mean = combined_mean.repeat(1, channels, 1)  # （32,11,512）
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusionMLP

        combined_mean_cat = torch.cat([input, combined_mean], -1)  # （32,11,1024）

        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat)) # （32,11,512）
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat  #（32,11,512）

        return output, None


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )

        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    # x_enc（32,96,7 ），x_mark_enc（32,96,4 ）
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # （32,11,512）值嵌入[Batch Variate Time]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # enc_out（32,11,512）

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]  # （32,96,7）

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
