# ------------------------------------------------------------------------
# TinyHAR model based on architecture suggested by Zhou et al.
# ------------------------------------------------------------------------
# https://github.com/teco-kit/ISWC22-HAR
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# https://github.com/mariusbock/tal_for_har/blob/master/inertial_baseline/TinyHAR.py
# ------------------------------------------------------------------------
# A partly adaption for WEAR dataset by: Zeyneddin Oz
# E-Mail: zeyneddin.oez@uni-siegen.de
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelfAttention_interaction(nn.Module):
    def __init__(self, sensor_channel, n_channels):
        super(SelfAttention_interaction, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)
        o = (
            self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta)
            + x.permute(0, 2, 1).contiguous()
        )

        return o.permute(0, 2, 1).contiguous()


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer_interaction(nn.Module):
    def __init__(
        self,
        sensor_channel,
        dim,
        depth=1,
        heads=4,
        dim_head=16,
        mlp_dim=16,
        dropout=0.0,
    ):
        super(Transformer_interaction, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Identity(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class seperate_FC_interaction(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(seperate_FC_interaction, self).__init__()
        self.fc_filter = nn.Linear(filter_num, filter_num)
        self.fc_channel = nn.Linear(sensor_channel, sensor_channel)

    def forward(self, x):
        x = self.fc_channel(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.fc_filter(x)
        return x


crosschannel_interaction = {
    "attn": SelfAttention_interaction,
    "transformer": Transformer_interaction,
    "identity": Identity,
    "FCinter": seperate_FC_interaction,
}


class FilterWeighted_Aggregation(nn.Module):
    def __init__(self, sensor_channel, n_channels):
        super(FilterWeighted_Aggregation, self).__init__()
        self.value_projection = nn.Linear(n_channels, n_channels)
        self.value_activation = nn.ReLU()

        self.weight_projection = nn.Linear(n_channels, n_channels)
        self.weighs_activation = nn.Tanh()
        self.softmatx = nn.Softmax(dim=1)

    def forward(self, x):
        weights = self.weighs_activation(self.weight_projection(x))
        weights = self.softmatx(weights)

        values = self.value_activation(self.value_projection(x))

        values = torch.mul(values, weights)
        return torch.sum(values, dim=1)


class NaiveWeighted_Aggregation(nn.Module):
    """Temporal attention module."""

    def __init__(self, sensor_channel, hidden_dim):
        super(NaiveWeighted_Aggregation, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc(x).squeeze(2)

        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        return context


class Weighted_Aggregation(nn.Module):
    """Temporal attention module."""

    def __init__(self, sensor_channel, hidden_dim):
        super(Weighted_Aggregation, self).__init__()
        self.weight_projection = nn.Linear(hidden_dim, hidden_dim)
        self.weighs_activation = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.weighs_activation(self.weight_projection(x))
        out = self.fc(x).squeeze(2)

        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        return context


class FC(nn.Module):

    def __init__(self, channel_in, channel_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(channel_in, channel_out)

    def forward(self, x):
        x = self.fc(x)
        return x


class seperate_FC_channel_first(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(seperate_FC_channel_first, self).__init__()
        self.fc_channel = nn.Linear(sensor_channel, 1)
        self.fc_filter = nn.Linear(filter_num, filter_num)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.fc_channel(x).squeeze(3)
        x = self.fc_filter(x)
        return x


class seperate_FC_filter_first(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(seperate_FC_filter_first, self).__init__()
        self.fc_filter = nn.Linear(filter_num, 1)
        self.fc_channel = nn.Linear(sensor_channel, filter_num)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc_filter(x).squeeze(3)
        x = self.fc_channel(x)
        x = self.activation(x)
        return x


class seperate_FC_filter_first_v2(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(seperate_FC_filter_first_v2, self).__init__()
        self.fc_filter_1 = nn.Linear(filter_num, filter_num)
        self.fc_channel_1 = nn.Linear(sensor_channel, sensor_channel)
        self.activation = nn.ReLU()

        self.fc_filter_2 = nn.Linear(filter_num, 1)
        self.fc_channel_2 = nn.Linear(sensor_channel, filter_num)

    def forward(self, x):
        x = self.activation(self.fc_filter_1(x))
        x = x.permute(0, 1, 3, 2)
        x = self.activation(self.fc_channel_1(x))
        x = x.permute(0, 1, 3, 2)

        x = self.fc_filter_2(x).squeeze(3)
        x = self.activation(self.fc_channel_2(x))
        return x


class FC_Weighted_Aggregation(nn.Module):
    def __init__(self, sensor_channel, hidden_dim):
        super(FC_Weighted_Aggregation, self).__init__()

        self.fc_filter_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_channel_1 = nn.Linear(sensor_channel, sensor_channel)
        self.activation = nn.ReLU()

        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.fc_filter_1(x)).permute(0, 2, 1)
        x = self.activation(self.fc_channel_1(x)).permute(0, 2, 1)

        out = self.fc(x).squeeze(2)

        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        return context


crosschannel_aggregation = {
    "filter": FilterWeighted_Aggregation,
    "naive": NaiveWeighted_Aggregation,
    "FCnaive": FC_Weighted_Aggregation,
    "naive2": Weighted_Aggregation,
    "FC": FC,
    "SFCF": seperate_FC_filter_first,
    "SFCF2": seperate_FC_filter_first_v2,
    "SFCC": seperate_FC_channel_first,
}


class temporal_GRU(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(temporal_GRU, self).__init__()
        self.rnn = nn.GRU(
            filter_num, filter_num, 1, bidirectional=False, batch_first=True
        )

    def forward(self, x):
        # Batch length Filter
        outputs, h = self.rnn(x)
        return outputs


class temporal_LSTM(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(temporal_LSTM, self).__init__()
        self.lstm = nn.LSTM(filter_num, filter_num, batch_first=True)

    def forward(self, x):
        outputs, h = self.lstm(x)
        return outputs


class temporal_conv_1d(nn.Module):
    def __init__(self, sensor_channel, filter_num, nb_layers=2):
        super(temporal_conv_1d, self).__init__()
        filter_num_list = [filter_num]
        # filter_num_step=int(filter_num/nb_layers)
        for i in range(nb_layers - 1):
            # filter_num_list.append((1+i)*filter_num_step)
            filter_num_list.append(filter_num)
        # filter_num_list.append(1)
        filter_num_list.append(filter_num)
        layers_conv = []
        for i in range(nb_layers):
            in_channel = filter_num_list[i]
            out_channel = filter_num_list[i + 1]
            layers_conv.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channel,
                        out_channel,
                        5,
                        padding="same",
                        padding_mode="replicate",
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.layers_conv = nn.ModuleList(layers_conv)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers_conv:
            x = layer(x)
        x = x.permute(0, 2, 1)
        return x


temporal_interaction = {
    "gru": temporal_GRU,
    "lstm": temporal_LSTM,
    "attn": SelfAttention_interaction,
    "transformer": Transformer_interaction,
    "identity": Identity,
    "conv": temporal_conv_1d,
}


class Temporal_Weighted_Aggregation(nn.Module):
    def __init__(self, sensor_channel, hidden_dim):
        super(Temporal_Weighted_Aggregation, self).__init__()

        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.weighs_activation = nn.Tanh()
        self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
        self.sm = torch.nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        out = self.weighs_activation(self.fc_1(x))
        out = self.fc_2(out).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        context = x[:, -1, :] + self.gamma * context
        return context


temmporal_aggregation = {
    "filter": FilterWeighted_Aggregation,
    "naive": NaiveWeighted_Aggregation,
    "tnaive": Temporal_Weighted_Aggregation,
    "FC": FC,
    "identiry": Identity,
}


class TinyHAR(nn.Module):
    def __init__(
        self,
        input_shape,
        number_class,
        filter_num,
        nb_conv_layers=4,
        filter_size=5,
        cross_channel_interaction_type="attn",  # attn  transformer  identity
        cross_channel_aggregation_type="FC",  # filter  naive  FC
        temporal_info_interaction_type="lstm",  # gru  lstm  attn  transformer  identity
        temporal_info_aggregation_type="naive",  # naive  filter  FC
        dropout=0.1,  # 0.5
        activation="ReLU",
        feature_extract=None,
    ):
        super(TinyHAR, self).__init__()

        self.feature_extract = feature_extract

        self.cross_channel_interaction_type = cross_channel_interaction_type
        self.cross_channel_aggregation_type = cross_channel_aggregation_type
        self.temporal_info_interaction_type = temporal_info_interaction_type
        self.temporal_info_aggregation_type = temporal_info_aggregation_type
        """
        PART 1 , ============= Channel wise Feature Extraction =============================        
        """
        filter_num_list = [1]
        for i in range(nb_conv_layers - 1):
            filter_num_list.append(filter_num)
        filter_num_list.append(filter_num)

        layers_conv = []
        for i in range(nb_conv_layers):
            in_channel = filter_num_list[i]
            out_channel = filter_num_list[i + 1]
            if i % 2 == 1:
                layers_conv.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, (filter_size, 1), (2, 1)),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(out_channel),
                    )
                )
            else:
                layers_conv.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, (filter_size, 1), (1, 1)),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(out_channel),
                    )
                )
        self.layers_conv = nn.ModuleList(layers_conv)
        downsampling_length = self.get_the_shape(input_shape)
        """
        PART2 , ================ Cross Channel interaction  =================================
        """
        self.channel_interaction = crosschannel_interaction[
            cross_channel_interaction_type
        ](input_shape[3], filter_num)
        """
        PART3 , =============== Cross Channel Fusion  ====================================
        """
        if cross_channel_aggregation_type == "FC":
            self.channel_fusion = crosschannel_aggregation[
                cross_channel_aggregation_type
            ](input_shape[3] * filter_num, 2 * filter_num)
        elif cross_channel_aggregation_type in ["SFCC", "SFCF"]:
            self.channel_fusion = crosschannel_aggregation[
                cross_channel_aggregation_type
            ](input_shape[3], 2 * filter_num)
        else:
            self.channel_fusion = crosschannel_aggregation[
                cross_channel_aggregation_type
            ](input_shape[3], 2 * filter_num)

        self.activation = nn.ReLU()
        """
        PART4  , ============= Temporal information Extraction =========================
        """
        self.temporal_interaction = temporal_interaction[
            temporal_info_interaction_type
        ](input_shape[3], 2 * filter_num)
        """
        PART 5 , =================== Temporal information Aggregation ================
        """
        self.dropout = nn.Dropout(dropout)
        if temporal_info_aggregation_type == "FC":
            self.flatten = nn.Flatten()
            self.temporal_fusion = temmporal_aggregation[
                temporal_info_aggregation_type
            ](downsampling_length * 2 * filter_num, 2 * filter_num)
        else:
            self.temporal_fusion = temmporal_aggregation[
                temporal_info_aggregation_type
            ](input_shape[3], 2 * filter_num)

        """
        PART 6 , =================== Prediction ================
        """
        self.prediction = nn.Linear(2 * filter_num, number_class)

    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)

        for layer in self.layers_conv:
            x = layer(x)

        return x.shape[2]

    def forward(self, x):
        # Remove the unsqueeze, since the input is already in the correct shape
        # x = x.unsqueeze(1)
        for layer in self.layers_conv:
            x = layer(x)

        x = x.permute(0, 3, 2, 1)
        """ =============== cross channel interaction ==============="""
        x = torch.cat(
            [
                self.channel_interaction(x[:, :, t, :]).unsqueeze(3)
                for t in range(x.shape[2])
            ],
            dim=-1,
        )
        x = self.dropout(x)
        """=============== cross channel fusion ==============="""
        if self.cross_channel_aggregation_type == "FC":
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = self.activation(self.channel_fusion(x))
        elif self.cross_channel_aggregation_type in ["SFCC", "SFCF", "SFCF2"]:
            x = x.permute(0, 3, 1, 2)
            x = self.activation(self.channel_fusion(x))
        else:
            x = torch.cat(
                [
                    self.channel_fusion(x[:, :, :, t]).unsqueeze(2)
                    for t in range(x.shape[3])
                ],
                dim=-1,
            )
            x = x.permute(0, 2, 1)
            x = self.activation(x)

        """cross temporal interaction """
        x = self.temporal_interaction(x)
        """cross temporal fusion """
        if self.temporal_info_aggregation_type == "FC":
            x = self.flatten(x)
            x = self.activation(self.temporal_fusion(x))  # B L C
        else:
            x = self.temporal_fusion(x)

        y = self.prediction(x)
        if self.feature_extract:
            return y, x
        else:
            return y
