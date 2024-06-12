import torch
import torch.nn as nn
from model_utils import EmbedAttenSeq, DecodeSeq

MIN_VAL_PARAMS = {
    "abm-covid": [
        1.4,
        0.001,
        0.1,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
    ],  # new start date on 202045 + beta dist params
    "abm-flu": [1.05, 0.1],
    "seirm": [0.0, 0.0, 0.0, 0.0, 0.01],
    "sirs": [0.0, 0.1],
}
MAX_VAL_PARAMS = {
    "abm-covid": [
        4.5,
        0.01,
        1,
        0.05,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],  # new start date on 202045  + beta dist params
    "abm-flu": [2.6, 5.0],
    "seirm": [1.0, 1.0, 1.0, 1.0, 1.0],
    "sirs": [1.0, 5.0],
}

WEEKS_AHEAD = 4

class CalibNN(nn.Module):
    def __init__(
        self,
        metas_train_dim,
        X_train_dim,
        device,
        training_weeks,
        hidden_dim=32,
        out_dim=1,
        n_layers=2,
        scale_output="abm-covid",
        bidirectional=True,
    ):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks
        """ tune """
        hidden_dim = 64
        out_layer_dim = 32

        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim,  # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        )

        out_layer_width = out_layer_dim
        self.hidden_layers = [
            nn.Linear(in_features=out_layer_width, out_features=out_layer_width // 2),
            nn.ReLU(),
            # nn.Linear(in_features=out_layer_width // 2, out_features=out_dim),
        ]
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        # we want to separate this layer to analyze gradients
        self.param_out_layer = nn.Linear(
            in_features=out_layer_width // 2, out_features=out_dim
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.hidden_layers.apply(init_weights)
        self.param_out_layer.apply(init_weights)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output], device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output], device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta):
        x, meta = x.to(self.device), meta.to(self.device)
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)

        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = (
            torch.arange(1, self.training_weeks + WEEKS_AHEAD + 1)
            .repeat(x_embeds.shape[0], 1)
            .unsqueeze(2)
        )
        Hi_data = ((time_seq - time_seq.min()) / (time_seq.max() - time_seq.min())).to(
            self.device
        )
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds)
        emb = self.hidden_layers(emb)
        out = self.param_out_layer(emb)
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out

class CalibAlignNN(nn.Module):
    def __init__(
        self,
        metas_train_dim,
        X_train_dim,
        device,
        training_weeks,
        hidden_dim=32,
        out_dim=1,
        out_dim_align=6,  # number of masks for tensor
        n_layers=2,
        scale_output="abm-covid",
        bidirectional=True,
    ):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks
        """ tune """
        hidden_dim = 64
        out_layer_dim = 32

        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim,  # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        )

        out_layer_width = out_layer_dim
        self.hidden_layers = [
            nn.Linear(in_features=out_layer_width, out_features=out_layer_width // 2),
            nn.ReLU(),
            # nn.Linear(in_features=out_layer_width // 2, out_features=out_dim),
        ]
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        # we want to separate this layer to analyze gradients
        self.param_out_layer = nn.Linear(
            in_features=7 * out_layer_width // 2, out_features=out_dim
        )

        self.align_out_layer = nn.Linear(
            in_features=7 * out_layer_width // 2, out_features=out_dim_align
        )
        self.align_adjust_layer = nn.Linear(
            in_features=7 * out_layer_width // 2, out_features=out_dim_align
        )
        self.initial_infect_layer = nn.Linear(
            in_features=7 * out_layer_width // 2, out_features=out_dim_align
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.hidden_layers.apply(init_weights)
        self.param_out_layer.apply(init_weights)
        self.align_out_layer.apply(init_weights)
        self.initial_infect_layer.apply(init_weights)

        self.flatten = nn.Flatten()

        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output], device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output], device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta):
        x, meta = x.to(self.device), meta.to(self.device)
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)

        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = (
            torch.arange(1, self.training_weeks + WEEKS_AHEAD + 1)
            .repeat(x_embeds.shape[0], 1)
            .unsqueeze(2)
        )
        Hi_data = ((time_seq - time_seq.min()) / (time_seq.max() - time_seq.min())).to(
            self.device
        )
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds)
        emb = self.hidden_layers(emb)  # [2, 3, 7]
        emb = self.flatten(emb)  # [2, 21]

        # calibration output
        out = self.param_out_layer(emb)
        calib_out = self.min_values[0] + (
            self.max_values[0] - self.min_values[0]
        ) * self.sigmoid(
            out
        )  # [2]

        align_out = self.align_out_layer(emb)  # (num_weeks, num_age_groups)
        align_out = self.sigmoid(align_out)

        # get the additional part of the alignment
        align_adjust = self.align_adjust_layer(emb)
        align_adjust = self.sigmoid(align_adjust)

        initial_infection_prob = self.initial_infect_layer(emb)
        initial_infection_prob = self.sigmoid(initial_infection_prob)

        return calib_out, align_out, align_adjust, initial_infection_prob


class LearnableParams(nn.Module):
    """doesn't use data signals"""

    def __init__(self, num_params, device, scale_output="abm-covid"):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor(
            MIN_VAL_PARAMS[scale_output][: self.num_params], device=self.device
        )
        self.max_values = torch.tensor(
            MAX_VAL_PARAMS[scale_output][: self.num_params], device=self.device
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        print("out shape: ", out.shape)
        """ bound output """
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out
