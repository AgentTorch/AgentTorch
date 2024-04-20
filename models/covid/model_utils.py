''' County and State Data Processing network'''

import numpy as np
import torch
import torch.nn as nn
from data_utils import get_county_train_data, counties, create_window_seqs
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import numpy as np
import torch
import math
import pandas as pd

cuda = torch.device('cpu')
dtype = torch.float
SMOOTH_WINDOW = 7

class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)

    def forward_mask(self, seq, mask):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.exp(weights)
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(
            1, 2)
        weights = weights / (weights.sum(-1, keepdim=True))
        return (weights @ keys).transpose(1, 0) * mask


class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out //
            2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(in_features=self.rnn_out + self.dim_metadata,
                      out_features=self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)

    def forward_mask(self, seqs, metadata, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward(self, seqs, metadata):
        # Take last output from GRU
        latent_seqs, encoder_hidden = self.rnn(seqs)
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out, encoder_hidden


class DecodeSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 5,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(DecodeSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.act_fcn = nn.Tanh()

        # to embed input
        self.embed_input = nn.Linear(self.dim_seq_in, self.rnn_out)

        # to combine input and context
        self.attn_combine = nn.Linear(2 * self.rnn_out, self.rnn_out)

        self.rnn = nn.GRU(
            input_size=self.rnn_out,
            hidden_size=self.rnn_out //
            2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.out_layer = [
            nn.Linear(in_features=self.rnn_out, out_features=self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        # initialize
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)
        self.embed_input.apply(init_weights)
        self.attn_combine.apply(init_weights)

    def forward(self, Hi_data, encoder_hidden, context):
        # Hi_data is scaled time
        inputs = Hi_data.transpose(1, 0)
        if self.bidirectional:
            h0 = encoder_hidden[2:]
        else:
            h0 = encoder_hidden[2:].sum(0).unsqueeze(0)
        # combine input and context
        inputs = self.embed_input(inputs)
        # repeat context for each item in sequence
        context = context.repeat(inputs.shape[0], 1, 1)
        inputs = torch.cat((inputs, context), 2)
        inputs = self.attn_combine(inputs)
        # Take last output from GRU
        latent_seqs = self.rnn(inputs, h0)[0]
        latent_seqs = latent_seqs.transpose(1, 0)
        latent_seqs = self.out_layer(latent_seqs)
        return latent_seqs


''' smooth data with moving average (common with fitting mechanistic models) '''

def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values

''' Specify which state '''
def fetch_county_data_covid(state='MA',county_id='25005',pred_week='202021',batch_size=32,noise_level=0):
    ''' Import COVID data for counties 
        
        Processing:
            - Sequences input is scaled
            - one-hot encoding for region (county)
            - Moving average to target
    '''
    np.random.seed(17)

    if county_id == 'all':
        all_counties = counties[state]
    else:
        all_counties = [county_id]

    c_seqs = []  # county sequences of features
    c_ys = []  # county targets
    for county in all_counties:
        X_county, y = get_county_train_data(county,
                                            pred_week,
                                            noise_level=noise_level)
        y = moving_average(y[:, 1].ravel(), SMOOTH_WINDOW).reshape(-1, 1)
        c_seqs.append(X_county.to_numpy())
        c_ys.append(y)
    c_seqs = np.array(c_seqs)  # Shape: [regions, time, features]
    c_ys = np.array(c_ys)  # Shape: [regions, time, 1]

    # Normalize
    # One scaler per county
    scalers = [StandardScaler() for _ in range(len(all_counties))]
    c_seqs_norm = []
    for i, scaler in enumerate(scalers):
        c_seqs_norm.append(scaler.fit_transform(c_seqs[i]))
    c_seqs_norm = np.array(c_seqs_norm)
    ''' Create static metadata data for each county '''

    county_idx = {r: i for i, r in enumerate(all_counties)}

    def one_hot(idx, dim=len(county_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans

    metadata = np.array([one_hot(county_idx[r]) for r in all_counties])
    ''' Prepare train and validation dataset '''

    min_sequence_length = 20
    metas, seqs, y, y_mask = [], [], [], []
    for meta, seq, ys in zip(metadata, c_seqs_norm, c_ys):
        seq, ys, ys_mask = create_window_seqs(seq, ys, min_sequence_length)
        metas.append(meta)
        seqs.append(seq[[-1]])
        y.append(ys[[-1]])
        y_mask.append(ys_mask[[-1]])

    all_metas = np.array(metas, dtype="float32")
    all_county_seqs = torch.cat(seqs, axis=0)
    all_county_ys = torch.cat(y, axis=0)
    all_county_y_mask = torch.cat(y_mask, axis=0)

    counties_train, metas_train, X_train, y_train, y_mask_train = \
        all_counties, all_metas, all_county_seqs, all_county_ys, all_county_y_mask

    train_dataset = SeqData(counties_train, metas_train, X_train, y_train,
                            y_mask_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    assert all_county_seqs.shape[1] == all_county_ys.shape[1]
    seqlen = all_county_seqs.shape[1]
    return train_loader, metas_train.shape[1], X_train.shape[2], seqlen



# # dataset class
# class SeqData(torch.utils.data.Dataset):

#     def __init__(self, region, meta, X, y, mask_y):
#         self.region = region
#         self.meta = meta
#         self.X = X
#         self.y = y
#         # self.mask_y = mask_y

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         return (self.region[idx], self.meta[idx], self.X[idx, :, :],
#                 self.y[idx])


# class ODE(nn.Module):

#     def __init__(self, params, device):
#         super(ODE, self).__init__()
#         county_id = params['county_id']
#         abm_params = f'Data/{county_id}_generated_params.yaml'
#         #Reading params
#         with open(abm_params, 'r') as stream:
#             try:
#                 abm_params = yaml.safe_load(stream)
#             except yaml.YAMLError as exc:
#                 print('Error in reading parameters file')
#                 print(exc)
#         params.update(abm_params)
#         self.params = params
#         self.device = device
#         self.num_agents = self.params['num_agents']  # Population


# class SEIRM(ODE):

#     def __init__(self, params, device):
#         super().__init__(params, device)

#     def init_compartments(self, learnable_params):
#         ''' let's get initial conditions '''
#         initial_infections_percentage = learnable_params[
#             'initial_infections_percentage']
#         initial_conditions = torch.empty((5)).to(self.device)
#         no_infected = (initial_infections_percentage /
#                        100) * self.num_agents  # 1.0 is ILI
#         initial_conditions[2] = no_infected
#         initial_conditions[0] = self.num_agents - no_infected
#         print('initial infected', no_infected)
#         self.state = initial_conditions

#     def step(self, t, values):
#         """
#         Computes ODE states via equations       
#             state is the array of state value (S,E,I,R,M)
#         """
#         params = {
#             'beta': values[0],
#             'alpha': values[1],
#             'gamma': values[2],
#             'mu': values[3],
#             'initial_infections_percentage': values[4],
#         }
#         if t == 0:
#             self.init_compartments(params)
#         # to make the NN predict lower numbers, we can make its prediction to be N-Susceptible
#         dSE = params['beta'] * self.state[0] * self.state[2] / self.num_agents
#         dEI = params['alpha'] * self.state[1]
#         dIR = params['gamma'] * self.state[2]
#         dIM = params['mu'] * self.state[2]

#         dS = -1.0 * dSE
#         dE = dSE - dEI
#         dI = dEI - dIR - dIM
#         dR = dIR
#         dM = dIM

#         # concat and reshape to make it rows as obs, cols as states
#         self.dstate = torch.stack([dS, dE, dI, dR, dM], 0)
#         NEW_INFECTIONS_TODAY = dEI
#         NEW_DEATHS_TODAY = dIM
#         # update state
#         self.state = self.state + self.dstate

#         return NEW_INFECTIONS_TODAY, NEW_DEATHS_TODAY


# class SIRS(ODE):

#     def __init__(self, params, device):
#         super().__init__(params, device)

#     def init_compartments(self, learnable_params):
#         ''' let's get initial conditions '''
#         initial_infections_percentage = learnable_params[
#             'initial_infections_percentage']
#         initial_conditions = torch.empty((2)).to(self.device)
#         no_infected = (initial_infections_percentage /
#                        100) * self.num_agents  # 1.0 is ILI
#         initial_conditions[1] = no_infected
#         initial_conditions[0] = self.num_agents - no_infected
#         print('initial infected', no_infected)

#         self.state = initial_conditions

#     def step(self, t, values):
#         """
#         Computes ODE states via equations       
#             state is the array of state value (S,I)
#         """
#         params = {
#             'beta': values[0],  # contact rate, range: 0-1
#             'initial_infections_percentage': values[1],
#         }
#         # set from expertise
#         params['D'] = 3.5
#         params['L'] = 2000
#         if t == 0:
#             self.init_compartments(params)
#         dS = (self.num_agents - self.state[0] -
#               self.state[1]) / params['L'] - params['beta'] * self.state[
#                   0] * self.state[1] / self.num_agents
#         dSI = params['beta'] * self.state[0] * self.state[1] / self.num_agents
#         dI = dSI - self.state[1] / params['D']

#         # concat and reshape to make it rows as obs, cols as states
#         self.dstate = torch.stack([dS, dI], 0)

#         NEW_INFECTIONS_TODAY = dSI
#         # ILI is percentage of outpatients with influenza-like illness
#         # ILI = params['lambda'] * dSI / self.num_agents
#         # this is what Shaman and Pei do https://github.com/SenPei-CU/Multi-Pathogen_ILI_Forecast/blob/master/code/SIRS_AH.m
#         ILI = dSI / self.num_agents * 100  # multiply 100 because it is percentage

#         # update state
#         self.state = self.state + self.dstate
#         return NEW_INFECTIONS_TODAY, ILI


if __name__ == '__main__':
    print("THIS SHOULD NOT EXECUTE!")
    """ Create model """