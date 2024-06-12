""" County and State Data Processing network"""
import torch
import torch.nn as nn

import torch.nn as nn
import torch
import math
import pandas as pd

cuda = torch.device("cpu")
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
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(1, 2)
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
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata, out_features=self.dim_out
            ),
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
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
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

def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values
