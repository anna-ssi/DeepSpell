import random

import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, enc, dec):
        batch_size = dec.shape[0]
        trg_vocab_size = self.decoder.output_dim

        hidden, cell = self.encoder(enc)

        outputs, hidden, cell = self.decoder(dec, hidden, cell)

        top = outputs.max(1)

        return top
