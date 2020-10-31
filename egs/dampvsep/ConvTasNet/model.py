import torch
from torch import nn

import asteroid.filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.optimizers import make_optimizer


class Model(nn.Module):
    def __init__(
        self,
        embedding,
        encoder,
        masker,
        decoder,
        enc_n_filters=256,
        enc_kernel_size=20,
        enc_stride=10,
        enc_padding=0,
        enc_dilation=1,
        sample_rate=16000,
        segment=3.0,
        emb_size=128,
    ):
        super(Model, self).__init__()
        conv1_shape = conv1_size_out(
            sample_rate=sample_rate,
            segment=segment,
            kernel_size=enc_kernel_size,
            dilation=enc_dilation,
            stride=enc_stride,
            padding=enc_padding,
        )
        self.embedding = embedding

        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

        self.adjust_shape = nn.Linear(
            conv1_shape * enc_n_filters + emb_size, conv1_shape * enc_n_filters
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)

        if self.embedding:
            intr_emb = self.embedding(x)
            shape_1 = tf_rep.shape[1]
            shape_2 = tf_rep.shape[2]
            flat_tf_rep = tf_rep.view(-1, shape_1 * shape_2)
            concat = torch.cat((flat_tf_rep, intr_emb), dim=1)
            tf_rep = nn.functional.relu(self.adjust_shape(concat))
            tf_rep = tf_rep.view(-1, shape_1, shape_2)

        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        return self.pad_output_to_inp(self.decoder(masked_tf_rep), x)


def make_model_and_optimizer(conf, embedding=None):
    """Function to define the model and optimizer for a config dictionary.
    Args:
        embedding: Embedding model
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    # Define building blocks for local model
    enc, dec = fb.make_enc_dec("free", **conf["filterbank"])
    masker = TDConvNet(in_chan=enc.n_feats_out, **conf["masknet"])

    embedding = embedding
    model = Model(embedding, enc, masker, dec, **conf["embedding"])
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer


def conv1_size_out(
    sample_rate=16000, segment=3.0, padding=0, dilation=1, kernel_size=20, stride=10
):
    out_size = (sample_rate * segment + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(out_size)
