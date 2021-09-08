import torch
import math
from torch import nn
from torch._C import device


class PositionalEncoding(nn.Module):
    """Vanilla PositionalEncoding encoding

    According to the Attention is all you need paper, this positional
    encoding inserts the notion of position in the input features.

    Args:
        d_model : Hidden dimensionality of the input.
        max_len : Maximum length of a sequence to expect
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerTimeSeries(nn.Module):
    """Transformer model adapted to time series data

    This model follows the implementation from [1], which adapted the original
    model from [2] to perform a time series prediction task instead of the original
    task of NLP. The adaptation to the time series task removed the input
    embedding from the encoder but maintained the positional encoding process.

    References:
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    [2] Wu, Neo, et al. "Deep transformer models for time series forecasting: The influenza
        prevalence case." arXiv preprint arXiv:2001.08317 (2020).

    Args:
        device            : Torch device to process the model
        encoder_vector_sz : Size of the encoder input vector, which also define the
                            number the encoder's input features.
        decoder_vector_sz : Size of the decoder input vector, which also define the
                            number the decoder's input features.
        encoder_mask      : Either use or not a mask to avoid look ahead in the encoder.
        decoder_mask      : Either use or not a mask to avoid look ahead in the decoder.
        num_layers        : Number of decoder and encoder layers.
        nhead             : Number of parallel multi attention heads.
        n_time_steps      : Number of timesteps used as inputs in the encoder and
                            decoder.
        d_model           : Model's vector dimmension.
        max_len : Array of objects with simulation data

    """
    def __init__(self,
                 device,
                 encoder_vector_sz=1,
                 decoder_vector_sz=1,
                 encoder_mask=False,
                 decoder_mask=False,
                 encoder_feedback=False,
                 num_layers=1,
                 nhead=10,
                 n_encoder_time_steps=30,
                 n_output_time_steps=30,
                 output_vector_sz=1,
                 d_model=90,
                 encoder_only=False,
                 dropout=0.5):
        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.device = device
        self.n_encoder_time_steps = n_encoder_time_steps
        self.encoder_mask = encoder_mask
        self.decoder_mask = decoder_mask
        self.encoder_only = encoder_only
        self.encoder_feedback = encoder_feedback
        self.n_encoder_time_steps = n_encoder_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.output_vector_sz = output_vector_sz

        # Positional encoding used in the decoder and encoder
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder and encoder layers
        # Note that this implementation follows the data input format of
        # (batch, timestamps, features).
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_layers)
        # Encoder input projection
        self.encoder_projection = nn.Linear(encoder_vector_sz, d_model)

        if encoder_only == False:
            # Positional encoding used in the decoder and encoder
            self.pos_decoder = PositionalEncoding(d_model)

            self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                            nhead=nhead,
                                                            dropout=dropout,
                                                            batch_first=True)

            self.transformer_decoder = nn.TransformerDecoder(
                self.decoder_layer, num_layers=num_layers)

            # Decoder input projectors
            if encoder_feedback:
                self.decoder_projection = nn.Linear(d_model, d_model)
            else:
                self.decoder_projection = nn.Linear(decoder_vector_sz, d_model)

        # Transformer output layer
        # Note that the output format of this model is (batch, pred_steps), where
        # as a time series predictor the pred_steps contain the estimation of future
        # behavior of one variable in time.
        self.linear = nn.Linear(d_model, self.output_vector_sz)
        self.out_fcn = nn.ReLU()
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        """Generates decoder mask

        This mask prevents the look ahead behavior in the decoder process.

        Args:
            sz : Size of decoder mask.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def encoder_process(self, src):
        """Encoder process of the Transformer network.

        According to [1], the tranformer encoder produces an encoded version
        of the entry series with the positional information and a self learned
        projection of the input data, generating a encoder matrix of dmodel
        dimension.

        Args:
            src: Input data of  the encoder, with shape
                 (batch, n_time_steps, encoder_vector_sz)
        """
        x = src
        x = self.encoder_projection(x)
        x = self.pos_encoder(x)

        if self.encoder_mask:
            # Mask to avoid look ahead behavior in the encoder process
            mask = self._generate_square_subsequent_mask(
                self.n_encoder_time_steps).to(self.device)
        else:
            mask = None

        x = self.transformer_encoder(x, mask=mask)

        return x

    def decoder_process(self, tgt, memory):
        """Decoder process of the Transformer network.

        The tranformer decoder process the encoder's output, which is the
        encoder's representation of the input, and the decoder's input
        to estimate the future values a particular feature.


        Args:
            tgt : Decoder input, which is previous values of the target
                  features, and depending on the way used to process information
                  this can also be previous outputs of the transformer network.

            memory : Memory, that is the encoder representation of its input.
        """
        x = tgt
        x = self.decoder_projection(tgt)
        x = self.pos_decoder(x)

        if self.decoder_mask:
            # Mask to avoid look ahead behavior in the decoder process
            mask = self._generate_square_subsequent_mask(
                self.n_output_time_steps).to(self.device)
        else:
            mask = None
        out = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask)

        return out

    def init_weights(self):
        """Initialize the output layer weights.
        """
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """Send the information through the transformer network.
        """
        if self.encoder_feedback:
            src = x
        else:
            src, tgt = x

        src = self.encoder_process(src)
        if self.encoder_only:
            out = self.linear(src)
            out = self.out_fcn(out)
        else:
            if self.encoder_feedback:
                # If the feedbeck is on pass the encoder output as the
                # decoder input sequence
                tgt = self.pos_decoder(src)
            out = self.decoder_process(tgt, src)
            out = self.linear(out)

        # Reshapes the output to (batch, n_pred_steps)
        out = torch.reshape(out, (-1, out.shape[1]))

        return out

    def encoder_attention(self, src):
        """Return the attention matrix given an x input

        Args:
            src : Input tensor with shape (batch, timesteps, dmodel)
        """
        x = src
        x = self.encoder_projection(x)
        x = self.pos_encoder(x)

        if self.encoder_mask:
            # Mask to avoid look ahead behavior in the decoder process
            mask = self._generate_square_subsequent_mask(src.shape[1]).to(
                self.device)
        else:
            mask = None
        return self.encoder_layer.self_attn(x, x, x, attn_mask=mask)

    def decoder_attention(self, x):
        """Return the attention matrix given an x input

        Args:
            x : Input tensor with shape (batch, timesteps, dmodel)
        """
        return self.decoder_layer.self_attn(x, x, x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = TransformerTimeSeries(device,
                                  n_encoder_time_steps=30,
                                  n_output_time_steps=30,
                                  encoder_feedback=True).to(device)
    src = torch.rand(10, 30, 1).to(device)
    tgt_in = torch.rand(10, 30, 1).to(device)

    out = model(src)
    print(f'output {out.shape}')

    print(f" Attention matrix: {model.encoder_attention(src)}")


if __name__ == '__main__':
    main()