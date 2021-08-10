import time
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class TransformerSet(Dataset):
    """Transformer dataset object

   This dataset object ensures that the transformer network is
   feed correctly and also saves the true targets values for
   posterior evaluation.

    Args:
        x_matrix : Matrix containing the past steps of a univariate
                   time series. With shape (time_steps, window_of_features)

        y_matrix : Matrix containing the future steps of a univariate time series.
                   With shape (time_steps, window_of_features)

        n_time_steps: Number of timesteps used in the entry of the transformer.
    """
    def __init__(self, x_matrix, y_matrix, n_time_steps):

        x_matrix = x_matrix.reshape(-1, n_time_steps, 1)
        n_encoder_input_steps = n_time_steps // 2
        self.encoder_input = x_matrix[:, :n_encoder_input_steps]
        self.decoder_input = x_matrix[:, n_encoder_input_steps:]
        self.true_target = y_matrix

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[
            idx], self.true_target[idx]


def batch_train(model, epoch, batch_size, train_loader, criterion, optimizer,
                scheduler, set_size, device):
    """Train the model in batch
    """
    model.train()  # Turn on the train mode
    batch_loss = 0.
    total_loss = 0.
    start_time = time.time()
    predictions = torch.tensor([]).to(device)
    ground_truth = torch.tensor([]).to(device)

    for i, batch in enumerate(train_loader):
        src, tgt_in, tgt_out = batch[0], batch[1], batch[2]
        src = Variable(torch.Tensor(src.float())).to(device)
        tgt_in = Variable(torch.Tensor(tgt_in.float())).to(device)
        tgt_out = Variable(torch.Tensor(tgt_out.float())).to(device)

        optimizer.zero_grad()
        output = model((src, tgt_in))
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        total_loss += batch_loss
        log_interval = int(set_size / batch_size / 5)
        if i % log_interval == 0 and i > 0:
            cur_loss = batch_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(epoch, i, set_size // batch_size,
                                        scheduler.get_lr()[0],
                                        elapsed * 1000 / log_interval,
                                        cur_loss))
            batch_loss = 0
            start_time = time.time()

    return total_loss