import torch
from torch import nn

import numpy as np
from scipy import stats

from arff2pandas import a2p
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# cuda availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def create_dataset(df_):
    """
        helper function to create tensors from the dataset
    """
    sequences = df_.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def vis_median_mad(SIGS, n_time, legend, colors, name, verts=[], alpha=.25):
    T, idx = np.array(list(range(n_time))), 0

    plt.figure()
    ax = plt.gca()
    for sigs in SIGS:
        sigs_median = np.median(sigs, axis=0)
        sigs_mad = stats.median_abs_deviation(sigs, axis=0)

        ax.plot(T, sigs_median, label=legend[idx], color=colors[idx])
        ax.fill_between(T, sigs_median-sigs_mad, sigs_median+sigs_mad, alpha=alpha, color=colors[idx])

        idx += 1

    for vert in verts:
        ax.plot([vert, vert], [-5, 3], 'k--')

    plt.xlabel('Time', fontsize=14)
    plt.ylim(-4.9, 2.4)
    plt.xticks([0, 50, 100, 140], fontsize=12)
    plt.yticks([-4, -2, 0, 2], fontsize=12)
    plt.legend(loc=8, fontsize=14)
    plt.locator_params(nbins=4)
    plt.gca().set_aspect(10, adjustable='box')
    # plt.savefig('ecg_eig_{}.pdf'.format(name),
    #             transparent=True, bbox_inches='tight', pad_inches=0, dpi=1200)
    plt.show()


# choose threshold to detect anomaly
def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


def get_loss_and_pred(model, dset):
    predictions, losses = predict(model, dset)
    sns.distplot(losses, bins=50, kde=True)
    plt.show()
    return predictions, losses


def ecg_data(basedir, args):
    with open(basedir + 'ECG5000_TRAIN.arff') as f:
        train = a2p.load(f)

    with open(basedir + 'ECG5000_TEST.arff') as f:
        test = a2p.load(f)

    # combine the training and test data into a single data frame. This will give us more data to train our Autoencoder.
    # also shuffle it
    df = train.append(test)
    df = df.sample(frac=1.0)

    # 5000 examples total, each row represent single heartbeat record
    CLASS_NORMAL = 1
    # class_names = ['Normal', 'R on T', 'PVC', 'SP', 'UB']

    # rename target column (classifier)
    new_columns = list(df.columns)
    new_columns[-1] = 'target'
    df.columns = new_columns

    # check out how many examples we have of each class
    df.target.value_counts()

    # get all normal heartbeats and drop the target (class) column:
    normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)

    # merge all other classes and mark them as anomalies
    anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

    # split the normal examples into train, validation and test sets
    train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=args.seed)
    val_df, test_df = train_test_split(val_df, test_size=0.33, random_state=args.seed)

    # create datasets
    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, _, _ = create_dataset(val_df)
    test_normal_dataset, _, _ = create_dataset(test_df)
    test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

    return seq_len, n_features, train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset


def unroll_states(model, ds, bsz, args):
    model.eval()
    hs = []
    for seq_true in ds[:bsz]:
        seq_true = seq_true.to(args.device)
        states = model.encoder.unroll_states(seq_true)
        hs.append(states.squeeze().detach().cpu().numpy())

    hs = np.stack(hs)
    init = np.zeros(hs.shape[-1])
    Z = np.insert(hs, 0, init, axis=1)

    return Z


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, rnn_type, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn_type = rnn_type

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.n_features,
                hidden_size=embedding_dim,
                num_layers=1,
                batch_first=True
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.n_features,
                hidden_size=embedding_dim,
                num_layers=1,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=self.n_features,
                hidden_size=embedding_dim,
                num_layers=1,
                batch_first=True
            )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        if self.rnn_type == "LSTM":
            x, (hidden_n, _) = self.rnn(x)
        else:
            x, hidden_n = self.rnn(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))

    def unroll_states(self, x):
        with torch.no_grad():
            x = x.reshape((1, self.seq_len, self.n_features))
            if self.rnn_type == "LSTM":
                x, (hidden_n, _) = self.rnn(x)
            else:
                x, hidden_n = self.rnn(x)
            return x


class Decoder(nn.Module):
    def __init__(self, seq_len, rnn_type, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn_type = rnn_type

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True
            )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, _ = self.rnn(x)

        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, rnn_type, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, rnn_type, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, rnn_type, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

