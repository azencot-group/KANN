import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import kann


def print_review(vocab, review, idx):
    rev_text = [vocab[i][:-1] for i in review[:idx]]
    print(rev_text)

    return rev_text


def plot_shaded_review(pref, rev_text, coef, cmap):
    wd = 60
    ht = int(np.ceil(len(rev_text) / float(wd)))

    sc = np.abs(coef)
    ii, jj, kk = 0, 0, 0
    M = np.zeros((ht, wd))
    for ch in rev_text:
        M[ii, jj] = sc[kk]
        jj += 1

        if jj == wd:
            jj, ii = 0, ii+1

        if ch == ' ':
            kk += 1
    if ii == ht:
        M[ii-1, -1] = 1.5 * np.max(sc)
    else:
        M[ii, jj:] = 1.5*np.max(sc)

    chars = []
    for tt in range(ht):
        chars.append(rev_text[tt*wd:(tt+1)*wd])
    chars = np.stack(chars)
    chars[-1] = chars[-1] + ' ' * (wd - len(chars[-1]))      # append blanks to fill last row

    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(M, cmap=cmap)

    # plot text in center of mat cells
    for (j, i), _ in np.ndenumerate(M):
        ax.text(i, j, chars[j][i], ha='center', va='center', fontsize=11)
    plt.axis('off')
    plt.savefig(pref + '.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.show()


def compute_projection_magnitude(model, batch, device):
    test_x, test_y, test_z = np.array(batch['inputs']), np.array(batch['labels']), np.array(batch['index'])

    model.eval()
    with torch.no_grad():
        h0 = model.init_hidden(test_x.shape[0])
        Z = model.unroll_states(torch.from_numpy(test_x).to(device), h0)
        Z = Z.detach().cpu().numpy()

    # compute Koopman operator
    kann_ = kann.KANN(Z, emb='TruncatedSVD')
    C = kann_.compute_KOP()
    D, V = np.linalg.eig(C)

    # sort eigenvalues
    eig_idx = np.argsort(np.abs(D))[-4:]
    eig_idx = eig_idx[::-1]

    # project states onto eigenvectors
    Zpv = kann_.Zp @ V
    Zpvn = np.abs(Zpv)

    return Zpvn, eig_idx


class Model(nn.Module):
    def __init__(self, embed_dim, hidden_size, output_dim, args):
        super(Model, self).__init__()

        self.hidden_dim = hidden_size
        self.output_dim = output_dim
        self.n_layers = args.num_layers
        self.vocab_size = args.vocab_size
        self.device = args.device

        self.embedding = nn.Embedding(args.vocab_size, embed_dim)
        self.rnn = nn.GRUCell(input_size=embed_dim, hidden_size=self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

    def forward(self, x, hidden, seq_len):

        batch_size = x.size(0)

        embeds = self.embedding(x)
        outputs = []

        for input_ in torch.unbind(embeds, dim=1):
            hidden = self.rnn(input_, hidden)
            outputs.append(self.linear(hidden))

        out = []
        for i, idx in enumerate(seq_len):
            out.append(outputs[idx-1][i])

        out = torch.stack(out)

        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        return torch.zeros(batch_size, self.hidden_dim).to(self.device)

    def unroll_states(self, x, hidden):

        embeds = self.embedding(x)
        outputs = []

        for input_ in torch.unbind(embeds, dim=1):
            hidden = self.rnn(input_, hidden)
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def readouts(self, last_hidden):

        out = self.linear(last_hidden)
        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(last_hidden.shape[0], -1)
        return sig_out[:, -1]

    def loss(self, logits, labels):
        return self.loss_func(logits, labels)

    def accuracy(self, pred, label):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == label.squeeze()).item() / float(len(pred))


def create_opt(model, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=args.decay_rate)

    return opt, lr_sched
