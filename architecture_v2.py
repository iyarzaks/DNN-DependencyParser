import torch
import torch.nn as nn
import numpy as np
from chu_liu_edmonds import decode_mst


class DependencyParser(nn.Module):
    def __init__(self, w_vocab_size, w_emb_dim, w_indx_counter, w2i, pos_vocab_size, pos_emb_dim, n_lstm_layers,
                 mlp_hid_dim, lstm_hid_dim, lbl_mlp_hid_dim=1, n_labels=1, loss_f='NLL', ex_w_emb=None,
                 with_labels=False,
                 lstm_drop_prob=0.0, mlp_drop_prob=0.0):
        super(DependencyParser, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.w_indx_counter = w_indx_counter
        self.w2i = w2i
        self.ex_emb_flag = False
        self.labels_flag = with_labels
        self.ex_w_emb = ex_w_emb
        # Embedding layers initialization
        self.word_embedding = nn.Embedding(w_vocab_size, w_emb_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)
        if ex_w_emb is not None:  # Use external word embeddings
            self.ex_emb_flag = True
            self.ex_word_embedding = nn.Embedding.from_pretrained(ex_w_emb, freeze=False)

        # LSTM dimensions
        self.n_lstm_layers = n_lstm_layers
        if self.ex_emb_flag:
            self.ex_emb_dim = self.ex_w_emb.size(-1)
            self.input_dim = w_emb_dim + self.ex_emb_dim + pos_emb_dim
        else:
            self.input_dim = w_emb_dim + pos_emb_dim
        self.hidden_dim = lstm_hid_dim

        # Bidirectional LSTM model initialization
        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.n_lstm_layers,
                               bidirectional=True,
                               batch_first=True,
                               dropout=lstm_drop_prob)

        # Edge scorer initialization
        self.mlp_dep = MLP(input_size=(2 * self.hidden_dim),
                           hidden_size=mlp_hid_dim,
                           dropout_prob=mlp_drop_prob)
        self.mlp_head = MLP(input_size=(2 * self.hidden_dim),
                            hidden_size=mlp_hid_dim,
                            dropout_prob=mlp_drop_prob)

        # Attention weights initialization
        self.weights = nn.Parameter(torch.Tensor(mlp_hid_dim + 1, mlp_hid_dim))
        nn.init.xavier_normal_(self.weights)

        # Chu-Liu-Edmonds decoder
        self.decoder = decode_mst

        # Define loss function
        self.s_max = nn.Softmax(dim=0)
        self.loss = nll_loss

    def forward(self, sentence, word_dropout=False):

        # Decompose the input
        word_indx_tensor, pos_indx_tensor, true_tree_heads = sentence
        n_words = word_indx_tensor.shape[1]

        # Word dropout
        if word_dropout:
            for cell_indx, word_indx in enumerate(word_indx_tensor):
                unk_prob = 0.25 / (self.w_indx_counter[word_indx] + 0.25)
                bernoulli_rv = np.random.binomial(1, unk_prob, 1)
                if bernoulli_rv:
                    word_indx_tensor[cell_indx] = self.w2i['<unk>']

        # Word & POS embedding
        word_emb_tensor = self.word_embedding(word_indx_tensor.to(self.device))
        pos_emb_tensor = self.pos_embedding(pos_indx_tensor.to(self.device))
        # Embeddings concatenation
        if self.ex_emb_flag:
            ex_word_em_tensor = self.ex_word_embedding(word_indx_tensor)
            input_vectors = torch.cat((word_emb_tensor, ex_word_em_tensor, pos_emb_tensor), dim=-1)
        else:
            input_vectors = torch.cat((word_emb_tensor, pos_emb_tensor), dim=-1)

        hidden_vectors, _ = self.encoder(input_vectors)
        hidden_vectors = hidden_vectors.squeeze()

        heads_tensor = self.mlp_head(hidden_vectors)
        dep_tensor = self.mlp_dep(hidden_vectors)
        pad_dep_tensor = torch.cat((dep_tensor, torch.ones(dep_tensor.shape[0]).unsqueeze(1)), dim=1)

        scores = torch.matmul(torch.matmul(pad_dep_tensor, self.weights), heads_tensor.T).T

        # Prediction & loss calculation
        predicted_tree = decode_mst(scores.detach().numpy(), n_words, has_labels=False)

        scores_s_max = self.s_max(scores)
        loss = self.loss(true_tree_heads, scores_s_max)
        return loss, predicted_tree[0]


def nll_loss(true_tree, scores_s_max):
    n_edges = true_tree.shape[0] - 1
    loss = 0
    for true_m, true_h in enumerate(true_tree[1:], 1):
        loss += torch.log(scores_s_max[true_h, true_m])
    return -(1 / n_edges) * loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.0):
        super(MLP, self).__init__()

        # initialize MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, edges):
        x = self.fc1(edges)
        x = self.activation(x)
        output = self.dropout(x)
        return output


if __name__ == '__main__':
    word_vocab_size = 2000
    word_embedding_dim = 100
    pos_vocab_size = 50
    pos_embedding_dim = 100
    bilstm_n_layers = 3
    mlp_hidden_dim = 500
    lbl_mlp_hid_dim = 100
    n_nodes = 5

    parser = DependencyParser(word_vocab_size,
                              word_embedding_dim,
                              None,
                              None,
                              pos_vocab_size,
                              pos_embedding_dim,
                              bilstm_n_layers,
                              mlp_hidden_dim,
                              lbl_mlp_hid_dim,
                              lstm_drop_prob=1/3,
                              mlp_drop_prob=1/3)



    word_idx_tensor = torch.randint(0, word_vocab_size, (n_nodes,))
    word_idx_tensor = word_idx_tensor.unsqueeze(0)
    pos_idx_tensor = torch.randint(0, pos_vocab_size, (n_nodes,))
    pos_idx_tensor = pos_idx_tensor.unsqueeze(0)
    true_tree_heads = np.array([-1, 2, 0, 2, 0])
    true_labels = np.array([3, 6, 0, 4])

    #sentence = (word_idx_tensor, pos_idx_tensor, true_tree_heads, true_labels)
    sentence = (word_idx_tensor, pos_idx_tensor, true_tree_heads)
    loss, predicted_tree = parser(sentence)

    print('predicted tree: ', predicted_tree)
    print('loss: ', loss)
    loss.backward()
