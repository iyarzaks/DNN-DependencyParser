import torch
import torch.nn as nn
import numpy as np
from chu_liu_edmonds import decode_mst
from itertools import permutations, product


class DependencyParser(nn.Module):
    def __init__(self, w_vocab_size, w_emb_dim, w_indx_counter, w2i, pos_vocab_size, pos_emb_dim, n_lstm_layers,
                 mlp_hid_dim, lbl_mlp_hid_dim, n_labels, loss_f='NLL', ex_w_emb=None, with_labels=False):
        super(DependencyParser, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.w_indx_counter = w_indx_counter
        self.w2i = w2i
        self.ex_emb_flag = False
        self.labels_flag = with_labels

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
        self.hidden_dim = self.input_dim

        # Bidirectional LSTM model initialization
        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.n_lstm_layers,
                               bidirectional=True,
                               batch_first=True)

        # Edge scorer initialization
        self.edge_scorer = MLP(input_size=(4 * self.hidden_dim),
                               hidden_size=mlp_hid_dim)

        # labels MLP initialization
        if self.labels_flag:
            self.labels_mlp = MLP(input_size=(4 * self.hidden_dim),
                                  hidden_size=lbl_mlp_hid_dim,
                                  n_labels=n_labels,
                                  for_labels=True)

        # Chu-Liu-Edmonds decoder
        self.decoder = decode_mst

        # Define loss function
        if loss_f == 'NLL':
            if self.labels_flag:
                self.loss = nll_loss_edg_lbl
            else:
                self.loss = nll_loss
        else:
            self.loss = loss_aug_inf

    def forward(self, sentence, word_dropout=False):

        # Decompose the input
        if self.labels_flag:
            word_indx_tensor, pos_indx_tensor, true_tree_heads, true_edges_labels = sentence
        else:
            word_indx_tensor, pos_indx_tensor, true_tree_heads = sentence
        n_words = word_indx_tensor.shape[1]

        # Word dropout
        if word_dropout:  # todo: test
            for cell_indx, word_indx in enumerate(word_indx_tensor):
                unk_prob = 0.25 / (self.w_indx_counter[word_indx] + 0.25)
                bernoulli_rv = np.random.binomial(1, unk_prob, 1)
                if bernoulli_rv:
                    word_idx_tensor[cell_indx] = self.w2i['<unk>']

        # Word & POS embedding
        word_emb_tensor = self.word_embedding(word_indx_tensor.to(self.device))
        pos_emb_tensor = self.pos_embedding(pos_indx_tensor.to(self.device))
        # Embeddings concatenation
        if self.ex_emb_flag:  # todo: test
            ex_word_em_tensor = self.ex_word_embedding(word_indx_tensor)
            input_vectors = torch.cat((word_emb_tensor, ex_word_em_tensor, pos_emb_tensor), dim=-1)
        else:
            input_vectors = torch.cat((word_emb_tensor, pos_emb_tensor), dim=-1)

        hidden_vectors, _ = self.encoder(input_vectors)

        # Create 'edge vectors' by concatenating couples of hidden vectors
        edges_map, true_edges_map = {}, {}
        true_indx = 0
        valid_edges = [(h, m) for m in range(1, n_words) for h in range(0, n_words) if h != m]
        heads, mods, true_edges_indx = [], [], []

        for indx, (h, m) in enumerate(valid_edges):
            heads.append(h)
            mods.append(m)
            edges_map[(h, m)] = indx

            if true_tree_heads[m] == h:
                true_edges_indx.append(indx)
                true_edges_map[(h, m)] = true_indx
                true_indx += 1

        edges_tensor = torch.cat((hidden_vectors[0, heads, :], hidden_vectors[0, mods, :]), dim=-1)
        true_edges_tensor = edges_tensor[true_edges_indx, :]

        # Activate the scorer
        scores_tensor = self.edge_scorer(edges_tensor)
        if self.labels_flag:
            l_softmax_tensor = self.labels_mlp(true_edges_tensor)

        # Represent the scores as a 2-dimensional numpy array
        scores_np_matrix = np.zeros((n_words, n_words))
        for (h, m) in edges_map.keys():
            scores_np_matrix[h][m] = scores_tensor[edges_map[(h, m)]].data[0]

        # Prediction & loss calculation
        predicted_tree = decode_mst(scores_np_matrix, n_words, has_labels=False)

        if self.labels_flag:
            loss = self.loss(true_tree_heads, scores_tensor, edges_map,
                             true_edges_labels, l_softmax_tensor, true_edges_map)
        else:
            loss = self.loss(true_tree_heads, scores_tensor, edges_map)
        return loss, predicted_tree[0]


def nll_loss(true_tree, scores_tensor, edges_map):
    scores_tensor = torch.exp(scores_tensor)
    n_edges = true_tree.shape[0] - 1

    loss = 0
    for true_m, true_h in enumerate(true_tree[1:], 1):
        start_indx = (true_m - 1) * n_edges
        loss += torch.log(scores_tensor[edges_map[(true_h, true_m)]]) \
                - torch.log(torch.sum(scores_tensor[start_indx: start_indx + n_edges]))

    return -(1 / n_edges) * loss


def nll_loss_edg_lbl(true_tree, scores_tensor, edges_map, true_labels, softmax_tensor, gt_edges_map):
    scores_tensor = torch.exp(scores_tensor)
    n_edges = true_tree.shape[0] - 1
    structure_loss = 0
    labeling_loss = 0
    labels_indx = 0
    for true_m, true_h in enumerate(true_tree[1:], 1):
        labeling_loss += torch.log(softmax_tensor[gt_edges_map[(true_h, true_m)]][true_labels[labels_indx]])
        labels_indx += 1
        denominator = 0
        for j in range(n_edges):
            if j != true_m:
                denominator += scores_tensor[edges_map[(j, true_m)]]
        structure_loss += torch.log(scores_tensor[edges_map[(true_h, true_m)]] / denominator)
    loss = structure_loss + labeling_loss
    return -(1 / n_edges) * loss


def loss_aug_inf(true_tree, scores_tensor, edges_map):
    # todo: make this method more efficient and generic
    true_edges = set([(true_h, true_m) for true_m, true_h in enumerate(true_tree[1:], 1)])
    n_edges = len(true_edges)
    n_words = n_edges + 1
    fine = 1

    # Populate score matrix - add a constant for edges that aren't part of the true tree
    scores_np_matrix = np.zeros((n_words, n_words))
    for (h, m) in edges_map.keys():
        if (h, m) in true_edges:
            scores_np_matrix[h][m] = scores_tensor[edges_map[(h, m)]].data[0]
        else:
            scores_np_matrix[h][m] = scores_tensor[edges_map[(h, m)]].data[0] + fine

    # Get the maximum spanning tree
    predicted_tree = decode_mst(scores_np_matrix, n_words, has_labels=False)[0]

    # Fill a tensor with the predicted tree scores
    pred_scores = torch.empty(n_edges, requires_grad=True)
    for pred_m, pred_h in enumerate(predicted_tree[1:], 1):
        if (pred_h, pred_m) not in true_edges:
            pred_scores[pred_m - 1] = scores_tensor[edges_map[(pred_h, pred_m)]] + fine
        else:
            pred_scores[pred_m - 1] = scores_tensor[edges_map[(pred_h, pred_m)]]

    # Fill a tensor with the true tree scores
    true_scores = torch.empty(n_edges, requires_grad=True)
    for true_m, true_h in enumerate(true_tree[1:], 1):
        true_scores[true_m - 1] = scores_tensor[edges_map[(true_h, true_m)]]

    # Loss calculation
    loss = torch.max(torch.tensor([0, 1 + torch.sum(true_scores) - torch.sum(pred_scores)], requires_grad=True))

    return -1 * loss  # todo: maybe we should multiply the loss by -1


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_labels=None, for_labels=False):
        super(MLP, self).__init__()

        # define MLP for labels flag
        self.labels_flag = for_labels

        # initialize MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = torch.relu
        if self.labels_flag and n_labels:
            self.fc2 = nn.Linear(hidden_size, n_labels)
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, edges):
        x = self.fc1(edges)
        x = self.activation(x)
        output = self.fc2(x)

        if self.labels_flag:
            output = self.softmax(output)

        return output


if __name__ == '__main__':
    word_vocab_size = 2000
    word_embedding_dim = 100
    pos_vocab_size = 50
    pos_embedding_dim = 25
    bilstm_n_layers = 2
    mlp_hidden_dim = 100
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
                              n_labels=10,
                              with_labels=False)

    word_idx_tensor = torch.randint(0, word_vocab_size, (n_nodes,))
    word_idx_tensor = word_idx_tensor.unsqueeze(0)
    pos_idx_tensor = torch.randint(0, pos_vocab_size, (n_nodes,))
    pos_idx_tensor = pos_idx_tensor.unsqueeze(0)
    true_tree_heads = np.array([-1, 2, 0, 2, 0])
    true_labels = np.array([3, 6, 0, 4])

    # sentence = (word_idx_tensor, pos_idx_tensor, true_tree_heads, true_labels)
    sentence = (word_idx_tensor, pos_idx_tensor, true_tree_heads)
    loss, predicted_tree = parser(sentence)

    print('predicted tree: ', predicted_tree)
    print('loss: ', loss)
    loss.backward()
