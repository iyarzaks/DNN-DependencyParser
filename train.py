# CUDA_LAUNCH_BLOCKING=1
from architecture import DependencyParser
from data_processing.pre_process_utils import PreProcessUtils
import torch
from data_processing.dependency_dataset import DependencyDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import numpy as np

HID_DIM = 20

LSTM_LAYERS = 2
TRAIN_FILE = "train.labeled"
TEST_FILE = "test.labeled"
EPOCHS = 5
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 100
HIDDEN_DIM = 1000
word_vocab, pos_vocab = PreProcessUtils.get_vocabs([TRAIN_FILE, TEST_FILE])

train_dataset = DependencyDataset(word_vocab, pos_vocab, TRAIN_FILE, 'train', padding=False, word_embeddings=None)
test_dataset = DependencyDataset(word_vocab, pos_vocab, TEST_FILE, 'test', padding=False, word_embeddings=None)

w_indx_counter = train_dataset.get_word_idx_counter()
w_vocab_size = len(word_vocab)+2
pos_vocab_size = len (pos_vocab)+2

model = DependencyParser(w_vocab_size, WORD_EMBEDDING_DIM, w_indx_counter, train_dataset.word_dict, pos_vocab_size,
                         POS_EMBEDDING_DIM, LSTM_LAYERS, HID_DIM, loss_f='NLL', ex_w_emb=None)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    model.cuda()

# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
loss_function = nn.NLLLoss()

# We will be using a simple SGD optimizer to minimize the loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

train_dataloader = DataLoader(train_dataset, shuffle=True)

test_dataloader = DataLoader(test_dataset, shuffle=False)

# Training start
print("Training Started")
accuracy_list = []
loss_list = []
epochs = EPOCHS
acc = 0  # to keep track of accuracy
loss = 0  # To keep track of the loss value
for epoch in range(epochs):
    i = 0
    for batch_idx, input_data in enumerate(train_dataloader):
        i += 1
        words_idx_tensor, pos_idx_tensor, sentence_length, edges = input_data
        edges = edges.cpu().numpy()[0]
        loss, predicted_tree = model((words_idx_tensor, pos_idx_tensor, edges))
        loss = loss / acumulate_grad_steps
        loss.backward()
        if i % acumulate_grad_steps == 0:
            optimizer.step()
            model.zero_grad()
        loss += loss.item()
        acc += np.sum(predicted_tree == edges)
    loss = loss / len(train_dataset)
    acc = acc / len(train_dataset.sentences_dataset)
    loss_list.append(float(loss))
    accuracy_list.append(float(acc))
    # test_acc = evaluate()
    # e_interval = i
    # print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
    #                                                                               np.mean(loss_list[-e_interval:]),
    #                                                                               np.mean(accuracy_list[-e_interval:]),
    #                                                                               test_acc))
