# CUDA_LAUNCH_BLOCKING=1
from architecture import DependencyParser
from data_processing.pre_process_utils import PreProcessUtils
import torch
from data_processing.dependency_dataset import DependencyDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import datetime
np.set_printoptions(precision = 3, suppress = True)
HID_DIM = 20

# Todo wrap function for grid search, save results and best model.

LSTM_LAYERS = 2
TRAIN_FILE = "train.labeled"
TEST_FILE = "test.labeled"
EPOCHS = 5
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 100
HIDDEN_DIM = 10

def evaluate(model, test_dataloader):
    acc = 0
    loss = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, input_data in enumerate(test_dataloader):
            words_idx_tensor, pos_idx_tensor, sentence_length, edges = input_data
            edges = edges.cpu().numpy()[0]
            loss, predicted_tree = model((words_idx_tensor, pos_idx_tensor, edges))
            loss += loss.item()
            acc += np.sum(predicted_tree[1:] == edges[1:])
        acc = acc / np.sum(test_dataloader.dataset.sentence_lens)
        loss = loss/len(test_dataloader.dataset)
    return acc, loss


# Training start
def train(model, optimizer, train_dataloader, test_dataloader, accumulate_grad_steps):
    print("Training Started")
    accuracy_list = []
    loss_list = []
    epochs = EPOCHS
    for epoch in range(epochs):
        start_time = time.time()
        tmp_point_time = start_time
        model.train()
        acc = 0  # to keep track of accuracy
        loss = 0  # To keep track of the loss value
        i = 0
        sentence_avarage_acc = 0
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor, pos_idx_tensor, sentence_length, edges = input_data
            edges = edges.cpu().numpy()[0]
            loss, predicted_tree = model((words_idx_tensor, pos_idx_tensor, edges))
            loss = loss / accumulate_grad_steps
            loss.backward()
            if i % accumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
                runtime = datetime.timedelta(seconds=time.time()-tmp_point_time)
                tmp_point_time = time.time()
                print("Epoch {} ,\t {} / {} , \tAccuracy: {}\t Time: {}".format(epoch + 1, i,
                                                                       len(train_dataloader.dataset),
                                                                       round((sentence_avarage_acc / accumulate_grad_steps),3),
                                                                                runtime)
                      )
                sentence_avarage_acc = 0
            loss += loss.item()
            cur_acc = predicted_tree[1:] == edges[1:]
            acc += np.sum(cur_acc)
            sentence_avarage_acc += (np.sum(cur_acc) / len(edges))

        loss = loss / len(train_dataloader.dataset)
        acc = acc / np.sum(train_dataloader.dataset.sentence_lens)
        loss_list.append(float(loss))
        accuracy_list.append(float(acc))
        test_acc, test_loss = evaluate(model, test_dataloader)
        epoch_runtime = datetime.timedelta(seconds=time.time() - start_time)
        print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {} \t Test Loss: {} \t Time: {}".format(epoch + 1,
                                                                                      loss.cpu().detach().numpy()[0],5,
                                                                                      round(acc,3),
                                                                                      round(test_acc,3),
                                                                                      test_loss.cpu().detach().numpy()[0],5),
                                                                                                                   epoch_runtime)


def main():
    word_vocab, pos_vocab = PreProcessUtils.get_vocabs([TRAIN_FILE, TEST_FILE])
    train_dataset = DependencyDataset(word_vocab, pos_vocab, TRAIN_FILE, 'train', padding=False, word_embeddings=None)
    test_dataset = DependencyDataset(word_vocab, pos_vocab, TEST_FILE, 'test', padding=False, word_embeddings=None)
    w_indx_counter = train_dataset.get_word_idx_counter()
    w_vocab_size = len(word_vocab)
    pos_vocab_size = len(pos_vocab)
    model = DependencyParser(w_vocab_size, WORD_EMBEDDING_DIM, w_indx_counter, train_dataset.word_dict, pos_vocab_size,
                             POS_EMBEDDING_DIM, LSTM_LAYERS, HID_DIM, loss_f='NLL', ex_w_emb=None)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("working on gpu")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    accumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    train(model, optimizer, train_dataloader, test_dataloader, accumulate_grad_steps)


if __name__ == '__main__':
    main()