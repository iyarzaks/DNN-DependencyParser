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
import json
import os
import itertools
np.set_printoptions(precision = 3, suppress = True)

TRAIN_FILE = "train.labeled"
TEST_FILE = "test.labeled"

# evaluate results on test data
def evaluate(model, test_dataloader, device):
    acc = 0
    loss = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, input_data in enumerate(test_dataloader):
            if model.labels_flag:
                words_idx_tensor, pos_idx_tensor, sentence_length, edges, labels = input_data
                edges = edges.cpu().numpy()[0]
                labels = labels.cpu().numpy()[0]
                words_idx_tensor = words_idx_tensor.to(device=device)
                pos_idx_tensor = pos_idx_tensor.to(device=device)
                loss, predicted_tree = model((words_idx_tensor, pos_idx_tensor, edges, labels))
            else:
                words_idx_tensor, pos_idx_tensor, sentence_length, edges = input_data
                edges = edges.cpu().numpy()[0]
                words_idx_tensor = words_idx_tensor.to(device=device)
                pos_idx_tensor = pos_idx_tensor.to(device=device)
                loss, predicted_tree = model((words_idx_tensor, pos_idx_tensor, edges))
            loss += loss.item()
            acc += np.sum(predicted_tree[1:] == edges[1:])
        acc = acc / np.sum(test_dataloader.dataset.sentence_lens)
        loss = loss/len(test_dataloader.dataset)
    return acc, loss


# train model in epochs
def train(model, optimizer, train_dataloader, test_dataloader, accumulate_grad_steps, epochs, labels_flag=False,
          word_dropout = False , device = "cpu"):
    accuracy_list = []
    loss_list = []
    test_loss_list = []
    test_accuracy_list = []
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
            if labels_flag:
                words_idx_tensor, pos_idx_tensor, sentence_length, edges, labels = input_data
                words_idx_tensor = words_idx_tensor.to(device=device)
                pos_idx_tensor = pos_idx_tensor.to(device=device)
                labels = labels.cpu().numpy()[0]
                edges = edges.cpu().numpy()[0]
                loss, predicted_tree = model((words_idx_tensor, pos_idx_tensor, edges, labels))
            else:
                words_idx_tensor, pos_idx_tensor, sentence_length, edges = input_data
                edges = edges.cpu().numpy()[0]
                words_idx_tensor = words_idx_tensor.to(device=device)
                pos_idx_tensor = pos_idx_tensor.to(device=device)
                loss, predicted_tree = model((words_idx_tensor, pos_idx_tensor, edges) , word_dropout=word_dropout)
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
            sentence_avarage_acc += (np.sum(cur_acc) / (len(edges)-1))
        loss = loss / len(train_dataloader.dataset)
        acc = acc / np.sum(train_dataloader.dataset.sentence_lens)
        loss_list.append(float(loss))
        accuracy_list.append(float(acc))
        test_acc, test_loss = evaluate(model, test_dataloader, device)
        test_loss_list.append(float(test_loss))
        test_accuracy_list.append(float(test_acc))
        epoch_runtime = datetime.timedelta(seconds=time.time() - start_time)
        print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {} \t Test Loss: {} \t Time: {}".format(epoch + 1,
                                                                                      loss.cpu().detach().numpy()[0],
                                                                                      round(acc,3),
                                                                                      round(test_acc,3),
                                                                                      test_loss.cpu().detach().numpy()[0],
                                                                                      epoch_runtime))
    return model, {"loss_list": loss_list,"accuracy_list": accuracy_list,
                       "test_loss_list": test_loss_list,"test_accuracy_list":test_accuracy_list}


# run specific configuration test
def run_test(configuration_dict, unique_id, save_model = False , epochs=1):
    print (f"Run test {unique_id}" )
    print (f"Test Parameters {configuration_dict}")
    labels_flag = configuration_dict['LABELS_FLAG']
    word_vocab, pos_vocab, label_vocab = PreProcessUtils.get_vocabs([TRAIN_FILE, TEST_FILE])
    w_vocab_size = len(word_vocab)
    pos_vocab_size = len(pos_vocab)
    train_dataset = DependencyDataset(word_vocab, pos_vocab, label_vocab, TRAIN_FILE, 'train', padding=False,
                                      word_embeddings=configuration_dict["WORD_EMBEDDINGS"], label_flag=labels_flag)
    test_dataset = DependencyDataset(word_vocab, pos_vocab, label_vocab, TEST_FILE, 'test', padding=False,
                                     word_embeddings=configuration_dict["WORD_EMBEDDINGS"], label_flag=labels_flag)
    w_indx_counter = train_dataset.get_word_idx_counter()
    word_embedding_vectors = None
    if configuration_dict["WORD_EMBEDDINGS"]:
        word_embedding_vectors = train_dataset.word_vectors
    model = DependencyParser(w_vocab_size, configuration_dict["WORD_EMBEDDING_DIM"], w_indx_counter,
                             train_dataset.word_dict, pos_vocab_size,
                             configuration_dict["POS_EMBEDDING_DIM"], configuration_dict["LSTM_LAYERS"],
                             configuration_dict["HID_DIM_MLP"], loss_f=configuration_dict["LOSS"],
                             ex_w_emb=word_embedding_vectors,with_labels=labels_flag,n_labels=len(label_vocab),
                             lbl_mlp_hid_dim=100)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("working on gpu")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=configuration_dict["LEARNING_RATE"])
    accumulate_grad_steps = configuration_dict["ACCUMULATE_GRAD_STEPS"]  # This is the actual batch_size, while we officially use batch_size=1
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    trained_model, results_dict = train(model, optimizer, train_dataloader, test_dataloader, accumulate_grad_steps,
                                        epochs=epochs, labels_flag=labels_flag,
                                        word_dropout=configuration_dict["WORD_DROP"],device=device)
    string_id = str(unique_id)
    os.mkdir(f"running_tests/{string_id}")
    if save_model:
        torch.save(trained_model, f"running_tests/{string_id}/trained_model")
    with open(f"running_tests/{string_id}/conf_dict", "w") as f:
        json.dump(configuration_dict,f)
    for file in results_dict:
        with open(f"running_tests/{string_id}/{file}", "w") as f:
            json.dump(results_dict[file], f)
    return max(results_dict['test_accuracy_list'])





def grid_search():
    best_result = 0
    word_embedings = [None]
    word_embed_dims = [100]
    pos_embed_dims = [25]
    hid_mlp_dims = [100]
    lstm_layers = [2]
    losses = ['NLL']
    learning_rates = [0.01]
    grad_steps = [50]
    labels_flag = [True]
    word_drops = [True]
    for i, params in enumerate(itertools.product(*[word_embedings,word_embed_dims,pos_embed_dims,
                                                         hid_mlp_dims,lstm_layers,losses,learning_rates,grad_steps,
                                                   labels_flag,word_drops])):
        we, wed, ped, hdm, ll, l, lr, gs, lf, wd = params
        conf_dic = {"WORD_EMBEDDINGS": we,
                    "WORD_EMBEDDING_DIM": wed,
                    "POS_EMBEDDING_DIM": ped,
                    "HID_DIM_MLP": hdm,
                    "LSTM_LAYERS": ll,
                    "LOSS": l,
                    "LEARNING_RATE": lr,
                    "ACCUMULATE_GRAD_STEPS": gs,
                    "LABELS_FLAG": lf,
                    "WORD_DROP": wd}
        test_acc = run_test(unique_id=i, configuration_dict=conf_dic)
        if test_acc > best_result:
            best_result = test_acc
            print(f"####################\n New best result: {best_result}, test id: {str(i)} \n#####################")

def main():
    grid_search()

if __name__ == '__main__':
    main()
