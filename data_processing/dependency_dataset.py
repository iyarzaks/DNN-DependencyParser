import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset

from pathlib import Path
from collections import Counter
from data_processing.data_reader import DataReader
import numpy as np
ROOT_TOKEN = "<root>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [ROOT_TOKEN, UNK_TOKEN]

class DependencyDataset (Dataset):
    def __init__(self, word_dict, pos_dict, label_vocab, file_path: str, subset: str, padding=False, word_embeddings=None,
                 label_flag=False):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        self.file = file_path
        self.label_vocab = label_vocab
        self.label_flag = label_flag
        self.data_reader = DataReader(self.file, word_dict, pos_dict, label_flag=self.label_flag , subset=subset)
        self.data_reader.read_data()
        self.word_idx_counter = self.data_reader.word_idx_counter
        self.vocab_size = len(self.data_reader.word_dict)
        self.pos_dict = pos_dict
        self.word_dict = word_dict
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.data_reader.word_dict, word_embeddings)
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = [[],[],[]]
        self.sentence_lens = [len(sentence[0]) for sentence in self.data_reader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        if self.label_flag:
            word_embed_idx, pos_embed_idx, sentence_len, edges, labels = self.sentences_dataset[index]
            return word_embed_idx, pos_embed_idx, sentence_len, edges, labels
        word_embed_idx, pos_embed_idx, sentence_len , edges = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len , edges

    @staticmethod
    def init_word_embeddings(word_dict, embeding_type):
        if embeding_type is not None:
            external_embedding = Vocab(Counter(word_dict), vectors=embeding_type, specials=SPECIAL_TOKENS)
            return external_embedding.stoi, external_embedding.itos, external_embedding.vectors
        else:
            return [[],[],[]]

    def get_word_idx_counter(self):
        return self.word_idx_counter

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_edges_list = list()
        sentence_len_list = list()
        if self.label_flag:
            sentence_labels_list = list()
            for sentence_idx, sentence in enumerate(self.data_reader.sentences):
                words_idx_list = [self.word_dict[ROOT_TOKEN]]
                pos_idx_list = [self.pos_dict[ROOT_TOKEN]]
                edges_list = [-1]
                labels_list = []
                for word, pos, edge, label in zip(sentence[0], sentence[1], sentence[2], sentence[3]):
                    try:
                        words_idx_list.append(self.word_dict[word])
                    except:
                        words_idx_list.append(self.word_dict[UNK_TOKEN])
                    try:
                        pos_idx_list.append(self.pos_dict[pos])
                    except:
                        pos_idx_list.append(self.pos_dict[UNK_TOKEN])
                    edges_list.append(edge)
                    try:
                        labels_list.append(self.label_vocab[label])
                    except:
                        labels_list.append(0)
                sentence_len = len(words_idx_list)
                sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
                sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
                sentence_edges_list.append(np.array(edges_list))
                sentence_labels_list.append(np.array(labels_list))
                sentence_len_list.append(sentence_len)
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                         sentence_pos_idx_list,
                                                                         sentence_len_list,
                                                                         sentence_edges_list,
                                                                         sentence_labels_list
                                                                         ))}

        else:
            for sentence_idx, sentence in enumerate(self.data_reader.sentences):
                words_idx_list = [self.word_dict[ROOT_TOKEN]]
                pos_idx_list = [self.pos_dict[ROOT_TOKEN]]
                edges_list = [-1]
                for word, pos , edge in zip(sentence[0],sentence[1],sentence[2]):
                    try:
                        words_idx_list.append(self.word_dict[word])
                    except:
                        words_idx_list.append(self.word_dict[UNK_TOKEN])
                    try:
                        pos_idx_list.append(self.pos_dict[pos])
                    except:
                        pos_idx_list.append(self.pos_dict[UNK_TOKEN])
                    edges_list.append(edge)
                sentence_len = len(words_idx_list)
                sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
                sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
                sentence_edges_list.append(np.array(edges_list))
                sentence_len_list.append(sentence_len)
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                         sentence_pos_idx_list,
                                                                         sentence_len_list,
                                                                         sentence_edges_list,
                                                                         ))}
