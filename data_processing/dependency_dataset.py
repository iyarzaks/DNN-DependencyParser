import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset

from pathlib import Path
from collections import Counter
from data_processing.data_reader import DataReader
import numpy as np
ROOT_TOKEN = "<root>"

class DependencyDataset (Dataset):
    def __init__(self, word_dict, pos_dict, file_path: str, subset: str, padding=False, word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        self.file = file_path
        self.data_reader = DataReader(self.file, word_dict, pos_dict)
        self.data_reader.read_data()
        self.word_idx_counter = self.data_reader.word_idx_counter
        self.vocab_size = len(self.data_reader.word_dict)
        self.pos_dict = pos_dict
        self.word_dict = word_dict
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.data_reader.word_dict)
        # self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.data_reader.pos_dict)
        # self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        # self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        # self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence[0]) for sentence in self.data_reader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len , edges = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len , edges

    @staticmethod
    def init_word_embeddings(word_dict):
        return [0,0,0]
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_idx_counter(self):
        return self.word_idx_counter




    # def get_word_embeddings(self):
    #     return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    # def init_pos_vocab(self, pos_dict):
        # idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        # pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}
        #
        # for i, pos in enumerate(sorted(pos_dict.keys())):
        #     # pos_idx_mappings[str(pos)] = int(i)
        #     pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
        #     idx_pos_mappings.append(str(pos))
        # print("idx_pos_mappings -", idx_pos_mappings)
        # print("pos_idx_mappings -", pos_idx_mappings)
        # return pos_idx_mappings, idx_pos_mappings

    # def get_pos_vocab(self):
    #     return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_edges_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.data_reader.sentences):
            words_idx_list = [self.word_dict[ROOT_TOKEN]]
            pos_idx_list = [self.pos_dict[ROOT_TOKEN]]
            edges_list = [-1]
            for word, pos , edge in zip(sentence[0],sentence[1],sentence[2]):
                words_idx_list.append(self.word_dict[word])
                pos_idx_list.append(self.pos_dict[pos])
                edges_list.append(edge)
            sentence_len = len(words_idx_list)
            if padding:
                while len(words_idx_list) < self.max_seq_len:
                    words_idx_list.append(self.word_dict[PAD_TOKEN])
                    pos_idx_list.append(self.pos_dict[PAD_TOKEN])
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_edges_list.append(np.array(edges_list))
            sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_len_list,
                                                                     sentence_edges_list
                                                                     ))}
