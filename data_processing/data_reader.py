from collections import defaultdict
UNKNOWN_TOKEN = "<unk>"
class DataReader:
    def __init__(self, file, word_dict, pos_dict, label_flag = False, subset='train'):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.word_idx_counter = defaultdict(int)
        self.label_flag = label_flag
        self.subset = subset

    def read_data(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence = [[],[],[]]
            if self.label_flag:
                cur_sentence = [[], [], [], []]
            for line in f:
                splited_words = line.replace('_','').split()
                if len(splited_words) > 1:
                    if self.subset == 'comp':
                        word = splited_words[1]
                        cur_sentence[0].append(word)
                        pos = splited_words[2]
                        cur_sentence[1].append(pos)
                        edge = 0
                        cur_sentence[2].append(edge)
                        if self.label_flag:
                            edge_label = 0
                            cur_sentence[3].append(edge_label)
                    else:
                        word = splited_words[1]
                        word_idx = self.word_dict[word]
                        self.word_idx_counter[word_idx] += 1
                        cur_sentence[0].append(word)
                        pos = splited_words[2]
                        pos_index = self.pos_dict[pos]
                        cur_sentence[1].append(pos)
                        edge = (int(splited_words[3]))
                        cur_sentence[2].append(edge)
                        if self.label_flag:
                            edge_label = splited_words[4]
                            cur_sentence[3].append(edge_label)
                else:
                    self.sentences.append(cur_sentence)
                    cur_sentence = [[],[],[]]
                    if self.label_flag:
                        cur_sentence = [[], [], [], []]

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)