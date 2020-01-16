from collections import defaultdict

class DataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.word_idx_counter = defaultdict(int)

    def read_data(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence = [[],[],[]]
            for line in f:
                splited_words = line.replace('_','').split()
                if len(splited_words) > 1:
                    word = splited_words[1]
                    word_idx = self.word_dict[word]
                    self.word_idx_counter[word_idx] += 1
                    cur_sentence[0].append(word)
                    pos = splited_words[2]
                    pos_index = self.pos_dict[pos]
                    cur_sentence[1].append(pos)
                    edge = (int(splited_words[3]))
                    cur_sentence[2].append(edge)
                else:
                    self.sentences.append(cur_sentence)
                    cur_sentence = [[],[],[]]

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)