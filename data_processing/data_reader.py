

class DataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []

    def read_data(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence = [[],[],[]]
            for line in f:
                splited_words = line.replace('_','').split()
                if len(splited_words) > 1:
                    word = splited_words[1]
                    word_idx = self.word_dict[word]
                    cur_sentence[0].append(word)
                    pos = splited_words[2]
                    pos_index = self.pos_dict[pos]
                    cur_sentence[1].append(pos)
                    edge = (int (splited_words[3]),int(splited_words[0]))
                    cur_sentence[2].append(edge)
                else:
                    if len (cur_sentence[0])>100:
                        print (len (cur_sentence[0]))
                        print (cur_sentence[0])
                    self.sentences.append(cur_sentence)
                    cur_sentence = [[],[],[]]

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)