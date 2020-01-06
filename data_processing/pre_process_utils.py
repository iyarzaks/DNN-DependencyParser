

class PreProcessUtils:

    @staticmethod
    def get_vocabs(file_path):
        print (file_path)
        word_dict = {}
        word_idx = 0
        pos_dict = {}
        pos_idx = 0
        with open(file_path, 'r') as f:
            for line in f:
                splited_words = line.replace('_','').split()
                if len(splited_words) > 1:
                    word = splited_words[1]
                    pos = splited_words[2]
                    if word not in word_dict:
                        word_dict[word] = word_idx
                        word_idx += 1
                    if pos not in pos_dict:
                        pos_dict[pos] = pos_idx
                        pos_idx += 1
        return word_dict, pos_dict
