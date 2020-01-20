UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
# ROOT_TOKEN = PAD_TOKEN # this can be used if you are not padding your batches
ROOT_TOKEN = "<root>" # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN, ROOT_TOKEN]

class PreProcessUtils:

    @staticmethod
    def get_vocabs(file_paths):
        word_dict = {}
        word_idx = 0
        pos_dict = {}
        pos_idx = 0
        label_dict = {}
        label_idx = 0
        for word in SPECIAL_TOKENS:
            word_dict[word] = word_idx
            word_idx += 1
            pos_dict[word] = pos_idx
            pos_idx += 1
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    splited_words = line.replace('_','').split()
                    if len(splited_words) > 1:
                        word = splited_words[1]
                        pos = splited_words[2]
                        label = splited_words[4]
                        if word not in word_dict:
                            word_dict[word] = word_idx
                            word_idx += 1
                        if pos not in pos_dict:
                            pos_dict[pos] = pos_idx
                            pos_idx += 1
                        if label not in label_dict:
                            label_dict[label] = label_idx
                            label_idx += 1
        return word_dict, pos_dict, label_dict
