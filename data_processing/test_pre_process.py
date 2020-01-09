from data_processing.pre_process_utils import PreProcessUtils
from data_processing.data_reader import DataReader
from data_processing.dependency_dataset import DependencyDataset

def test_get_vocabs():
    word_vocab, pos_vocab =PreProcessUtils.get_vocabs("train.labeled")
    print (word_vocab)

def test_read_data():
    word_vocab, pos_vocab = PreProcessUtils.get_vocabs("train.labeled")
    data_reader = DataReader(file='train.labeled', word_dict=word_vocab, pos_dict=pos_vocab)
    data_reader.read_data()


def test_partial_dataset():
    word_vocab, pos_vocab = PreProcessUtils.get_vocabs("train.labeled")
    dataset  = DependencyDataset(word_vocab, pos_vocab, "train.labeled", 'train', padding=True, word_embeddings=None)
    for words,pos,len,edges in dataset:
        print (words)

