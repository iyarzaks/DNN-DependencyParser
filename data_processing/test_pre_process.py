from data_processing.pre_process_utils import PreProcessUtils
from data_processing.data_reader import DataReader

def test_get_vocabs():
    word_vocab, pos_vocab =PreProcessUtils.get_vocabs("train.labeled")
    print (word_vocab)

def test_read_data():
    word_vocab, pos_vocab = PreProcessUtils.get_vocabs("train.labeled")
    data_reader = DataReader(file='train.labeled', word_dict=word_vocab, pos_dict=pos_vocab)
    data_reader.read_data()
    print (data_reader.sentences)
