from data_processing.pre_process_utils import PreProcessUtils
from data_processing.dependency_dataset import DependencyDataset
import torch
from torch.utils.data.dataloader import DataLoader
TRAIN_FILE = "train.labeled"
TEST_FILE = "test.labeled"
COMP_FILE = "comp.unlabeled"

def get_predictions(model_path):
    word_vocab, pos_vocab, label_vocab = PreProcessUtils.get_vocabs([TRAIN_FILE, TEST_FILE])
    comp_dataset = DependencyDataset(word_vocab, pos_vocab, label_vocab, COMP_FILE, 'comp', padding=False,
                                         word_embeddings=None, label_flag=False)
    trained_model = torch.load("running_tests/basic_model_3_epochs/trained_model")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("working on gpu")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        trained_model.cuda()
    comp_dataloader = DataLoader(comp_dataset, shuffle=False)
    predictions = []
    for batch_idx, input_data in enumerate(comp_dataloader):
        words_idx_tensor, pos_idx_tensor, sentence_length, edges = input_data
        edges = edges.cpu().numpy()[0]
        words_idx_tensor = words_idx_tensor.to(device=device)
        pos_idx_tensor = pos_idx_tensor.to(device=device)
        _, predicted_tree = trained_model((words_idx_tensor, pos_idx_tensor, edges), word_dropout=False)
        predictions.extend(predicted_tree[1:])
    return predictions

def fill_file(predictions, output_file):
    prediction_idx = 0
    new_lines = []
    with open(COMP_FILE, 'r') as f:
        file_str = f.read()
    for line in file_str.split('\n'):
        splited_words = line.split()
        if len(splited_words) > 1:
            splited_words[6] = str(predictions[prediction_idx])
            prediction_idx += 1
            new_line = " ".join(splited_words)
            new_lines.append(new_line)
        else:
            new_lines.append('\n')
    with open(output_file, "w") as f:
        f.write("\n".join(new_lines).replace('\n\n','\n'))


def main():
    output_file = "comp.output"
    model_path = "running_tests/basic_model_3_epochs/trained_model"
    predictions = get_predictions(model_path)
    fill_file(predictions, output_file)


if __name__ == '__main__':
    main()
