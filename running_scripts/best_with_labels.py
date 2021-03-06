import train_utils

def main():
    epochs = 15
    conf_dic = {"WORD_EMBEDDINGS": None,
                "WORD_EMBEDDING_DIM": 100,
                "POS_EMBEDDING_DIM": 25,
                "HID_DIM_MLP": 100,
                "LSTM_LAYERS": 2,
                "LOSS": "NLL",
                "LEARNING_RATE": 0.01,
                "ACCUMULATE_GRAD_STEPS": 50,
                "LABELS_FLAG": True,
                "WORD_DROP": True
                }
    train_utils.run_test(configuration_dict=conf_dic, unique_id="labels_15_epochs", save_model=True, epochs=epochs)

if __name__ == '__main__':
    main()

