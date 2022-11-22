import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import pickle
import argparse

from constants import BASELINE_MODEL_PATH
from utils import get_embedding_visualization, get_emotions_dataset, tokenize_emotions_dataset, get_pre_trained_model, \
                  get_numerical_representations, get_tokenizer, plot_confusion_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tbm', '--train_baseline_model', action='store_true')

    return parser.parse_args()


def get_baseline_model(filename):
    baseline_model = pickle.load(open(filename, 'rb'))
    return baseline_model


def main(train_baseline_model):

    ''' random_dataset_size = 10000
    random_dataset_x = np.random.rand(random_dataset_size, 768)
    random_dataset_y = np.random.randint(6, size=(random_dataset_size, 1)) '''

    if not train_baseline_model and get_baseline_model(filename=BASELINE_MODEL_PATH):
        raise Exception("Sorry, could not find a model to load. Run the script with the flag '-tbm")

    # Step 1: Get dataset
    emotions = get_emotions_dataset(get_info=True)
    labels = emotions

    # Step 2: Get the tokenizer
    tokenizer = get_tokenizer()
    
    # Step 2: Tokenize dataset
    emotions_encoded, sample_data_encoded = tokenize_emotions_dataset(emotions_dataset=emotions,
                                                                      tokenizer=tokenizer,
                                                                      get_info=True)
    
    # Step 3: Fetch pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_trained_model = get_pre_trained_model(device=device, sample_tokenized_data=sample_data_encoded, get_info=True)
    
    # Step 4: Convert tokenized data to PyTorch format
    emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    print(emotions_encoded)
    
    # Step 5: Get hidden state numerical representations of the Emotions Dataset
    # Note: We get hidden state of the CLS token for each tweet. This acts as the aggregate representation of the sequence
    emotions_numerical_representations = get_numerical_representations(emotions_encoded_dataset=emotions_encoded,
                                                                       device=device,
                                                                       tokenizer=tokenizer,
                                                                       model=pre_trained_model,
                                                                       get_info=True)

    # Step 6: Create a Feature Matrix
    X_train = np.array(emotions_numerical_representations['train']['hidden_state'])
    X_valid = np.array(emotions_numerical_representations['validation']['hidden_state'])
    y_train = np.array(emotions_numerical_representations['train']['label'])
    y_valid = np.array(emotions_numerical_representations['validation']['label'])

    print(f'Training Data Shape: {X_train.shape}')
    print(f'Training Data Label Shape: {y_train.shape}')

    # Step 6: Visualize the training dataset numerical representations
    get_embedding_visualization(X_train, y_train, labels=emotions['train'].features['label'].names)

    # Step 7: Train a simple classifier as a baseline model
    if train_baseline_model:
        lr_clf = LogisticRegression(max_iter=300)
        lr_clf.fit(X_train, y_train)
        pickle.dump(lr_clf, open(BASELINE_MODEL_PATH, 'wb'))
    else:
        lr_clf = pickle.load(open(BASELINE_MODEL_PATH, 'rb'))
    
    print(lr_clf.score(X_valid, y_valid))

    # Step 8: Get Confusion Matrix on performance of the model
    y_preds = lr_clf.predict(X_valid)
    plot_confusion_matrix(y_preds, y_valid, labels)


if __name__ == '__main__':
    args = get_args()
    print(args.train_baseline_model)
    main(train_baseline_model=args.train_baseline_model)
    