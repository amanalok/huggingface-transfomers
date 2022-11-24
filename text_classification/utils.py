import random
from umap import UMAP
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

from constants import PRE_TRAINED_MODEL_NAME


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}


def get_pretrained_model_with_classification_head(device):
    model_ckpt = PRE_TRAINED_MODEL_NAME
    num_classes = 6
    model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_classes).to(device))
    return model


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title('Normalized Confusion Matrix')
    plt.show()


def _extract_hidden_states(batch, device, tokenizer, model):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return {'hidden_state': last_hidden_state[:, 0].cpu().numpy()}


def get_numerical_representations(emotions_encoded_dataset, device, tokenizer, model, get_info=False):
    emotions_encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    emotions_hidden = emotions_encoded_dataset.map(_extract_hidden_states, 
                                                    batched=True, 
                                                    fn_kwargs={
                                                        'device': device,
                                                        'tokenizer': tokenizer,
                                                        'model': model
                                                    })
    
    if get_info:
        print(emotions_hidden)
        print(emotions_hidden['train'].column_names)

    return emotions_hidden


def get_pre_trained_model(device, sample_tokenized_data, get_info=False):
    model_ckpt = PRE_TRAINED_MODEL_NAME
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    if get_info:
        inputs = {k: v.to(device) for k, v in sample_tokenized_data.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        print(outputs)
        print(outputs)

    return model


def _tokenize(batch, tokenizer):
    return tokenizer(batch['text'], padding=True, truncation=True)


def get_tokenizer():
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    return tokenizer


def tokenize_emotions_dataset(emotions_dataset, tokenizer, get_info=False):
    emotions_dataset_encoded = emotions_dataset.map(_tokenize, batched=True, batch_size=None, fn_kwargs={'tokenizer': tokenizer})
    inputs = None
    if get_info:
        print(emotions_dataset_encoded)
        print(emotions_dataset_encoded['train'].column_names)
        for key in ['text', 'label', 'input_ids', 'attention_mask']:
            print(emotions_dataset_encoded['train'][key][0])
        
        text = "this is a test"
        inputs = tokenizer(text, return_tensors='pt')
        print(f'Input Text: {text}')
        print(f'Tokenized Output Shape: {inputs["input_ids"].size()}')
        print(inputs)
    
    return emotions_dataset_encoded, inputs


def get_emotions_dataset(get_info=False):
    emotions = load_dataset('emotion')
    if get_info:
        print(emotions)

    train_ds = emotions['train']
    train_ds_size = len(train_ds)
    if get_info:
        print(f'Size of training dataset: {train_ds_size}')
        random_index = random.randrange(0, train_ds_size)
        print(f'Below is an example of a training instance:\n {train_ds[random_index]}')
        print(f'Dataset Details:\n {train_ds.features}')
        print(f'First 5 tweets in the dataset:\n {train_ds["text"][:5]}')

    return emotions


def get_embedding_visualization(
    X: np.ndarray,
    y: np.ndarray,
    labels: list
) -> None:

    X_scaled = MinMaxScaler().fit_transform(X, y)
    mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)

    df_emb = pd.DataFrame(mapper.embedding_, columns=['X', 'Y'])
    df_emb['label'] = y

    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    axes = axes.flatten()

    cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f'label == {i}')
        axes[i].hexbin(df_emb_sub['X'], df_emb_sub['Y'], cmap=cmap, gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()