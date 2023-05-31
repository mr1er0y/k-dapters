"""Full training script"""
import argparse
import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp.autocast_mode as autocast_mode
import torch.cuda.amp.grad_scaler as grad_scaler
import torch.utils.data
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW

from model import AdapterRobertaClassifier


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe["abstract"]
        self.labels = dataframe["labels"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train(epoch, model, optimizer, scheduler, loss_function, mode, dataloader, tokenizer, cuda, scaler):
    '''The training function for RobertaClassifier and AdapterRobertaClassifier.'''
    if mode == 'train':
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    losses = []
    # for i in range(0, len(data), batch_size):
    for batch_features, batch_labels in dataloader:
        if mode == 'train':
            optimizer.zero_grad()

        # input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in batch_features], batch_first=True,
        #                           padding_value=1)
        # masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in batch_features], batch_first=True,
        #                      padding_value=0)
        # labels = torch.LongTensor([item['label'] for item in batch_labels])
        encoding = tokenizer.encode_plus(
            batch_features,
            max_length=512,  # Максимальная длина последовательности
            padding='max_length',  # Добавление паддинга до максимальной длины
            truncation=True,  # Обрезка текста, если он превышает максимальную длину
            return_tensors='pt'  # Возвращение тензоров PyTorch
        )
        input_data = encoding["input_data"]
        masks = encoding["masks"]
        labels = batch_labels
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
        with autocast_mode.autocast():
            outputs = model(input_data, masks)
            # outputs, con_loss = model(input_data, masks, labels)
            loss = loss_function(outputs, labels)
            # loss = loss_function(outputs, labels) + 0.8*con_loss
        if mode == 'train':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        ground_truth += labels.cpu().numpy().tolist()
        # predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        predicts += (torch.sigmoid(outputs) > 0.5).cpu().numpy().tolist()
        losses.append(loss.item())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average='weighted') * 100, 2)
    micro_f1 = round(f1_score(ground_truth, predicts, average='micro') * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average='macro') * 100, 2)
    if mode == 'train':
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                           micro_f1,
                                                                                           macro_f1))
    if mode == 'dev':
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                         micro_f1,
                                                                                         macro_f1))
    if mode == 'test':
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                          micro_f1,
                                                                                          macro_f1))
        print(f1_score(ground_truth, predicts, average=None))


def main(CUDA: bool, LR: float, SEED: int, DATASET: str, BATCH_SIZE: int, model_checkpoint: str,
         speaker_mode: str, num_past_utterances: int, num_future_utterances: int,
         NUM_TRAIN_EPOCHS: int, WEIGHT_DECAY: float, WARMUP_RATIO: float, **kwargs):
    lr = float(LR)
    path = "drive/MyDrive/Data_arXiv/filtered_arxiv_db.csv"

    '''Load data'''
    df = pd.read_csv(path)
    df[['created_date', 'update_date']] = df[['created_date', 'update_date']].apply(pd.to_datetime)
    df = df.drop(['versions', 'description', 'new_category', 'sub_category'], axis=1)
    df.columns = ['id', 'title', 'authors', 'category', 'published_date', 'updated_date', 'abstract']
    df["category"] = df["category"].apply(eval)

    # Extract the categories column as a list of lists
    categories = []
    for el in df["category"]:
        categories.extend(el)
    categories = np.unique(categories)
    NUM_CLASS = len(categories)

    # Initialize the MultiLabelBinarizer and fit_transform the categories
    mlb = MultiLabelBinarizer()
    df['labels'] = mlb.fit_transform(df["category"].values).tolist()

    df_train, df_test_val = train_test_split(df, test_size=0.3, random_state=SEED)
    df_test, df_val = train_test_split(df_test_val, test_size=0.5, random_state=SEED)

    dataset_train = CustomDataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)

    dataset_test = CustomDataset(df_test)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE)

    dataset_val = CustomDataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

    # model = VADRobertaClassifier(model_checkpoint, 768, NUM_CLASS)
    # model = RobertaClassifier(model_checkpoint, NUM_CLASS)
    # model = PrefixRobertaClassifier(model_checkpoint, 1024, NUM_CLASS)
    model = AdapterRobertaClassifier(args, NUM_CLASS)
    # model = AdapterLDARobertaClassifier(args, NUM_CLASS)
    # model = ConRobertaClassifier(args, NUM_CLASS)
    # predicter = PredcitVADandClassfromLogit(args, label_type='single', label_VAD=label_VAD)
    predicter = None

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    '''Use linear scheduler.'''
    total_steps = float(10 * len(df_train)) / BATCH_SIZE
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), math.ceil(total_steps))
    loss_function = nn.BCELoss()

    '''Due to the limitation of computational resources, we use mixed floating point precision.'''
    scaler = grad_scaler.GradScaler()

    if CUDA:
        model.cuda()
    # random.shuffle(tr_data)
    for n in range(NUM_TRAIN_EPOCHS):
        train(n, model, optimizer, scheduler, loss_function, "train", dataloader_train, tokenizer, CUDA, scaler)
        train(n, model, optimizer, scheduler, loss_function, "dev", dataloader_val, tokenizer, CUDA, scaler)
        train(n, model, optimizer, scheduler, loss_function, "test", dataloader_test, tokenizer, CUDA, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='erc RoBERTa text huggingface training')
    parser.add_argument('--DATASET', type=str, default="IEMOCAP")
    parser.add_argument('--CUDA', type=bool, default=True)
    parser.add_argument('--model_checkpoint', type=str, default="roberta-linadapter")
    parser.add_argument('--speaker_mode', type=str, default="upper")
    parser.add_argument('--num_past_utterances', type=int, default=1000)
    parser.add_argument('--num_future_utterances', type=int, default=1000)
    parser.add_argument('--BATCH_SIZE', type=int, default=4)
    parser.add_argument('--LR', type=float, default=1e-5)
    parser.add_argument('--HP_ONLY_UPTO', type=int, default=10)
    parser.add_argument('--NUM_TRAIN_EPOCHS', type=int, default=10)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.01)
    parser.add_argument('--WARMUP_RATIO', type=float, default=0.2)
    parser.add_argument('--HP_N_TRIALS', type=int, default=5)
    parser.add_argument('--OUTPUT-DIR', type=str, default="./output")
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of original model.")
    parser.add_argument("--freeze_adapter", default=True, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument('--fusion_mode', type=str, default='add',
                        help='the fusion mode for bert feature and adapter feature |add|concat')

    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,23", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=0, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument('--meta_fac_adaptermodel', default="./pretrained_models/fac-adapter/pytorch_model.bin",
                        type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default="./pretrained_models/lin-adapter/pytorch_model.bin",
                        type=str, help='the pretrained linguistic adapter model')
    parser.add_argument('--alpha', default=0.8,
                        type=float, help='The loss coefficient.')

    args = parser.parse_args()
    args = vars(args)
    if "linadapter" in args['model_checkpoint']:
        args['meta_fac_adaptermodel'] = ''
    if "facadapter" in args['model_checkpoint']:
        args['meta_lin_adaptermodel'] = ''

    args['adapter_list'] = args['adapter_list'].split(',')
    args['adapter_list'] = [int(i) for i in args['adapter_list']]
    device = torch.device("cuda" if torch.cuda.is_available() and args['CUDA'] is True else "cpu")
    args['n_gpu'] = torch.cuda.device_count()
    args['device'] = device

    logging.info(f"arguments given to {__file__}: {args}")
    main(**args)
