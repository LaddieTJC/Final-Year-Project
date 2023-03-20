from sqlite3 import DataError
from soupsieve import select
from transformers import DistilBertTokenizer, AlbertTokenizer,AutoTokenizer,AutoModelForSeq2SeqLM
import torch
from models.model_builder import ExtSummarizer, HiWestSummarizer
from pathlib import Path
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np
import os
import pandas as pd
import math

BASE_DIR = Path(__file__).resolve().parent.parent

def softmax_t(x,t=1):
    exp_sum = []
    softmax = []
    for i in x:
        exp_sum.append(math.exp(i/t))
    denom = sum(exp_sum)
    for i in x:
        softmax.append(round(math.exp(i/t)/denom,7))
    return softmax

def summarize(input_data,cumudist,num_sents=3, model='hiwest', device='cpu'):
  
    if model == 'hiwestdistil':     
        ## TODO: Add other tokenizer and models
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
        model,t5_model,t5_tokenizer = load_model('hiwestdistilbert')
    elif model == 'bertsum':
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
        model,t5_model,t5_tokenizer = load_model('bertsum')
    else:
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
        model,t5_model,t5_tokenizer  = load_model('hiwestalbert')
    # else:
    #     tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)


    input_data, sents = preprocess(input_data)
    source_data = process_text(input_data, tokenizer, device)
    selected_ids, sent_scores = get_scores_summary(model, source_data, num_sents, device=device)
    sentences = []
    try:
        assert(len(sents) == len(sent_scores[0]))
    except:
        raise DataError('Input format incorrect. Please check for proper sentence structures.')
    for i, text in enumerate(sents):
        sent = {
            "text": text,
            "scores": sent_scores[0][i],
            "index": i
        }
        sentences.append(sent)
    df = pd.DataFrame(data=sentences)
    df['softmax'] = softmax_t(df['scores'].to_list())
    df = df.sort_values(by=['softmax'],ignore_index=True,ascending=False)
    df['cumulative_dist'] = np.cumsum(df['softmax'])
    # df.to_csv('summary_score.csv')
    # return sentences, np.sort(selected_ids)
    if cumudist < df['cumulative_dist'][0]:
        df_output = df[df.index == 0]
        sents = ' '.join(df_output['text'].tolist())
        # sents = f"Top {count} sentences: \r\n {sents}"
    elif cumudist == 1:
        df_output = df
        sents = ' '.join(df_output['text'].tolist())
    else:
        df_output = df[df['cumulative_dist'] <= cumudist]
        sents = ' '.join(df_output['text'].tolist())
        # sents = f"Top {count} sentences: \r\n {sents}"
    t5_output = t5_summary(t5_model,t5_tokenizer,sents)
    return sents,t5_output,sentences

def t5_summary(model,tokenizer,k_sents):
    content = ["summarize:" + k_sents]
    content = tokenizer(content, max_length=512, truncation=True, return_tensors="pt")
    content = model.generate(**content, num_beams=8, do_sample=True, min_length=10, max_length=64)
    final_summary = tokenizer.batch_decode(content, skip_special_tokens=True)[0]
    return final_summary


def load_model(model='bertsum', device='cpu'):
    print(f'Loading model-{model}... ')
    t5_model_name = "checkpoint-5600"
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
    print("Checkpoint loaded.")
    if model == 'bertsum':
        checkpoint = torch.load(BASE_DIR / 'checkpoints/bertsum.pt', map_location='cpu')["model"]
        model = ExtSummarizer(device=device, checkpoint=checkpoint, bert_type='distilbert').to(device)
    elif model == 'hiwestalbert':
        checkpoint = torch.load(BASE_DIR / 'checkpoints/hiwest.pt', map_location='cpu')["model"]
        model = HiWestSummarizer(device=device, checkpoint=checkpoint, bert_type='albert').to(device)
    else:
        checkpoint = torch.load(BASE_DIR / 'checkpoints/hiwest_distil.pt', map_location='cpu')["model"]
        model = HiWestSummarizer(device=device, checkpoint=checkpoint, bert_type='distilbert').to(device)
    return model ,t5_model,t5_tokenizer


def preprocess(input_data):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    raw_text = input_data.replace("\n", " ").replace("[CLS] [SEP]", " ")
    sents = sent_tokenize(raw_text)
    processed_text = "[CLS] [SEP]".join(sents)
    return processed_text, sents

def process_text(processed_text, tokenizer, device='cpu', max_pos=512):
    print(f'Processing text... ')
    if tokenizer.name_or_path == 'albert-base-v2':
        tokenizer.vocab = tokenizer.get_vocab()
        sep_vid = tokenizer.vocab["[SEP]"]
        cls_vid = tokenizer.vocab["[CLS]"]
    else:
        sep_vid = tokenizer.vocab["[SEP]"]
        cls_vid = tokenizer.vocab["[CLS]"]

    print(sep_vid)
    print(cls_vid)


    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, mask_src, segments_ids, clss, mask_cls

    src, mask_src, segments_ids, clss, mask_cls = _process_src(processed_text)
    segs = torch.tensor(segments_ids)[None, :].to(device)
    src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]
    return src, mask_src, segs, clss, mask_cls, src_text


def get_scores_summary(model, input_data, max_length=2, device='cpu'):
    print("Generating scores...")
    with torch.no_grad():
        src, mask, segs, clss, mask_cls, src_str = input_data
        sent_scores, mask = model(src, segs, clss, mask, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)

        return selected_ids[0], sent_scores

def try_ftp():
    print("abc")
