
from flask import Flask, redirect, request, render_template, redirect,url_for, flash,session
from flask_wtf import FlaskForm
from flask_wtf.file import  FileRequired
import torch

from wtforms import *
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote_plus as urlquote
from flask_login import LoginManager,login_user,current_user, login_required,logout_user
from summary import SummaryForm
from PyPDF2 import PdfReader
from pathlib import Path
from src.westsum import summarize
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import norm
# from wtf.app import app as application
import os
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
import spacy
from spacy import displacy

model = AutoModelForSeq2SeqLM.from_pretrained("hyunwoongko/ctrlsum-cnndm")
# model = AutoModelForSeq2SeqLM.from_pretrained("hyunwoongko/ctrlsum-arxiv")
# model = AutoModelForSeq2SeqLM.from_pretrained("hyunwoongko/ctrlsum-bigpatent")

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-cnndm")
# tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-arxiv")
# tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-bigpatent")

app = Flask(__name__)
app.secret_key = 'NTU is the best'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:%s@localhost/FYP' % urlquote('L@d060296')
app.config['UPLOAD_FOLDER'] = 'static/files'
# db = SQLAlchemy(app)
# login = LoginManager(app)
# login.init_app(app)


@app.route('/', methods=['GET','POST'])
def index():
    NER = spacy.load("en_core_web_sm")
    keyword_list = ['DATE','ORDINAL','CARDINAL','TIME']
    en_list = set()
    """Home or summarization page for user to summarize their text or PDF file"""
    sumForm = SummaryForm()
    df = pd.DataFrame()
    sents = ""
    count = 0
    df_output = ""
    ctrlsum_result = ""
    if sumForm.validate_on_submit():
        f = sumForm.upload.data
        file_type = str(f.filename).split(".")[-1]
        if file_type == 'txt':
            doc_text = str(f.read().decode())
        if file_type == 'pdf':
            reader = PdfReader(f)
            page = reader.pages[0]
            doc_text = page.extract_text()
        doc_text = doc_text
        # inputs = tokenizer(doc_text,truncation = True, return_tensors="pt")
        # outputs = model.generate(**inputs)
        # outputs = tokenizer.decode(outputs[0])
        # data = hf_query(doc_text, api_id)
        if request.method == "POST":
            cumudist = float(request.form.get("dist"))
        df = summarize(doc_text)
        if cumudist < df['cumulative_dist'][0]:
            count = 1
            df_output = df[df.index == 0]
            sents = ' '.join(df_output['text'].tolist())
        else:
            df_output = df[df['cumulative_dist'] < cumudist]
            count = df_output['text'].count()
            sents = ' '.join(df_output['text'].tolist())
        text2= NER(sents)
        for word in text2.ents:
            if word.label_ not in keyword_list:
                en_list.add(word.text)

        # sents = "£1,400 =>"+sents
        # sents = tokenizer(sents, return_tensors="pt")
        # input_ids, attention_mask = sents["input_ids"], sents["attention_mask"]
        # sents = model.generate(input_ids, attention_mask=attention_mask, num_beams=5)
        # ctrlsum_result = tokenizer.batch_decode(sents)[0]
        # for id in summary_ids:
        #     summary += sentences[id]['text']
        #     summary += " "
        # summary = summary.replace("\r\n","")
        # text = Text(og_text=doc_text,sum_text=str(sents),user_id=current_user.user_id)
        # db.session.add(text)
        # db.session.commit()
    return render_template('index.html', form=sumForm, output=en_list)
        
if __name__ == '__main__':
    app.run(debug=True)