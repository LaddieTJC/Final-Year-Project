
from flask import Flask, redirect, request, render_template, redirect,url_for, flash,session
from flask_wtf import FlaskForm
from flask_wtf.file import  FileRequired
import torch
from register import RegistrationForm,LoginForm
from user import User,Text
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

app = Flask(__name__)
app.secret_key = 'NTU is the best'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:%s@localhost/FYP' % urlquote('L@d060296')
app.config['UPLOAD_FOLDER'] = 'static/files'
db = SQLAlchemy(app)
login = LoginManager(app)
login.init_app(app)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))

@app.route('/index', methods=['GET','POST'])
def index():

    """Home or summarization page for user to summarize their text or PDF file"""

    if current_user.is_authenticated:
        sumForm = SummaryForm()
        df = pd.DataFrame()
        sents = ""
        count = 0
        df_output = ""
        html_sents = ""
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
                sents = f"Top {count} sentences: \r\n {sents}"
            else:
                df_output = df[df['cumulative_dist'] < cumudist]
                count = df_output['text'].count()
                sents = ' '.join(df_output['text'].tolist())
                html_sents = f"Top {count} sentences: \r\n {sents}"
            # for id in summary_ids:
            #     summary += sentences[id]['text']
            #     summary += " "
            # summary = summary.replace("\r\n","")
            text = Text(og_text=doc_text,sum_text=str(sents),user_id=current_user.user_id)
            db.session.add(text)
            db.session.commit()
        return render_template('index.html', form=sumForm, output=html_sents)
        
    else:
        flash('Please login.', 'danger')
        return redirect(url_for('login'))

@app.route('/history', methods=['GET','POST'])
def history():

    """ View the list of text or PDF files the user summarized"""

    # texts = db.session.query(Text).filter(Text.user_id==User.user_id)
    # texts = Text.query.all()
    texts = Text.query.join(User).filter(Text.user_id==current_user.user_id)
    return render_template('history.html',texts=texts,cr=current_user.get_id)

@app.route("/logout",methods=['GET'])
def logout():
    logout_user()
    flash('you have logged out successfully','success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET','POST'])
def register():

    """Registration Page for new user"""

    reg_form = RegistrationForm()
    if reg_form.validate_on_submit():
        email = reg_form.email.data
        password = reg_form.password.data

        #check if email exists
        user = User(email=email,user_pw=password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registered Succesfully. Please Login.','success')
        return redirect(url_for('login'))
    return render_template('register.html', form=reg_form)

@app.route("/", methods=['GET','POST'])
def login():

    """Login Page"""

    login_form = LoginForm()
    if login_form.validate_on_submit():
        user_object = User.query.filter_by(email=login_form.email.data).first()
        login_user(user_object)
        return redirect(url_for('index'))

    return render_template('login.html', form=login_form)


if __name__ == '__main__':
    app.run(debug=True)



##the Great Wall of China is an ancient series of walls and fortifications, totaling more than 13,000 miles in length, located in northern China. Perhaps the most recognizable symbol of China and its long and vivid history, the Great Wall was originally conceived by Emperor Qin Shi Huang in the third century B.C. as a means of preventing incursions from barbarian nomads. The best-known and best-preserved section of the Great Wall was built in the 14th through 17th centuries A.D., during the Ming dynasty. Though the Great Wall never effectively prevented invaders from entering China, it came to function as a powerful symbol of Chinese civilization’s enduring strength.