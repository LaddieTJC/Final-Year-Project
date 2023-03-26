
from flask import Flask, redirect, request, render_template, redirect,url_for, flash,session
from register import RegistrationForm,LoginForm, changePasswordForm, changeNameForm, changeEmailForm
from user import User,Text
from wtforms import *
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote_plus as urlquote
from flask_login import LoginManager,login_user,current_user, login_required,logout_user
from summary import SummaryForm, T5Form
from PyPDF2 import PdfReader
from src.westsum import summarize
import pandas as pd
from flask_mail import Mail, Message
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'laddie.fyp@gmail.com'
app.config['MAIL_PASSWORD'] = 'drijtxewxbnnzotd'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)
app.secret_key = 'NTU is the best'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:%s@localhost/FYP' % urlquote('L@d060296')
app.config['UPLOAD_FOLDER'] = 'static/files'
db = SQLAlchemy(app)
login = LoginManager(app)
login.init_app(app)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))

@app.route('/profile')
def profile():
    if current_user.is_authenticated:
        user = User.query.get(current_user.user_id)
        texts = Text.query.join(User).filter(Text.user_id==current_user.user_id).count()

        return render_template('profile.html',user=user, numTexts =texts)
    else:
        flash('Please login.', 'danger')
        return redirect(url_for('login'))
    

@app.route('/summarizer', methods=['GET','POST'])
def summarizer():

    """Home or summarization page for user to summarize their text or PDF file"""

    if current_user.is_authenticated:
        sumForm = SummaryForm()
        sents = "" 
        all_sents = ""
        final_summary = ""
        doc_text = ""
        selectedSents = ""
        t5Form = T5Form()
        if sumForm.submit_btn.data and sumForm.validate_on_submit():
            f = sumForm.upload.data
            file_type = str(f.filename).split(".")[-1]
            if file_type == 'txt':
                doc_text = str(f.read().decode())
            if file_type == 'pdf':
                reader = PdfReader(f)
                page = reader.pages[0]
                doc_text = page.extract_text()
            if len(word_tokenize(doc_text)) > 500:
                sents = "Number of tokens in the document exceed 512"
                doc_text = ""
            else:
                if request.method == "POST":
                    cumudist = float(request.form.get("dist"))
                sents,final_summary,all_sents = summarize(doc_text,cumudist)
                # text = Text(og_text=doc_text,sum_text=str(final_summary),top_k_sents=str(sents),user_id=current_user.user_id)
                # db.session.add(text)
                # db.session.commit()
        return render_template('summarizer.html', form=sumForm,t5Form=t5Form, original=doc_text,output=sents,final_output=final_summary,sentences=all_sents)

    else:
        flash('Please login.', 'danger')
        return redirect(url_for('login'))

@app.route('/history', methods=['GET','POST'])
def history():

    """ View the list of text or PDF files the user summarized"""

    # texts = db.session.query(Text).filter(Text.user_id==User.user_id)
    # texts = Text.query.all()
    if current_user.is_authenticated:
        texts = Text.query.join(User).filter(Text.user_id==current_user.user_id)
        return render_template('history.html',texts=texts,cr=current_user.get_id)
    else:
        flash('Please login.', 'danger')
        return redirect(url_for('login'))

@app.route("/logout",methods=['GET'])
def logout():
    logout_user()
    flash('you have logged out successfully','logout')
    return redirect(url_for('landingPage'))

@app.route('/register', methods=['GET','POST'])
def register():

    """Registration Page for new user"""

    regForm = RegistrationForm()
    if regForm.validate_on_submit():
        email = regForm.email.data
        password = regForm.password.data
        name = regForm.name.data
        #check if email exists
        user = User(email=email,user_pw=password,user_name=name)
        
        db.session.add(user)
        db.session.commit()
        msg = Message('Hello from the other side!', sender ='laddie.fyp@gmail.com', recipients = [email])
        msg.body = f"Hey {name}, Thanks for signing up, welcome to the family."
        mail.send(msg)
        flash('Registered Succesfully. Please Login.','register')
        return redirect(url_for('login'))
    return render_template('register.html', form=regForm)

@app.route("/")
def landingPage():
    return render_template('landingPage.html')

@app.route("/extractive")
def extractive():
    return render_template('extractive.html')

@app.route("/abstractive")
def abstractive():
    return render_template('abstractive.html')

@app.route("/changeName", methods=['GET','POST'])
def changeName():
    nameForm = changeNameForm()
    user = User.query.filter_by(user_id=current_user.user_id).first()
    outputStr = ""
    if nameForm.validate_on_submit():
        pw = nameForm.password.data
        name = nameForm.name.data
        if pw == user.user_pw:
            user.user_name = name
            db.session.merge(user)
            db.session.commit()
            outputStr = "You changed your name."
            nameForm = changeNameForm(formdata=None)
        else:
            outputStr = "Password does not match."
    return render_template('changeName.html', form=nameForm, outputStr=outputStr, user=user)

@app.route("/changeEmail", methods=['GET', 'POST'])
def changeEmail():
    emailForm = changeEmailForm()
    user = User.query.filter_by(user_id=current_user.user_id).first()
    outputStr = ""
    if emailForm.validate_on_submit():
        pw = emailForm.password.data
        email = emailForm.email.data
        if pw == user.user_pw:
            user.email = email
            db.session.merge(user)
            db.session.commit()
            outputStr = "you changed your email"
        elif user.email == email:
            outputStr = "Email should not match the old one"
        else:
            outputStr = "password does not match"
    return render_template('changeEmail.html', user=user, form=emailForm, outputStr=outputStr)

@app.route("/changepw", methods=['GET','POST'])
def changePassword():
    # user = session.query(User).filter(User.user_id == current_user.user_id).one()
    # print(user.values)
    user = User.query.filter_by(user_id=current_user.user_id).first()
    outputStr = ""
    passwordForm = changePasswordForm()
    if passwordForm.validate_on_submit():
        oldpw = passwordForm.oldpw.data
        newpw = passwordForm.newpw.data
        if oldpw == user.user_pw:
            user.user_pw = newpw
            db.session.merge(user)
            db.session.commit()
            outputStr = "Password Changed"
            passwordForm = changePasswordForm(formdata=None)
        #     flag_modified(user, 'data')
        #     db.session.merge(user)
        #     db.session.flush()
            # User.query.filter_by(user_id = current_user.user_id).update(dict(user_pw=newpw))
        #     db.session.commit()
        else:
            outputStr = "Old Password does not match"
    return render_template('changePassword.html', user=user, form=passwordForm, outputStr=outputStr)

@app.route("/login", methods=['GET','POST'])
def login():
    """Login Page"""

    loginForm = LoginForm()
    if loginForm.validate_on_submit():
        user_object = User.query.filter_by(email=loginForm.email.data).first()
        login_user(user_object)
        return redirect(url_for('summarizer'))

    return render_template('login.html', form=loginForm)


if __name__ == '__main__':
    app.run(debug=True)



##the Great Wall of China is an ancient series of walls and fortifications, totaling more than 13,000 miles in length, located in northern China. Perhaps the most recognizable symbol of China and its long and vivid history, the Great Wall was originally conceived by Emperor Qin Shi Huang in the third century B.C. as a means of preventing incursions from barbarian nomads. The best-known and best-preserved section of the Great Wall was built in the 14th through 17th centuries A.D., during the Ming dynasty. Though the Great Wall never effectively prevented invaders from entering China, it came to function as a powerful symbol of Chinese civilization’s enduring strength.