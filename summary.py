from flask_wtf import FlaskForm
from wtforms import SubmitField,FileField
from flask_wtf.file import FileAllowed, FileRequired


class SummaryForm(FlaskForm):
    upload = FileField('Please upload a file',validators=[FileRequired(), FileAllowed(['txt','pdf'],'Only pdf and txt files')])
    submit_btn = SubmitField('Submit')

class T5Form(FlaskForm):
    t5_submit_btn = SubmitField("submit")