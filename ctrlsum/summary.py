from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField,EmailField,SelectField,FileField
from wtforms.validators import InputRequired, Length, EqualTo, ValidationError
from flask_wtf.file import FileAllowed, FileRequired


class SummaryForm(FlaskForm):
    upload = FileField('Please upload a file',validators=[FileRequired(), FileAllowed(['txt','pdf'],'Only pdf and txt files')])
    submit_btn = SubmitField('Submit')
