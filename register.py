from flask_wtf import FlaskForm

from torch import equal
from wtforms import StringField, PasswordField, SubmitField,EmailField
from wtforms.validators import InputRequired, Length, EqualTo, ValidationError, Regexp
from user import User

def invalid_credentials(form,field):
    #Username and password checker
    email_entered = form.email.data
    password_entered = field.data
    user_object = User.query.filter_by(email=form.email.data).first()
    if user_object is None:
        raise ValidationError("email does not exist")
    elif password_entered != user_object.user_pw:
        raise ValidationError("Password is incorrect")

class RegistrationForm(FlaskForm):

    email = EmailField('email', validators=[InputRequired(message="Email Required"), Length(min=10,max=30,message="Email must be in a correct format")]) 
    password = PasswordField('password',validators=[InputRequired(message="Password Required"), Length(min=8,max=10,message="Password must be between 8 to 10")])
    cfrm_password = PasswordField('cfrm_password',validators=[InputRequired(message="Password Required"), EqualTo('password', message='Password must match')])
    name = StringField('name',validators=[InputRequired(message="Name Required"),Length(min=5,max=30,message="Max length of 30"),Regexp("^\w+$", message="Name should not contain any special characters")])
    submit_btn = SubmitField('Create')

    def validate_email(self, email):
        user_object = User.query.filter_by(email=email.data).first()
        if user_object:
            raise ValidationError("email already exist, please use another email")


class LoginForm(FlaskForm):

    email = EmailField('email_label', validators=[InputRequired(message="Email Required")])
    password = PasswordField('password_label', validators=[InputRequired(message="Password Required"), invalid_credentials])
    submit_btn = SubmitField('Login')

class changePasswordForm(FlaskForm):
    oldpw = PasswordField('oldpw', validators=[InputRequired(message="Old Password Required")])
    newpw = PasswordField('newpw', validators=[InputRequired(message="New Password Required")])
    newCfmpw = PasswordField('newCfmpw', validators=[InputRequired(message="Old Password Required"), EqualTo('newpw', message='New password must match')])
    submit_btn = SubmitField('Change Password')


class changeNameForm(FlaskForm):
    password = PasswordField('password', validators=[InputRequired(message="Old Password Required")])
    name = StringField('name',validators=[InputRequired(message="Name Required"),Length(min=5,max=30,message="Max length of 30"),Regexp("^[0-9A-Za-z_]+$", message="Name should not contain any special characters")])
    submit_btn = SubmitField('Change Password')


class changeEmailForm(FlaskForm):
    password = PasswordField('password', validators=[InputRequired(message="Old Password Required")])
    email = EmailField('email', validators=[InputRequired(message="Email Required"), Length(min=10,max=30,message="Email must be in a correct format")])
    submit_btn = SubmitField('Change Password')
    
    def validate_email(self, email):
        user_object = User.query.filter_by(email=email.data).first()
        if user_object:
            raise ValidationError("email already exist, please use another email")