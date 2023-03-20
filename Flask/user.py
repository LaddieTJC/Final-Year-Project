
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):

    __tablename__ = "users"
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(30), unique=True,nullable=False)
    user_pw = db.Column(db.String(), nullable=False)
    # user_name = db.Column(db.String(30))

    def get_id(self):
        return (self.user_id)
 
class Text(db.Model):

    __tablename__ = "texts"
    text_id = db.Column(db.Integer, primary_key=True)
    og_text = db.Column(db.Text)
    sum_text = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"),nullable=False)
    top_k_sents = db.Column(db.Text)