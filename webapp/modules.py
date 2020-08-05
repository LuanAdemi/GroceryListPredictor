from webapp import db, login_manager
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
	return User.query.get(int(user_id))


class User(UserMixin, db.Model):
	id = db.Column(db.Integer,primary_key=True)
	username = db.Column(db.String(20), unique=True, nullable=False)
	email = db.Column(db.String(120), unique=True, nullable=False)
	password = db.Column(db.String(60), nullable=False)
	image_file = db.Column(db.String(200), nullable=False, default="default.png")
	no_lists = db.Column(db.Integer, nullable=False, default=0)
	created = db.relationship('GroceryList', backref='user')
	def __repr__(self):
		return f"User('{self.username}', '{self.email}', '{self.image_file})'"


class GroceryList(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
	name = db.Column(db.String(120), nullable=False)
	items = db.Column(db.Text, nullable=False)
	def __repr__(self):
		return f"GroceryList('{self.id}', '{self.name}')"

