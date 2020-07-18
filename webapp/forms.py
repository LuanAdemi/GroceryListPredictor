from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, SelectField, DateTimeField, SelectMultipleField, MultipleFileField, FileField, DecimalField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from webapp.modules import User


class RegistrationForm(FlaskForm):
	username = StringField("Username", validators=[DataRequired(), Length(min=2,max=20)])
	email = StringField("Email", validators=[DataRequired(),Email()])
	password = PasswordField("Password", validators=[DataRequired()])
	confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo("password")])
	submit = SubmitField("Sign Up")
	def validate_username(self,username):
		user = User.query.filter_by(username=username.data).first()
		if user:
			raise ValidationError("username already exists")

	def validate_email(self,email):
		user = User.query.filter_by(email=email.data).first()
		if user:
			raise ValidationError("email already registered")


class LoginForm(FlaskForm):
	email = StringField("Email", validators=[DataRequired(),Email()])
	password = PasswordField("Password", validators=[DataRequired()])
	remember = BooleanField("Remember Me")
	submit = SubmitField("Log In")

