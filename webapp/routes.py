from flask import url_for, render_template, flash, redirect, request, make_response
from webapp import app, db, bcrypt, login_manager
from flask_login import login_user, current_user, logout_user, login_required
from webapp.modules import User
from webapp.forms import RegistrationForm, LoginForm, UpdateAccountForm
from webapp.dashboard import Dashboard
from flask import Markup
from PIL import Image, ImageOps
import secrets

import os.path


@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")

def save_profile_picture(form_picture):
	random_hex = secrets.token_hex(16)
	_, f_ext = os.path.splitext(form_picture.filename)
	picture_fn = random_hex + f_ext
	picture_path = os.path.join(app.root_path, "static/userPictures", picture_fn)
	
	output_size = (150,150)
	i = Image.open(form_picture)
	i = ImageOps.fit(i,output_size, Image.ANTIALIAS)
	i.save(picture_path)
	return picture_fn

@app.route("/dashboard/account", methods=["GET","POST"])
@login_required
def account():
	form = UpdateAccountForm()
	if form.validate_on_submit():
		if form.picture.data:
			picture_file = save_profile_picture(form.picture.data)
			current_user.image_file = picture_file
		current_user.username = form.username.data
		current_user.email = form.email.data
		db.session.commit()
		flash("Your account has been updated!", "success")
		return redirect(url_for("account"))
	elif request.method == "GET":
		form.username.data = current_user.username
		form.email.data = current_user.email
	image_file = url_for("static",filename="userPictures/" + current_user.image_file)
	return render_template("dashboard/account.html", profilePic=image_file, form=form, username=current_user.username)


# TODO: database link
@app.route("/dashboard/overview")
def dashboardOverview():
	if not current_user.is_authenticated:
		return redirect(url_for("home"))
	dash = Dashboard("Neonode")
	image_file = url_for("static",filename="userPictures/" + current_user.image_file)
	return render_template("dashboard/overview.html", 
	username=current_user.username, 
	profilePic=image_file,
	numReceipts=len(dash.receipts),
	numWeeks=dash.weeks,
	numLists=len(dash.lists),
	accuracy=dash.accuracy
	)

@app.route("/dashboard/receipts")
def dashboardReceipts():
	if not current_user.is_authenticated:
		return redirect(url_for("home"))
	dash = Dashboard("Neonode")
	image_file = url_for("static",filename="userPictures/" + current_user.image_file)
	receipts = dash.generateHTMLForReceipts()
	return render_template("dashboard/receipts.html", 
	username=current_user.username, 
	profilePic=image_file,
	receipts=receipts,
	latestUpdate=dash.latestUpdate
	)

@app.route("/gettingstarted", methods=["GET","POST"])
def gettingstarted():
	if current_user.is_authenticated:
		return redirect(url_for("dashboardOverview"))
		
	rform = RegistrationForm()
	lform = LoginForm()

	if rform.validate_on_submit():
		hashed_password = bcrypt.generate_password_hash(rform.password.data).decode("utf-8")
		user = User(username=rform.username.data, email=rform.email.data, password=hashed_password)
		db.session.add(user)
		db.session.commit()
		flash(f"Account created for {rform.username.data}! You were logged in!", "success")
		login_user(user)
		next_page = request.args.get("next")
		return redirect(next_page) if next_page else redirect(url_for("dashboardOverview"))

	if lform.validate_on_submit():
		user = User.query.filter_by(email=lform.email.data).first()
		if user and bcrypt.check_password_hash(user.password, lform.password.data):
			login_user(user)
			next_page = request.args.get("next")
			return redirect(next_page) if next_page else redirect(url_for("dashboardOverview"))
		else:
			flash("Login unsuccessful. Please check username and password", "danger")
	
	return render_template("gettingstarted.html", rform=rform, lform=lform)


@app.route("/logout")
def logout():
	logout_user()
	return redirect(url_for("home"))

