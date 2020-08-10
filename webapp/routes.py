from flask import url_for, render_template, flash, redirect, request, make_response
from webapp import app, db, bcrypt, login_manager
from flask_login import login_user, current_user, logout_user, login_required
from webapp.modules import User, GroceryList, Receipt
from webapp.forms import RegistrationForm, LoginForm, UpdateAccountForm, GroceryListForm, ReceiptUploadForm
from webapp.dashboard import Dashboard
from flask import Markup
from PIL import Image, ImageOps
import secrets
import os
from datetime import datetime
grocery_list = []

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

def save_receipt(form_picture):
	random_hex = secrets.token_hex(16)
	_, f_ext = os.path.splitext(form_picture.filename)
	picture_fn = random_hex + f_ext
	picture_path = os.path.join(app.root_path, "static/receipt_images", picture_fn)
	
	i = Image.open(form_picture)
	i.save(picture_path)
	return picture_fn

@app.route("/dashboard/receipts", methods=["GET","POST"])
def dashboardReceipts():
	if not current_user.is_authenticated:#should be done with @login_required (but it somehow does not work for account page)
		return redirect(url_for("home"))
	dash = Dashboard("Neonode")
	image_file = url_for("static",filename="userPictures/" + current_user.image_file)
	receipts = dash.generateHTMLForReceipts()
	form = ReceiptUploadForm()
	if form.validate_on_submit():
		for pic in form.receipt_images.data:
			receipt_file = save_receipt(pic)
			rec = Receipt(image_file=receipt_file, user=current_user)
			db.session.add(rec)
			db.session.commit()
		flash("Receipt(s) uploaded!", "success")
	return render_template("dashboard/receipts.html", 
	username=current_user.username, 
	profilePic=image_file,
	receipts=receipts,
	latestUpdate=dash.latestUpdate,
	form=form
	)

@app.route("/dashboard/lists")
def groceryLists():
	if not current_user.is_authenticated:
		return redirect(url_for("home"))
	lists = GroceryList.query.filter_by(user=current_user).all()
	image_file = url_for("static",filename="userPictures/" + current_user.image_file)
	return render_template("dashboard/grocery_lists.html", 
	username=current_user.username, 
	profilePic=image_file,
	lists=lists)

@app.route("/dashboard/list/<int:listID>", methods=["GET","POST"])
def groceryList(listID):
	global grocery_list
	filename = os.getcwd() + "/webapp/" + "allproducts.txt"#may not work for windows
	with open(filename, "r") as file:
		f = file.read()
		products = f.split("\n")
	gr_list = GroceryList.query.filter_by(id=listID).first()
	items = gr_list.items.split(",")
	if not grocery_list:
		grocery_list = items
	name = gr_list.name
	if request.method == "POST":
		if "save-list" in request.form:
			s = grocery_list[0]
			for i in range(1, len(grocery_list)):
				s +=","+grocery_list[i]
			gr_list.items = s
			if request.form["list-name"]:
				gr_list.name = request.form["list-name"]
			gr_list.num_items = len(grocery_list)
			gr_list.timestamp=datetime.now()
			db.session.commit()
			del grocery_list[:] # empty the current grocerList Array
			return redirect(url_for("groceryLists"))
		for product in products:
			if product in request.form:
				grocery_list.append(product)
				return redirect(url_for("groceryList", listID=listID))
			if "Remove-"+product in request.form:
				grocery_list.remove(product)
				return redirect(url_for("groceryList" , listID=listID))
	if not current_user.is_authenticated or current_user.id != gr_list.user_id:
		return redirect(url_for("home"))
	
	image_file = url_for("static",filename="userPictures/" + current_user.image_file)
	return render_template("dashboard/view_list.html", 
	username=current_user.username, 
	profilePic=image_file,
	gr_name=name,
	gr_items=grocery_list,
	products=products)

@app.route("/dashboard/create_list", methods=["GET","POST"])
@login_required
def create_list():
	filename = os.getcwd() + "/webapp/" + "allproducts.txt"#may not work for windows
	
	with open(filename, "r") as file:
		f = file.read()
		products = f.split("\n")
	image_file = url_for("static", filename="userPictures/" + current_user.image_file)
	if request.method == "POST":
		if "create-list" in request.form:
			s = grocery_list[0]
			for i in range(1, len(grocery_list)):
				s +=","+grocery_list[i]
			gr_list = GroceryList(name=request.form["list-name"], user=current_user, items=s, num_items=len(grocery_list), timestamp=datetime.now())
			db.session.add(gr_list)
			db.session.commit()
			del grocery_list[:] # empty the current grocerList Array
			return redirect(url_for("groceryLists"))
		if "manual-adding" in request.form:
			product = request.form["add-manually"]
			with open(filename, "a") as file:
				file.write("\n"+product)
			grocery_list.append(product)
			return redirect(url_for("create_list"))
		for product in products:
			if product in request.form:
				grocery_list.append(product)
				return redirect(url_for("create_list"))
			if "Remove-"+product in request.form:
				grocery_list.remove(product)
				return redirect(url_for("create_list"))
	else:
		return render_template("dashboard/create_list.html",
			username=current_user.username,
			profilePic=image_file,
			products=products,
			grocery_list = grocery_list,
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

