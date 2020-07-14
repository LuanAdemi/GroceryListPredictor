from flask import url_for, render_template, flash, redirect, request, make_response
from webapp import app, db, bcrypt, login_manager
from flask_login import login_user, current_user, logout_user, login_required
from webapp.modules import User
from webapp.forms import RegistrationForm, LoginForm


@app.route("/")
@app.route("/home")
def home():
	return render_template("base.html")

# TODO: Route to dashboard
@app.route("/gettingstarted", methods=["GET","POST"])
def gettingstarted():
	if current_user.is_authenticated:
		return redirect(url_for("home"))
		
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
		return redirect(next_page) if next_page else redirect(url_for("home"))

	if lform.validate_on_submit():
		user = User.query.filter_by(email=lform.email.data).first()
		if user and bcrypt.check_password_hash(user.password, lform.password.data):
			login_user(user)
			next_page = request.args.get("next")
			return redirect(next_page) if next_page else redirect(url_for("home"))
		else:
			flash("Login unsuccessful. Please check username and password", "danger")
	
	return render_template("gettingstarted.html", rform=rform, lform=lform)

@app.route("/about")
def about():
	return render_template("about.html")


@app.route("/logout")
def logout():
	logout_user()
	return redirect(url_for("home"))

