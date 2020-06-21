from flask import url_for, render_template
from webapp import app


@app.route("/")
@app.route("/home")
def home():
	return render_template("index.html")


@app.route("/about")
def about():
	return render_template("about.html")


