from flask import Flask, session, redirect, url_for, render_template
from app import app

@app.route("/")
def home():
    return "Home Page - Flask is working!"


@app.route("/add_category")
def add_category():
    return render_template("categotize.html")
