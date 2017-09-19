from flask import Flask

app = Flask(__name__)

@app.route('/users/<string:username>')
def hello_world(username='MyName'):
    return("Hello {}!".format(username))
