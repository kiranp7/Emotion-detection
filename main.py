from flask import Flask,render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/runcode')
def run_code():
    subprocess.Popen(['python','emotion.py'])
    return 'PYTHON SCRIPT EXECUTED SUCCESSFULLY, LOADING......'





if __name__=='__main__':
    app.run(debug=True)