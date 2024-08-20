from flask import Flask, render_template, request, redirect, url_for, session
import os
import subprocess

# Import the Blueprint
from models.overheadshoulderpress import overheadshoulderpress_app, reset_overheadshoulderpress
from models.bentoverrows import bentoverrows_app, reset_bentoverrows
from models.calfraises import calfraises_app, reset_calfraises
from models.crunches import crunches_app, reset_crunches
from models.frontraise import frontraise_app, reset_frontraise
from models.lateralraises import lateralraise_app, reset_lateralraise
from models.lunges import lunges_app, reset_lunges
from models.planks import planks_app, reset_planks
from models.pushups import pushups_app, reset_pushups
from models.squats import squats_app, reset_squats
from models.standingsidecrunches import standingsidecrunches_app, reset_standingsidecrunches
from models.uprightrows import uprightrows_app, reset_uprightrows
from models.bicepcurl import bicepcurl_app, reset_bicepcurl

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Register the Blueprint
app.register_blueprint(overheadshoulderpress_app)
app.register_blueprint(bentoverrows_app)
app.register_blueprint(calfraises_app)
app.register_blueprint(crunches_app)
app.register_blueprint(frontraise_app)
app.register_blueprint(lateralraise_app)
app.register_blueprint(lunges_app)
app.register_blueprint(planks_app)
app.register_blueprint(pushups_app)
app.register_blueprint(squats_app)
app.register_blueprint(standingsidecrunches_app)
app.register_blueprint(uprightrows_app)
app.register_blueprint(bicepcurl_app)

# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Simple user authentication (for demonstration)
        if username == 'user' and password == 'password':
            session['username'] = username
            return redirect(url_for('model_selection'))
        else:
            return "Invalid credentials!", 403  # Explicitly return 403 on invalid credentials
    return render_template('login.html')

# Route for the model selection page
@app.route('/model_selection')
def model_selection():
    if 'username' in session:
        # Reset counters for all models to ensure they start fresh
        reset_overheadshoulderpress()
        reset_bentoverrows()
        reset_calfraises()
        reset_crunches()
        reset_frontraise()
        reset_lateralraise()
        reset_lunges()
        reset_planks()
        reset_pushups()
        reset_squats()
        reset_standingsidecrunches()
        reset_uprightrows()
        reset_bicepcurl()
        return render_template('model_selection.html')
    else:
        return redirect(url_for('login'))

# Route for each model's page
@app.route('/model/<model_name>')
def model_page(model_name):
    if 'username' in session:
        # Reset counters before starting a new exercise
        reset_overheadshoulderpress()
        reset_bentoverrows()
        reset_calfraises()
        reset_crunches()
        reset_frontraise()
        reset_lateralraise()
        reset_lunges()
        reset_planks()
        reset_pushups()
        reset_squats()
        reset_standingsidecrunches()
        reset_uprightrows()
        reset_bicepcurl()

        # Start the appropriate model based on the selection
        model_script = os.path.join('models', f"{model_name}.py")
        if os.path.exists(model_script):
            subprocess.Popen(['python', model_script])
            return render_template('model_page.html', model_name=model_name)
        else:
            return "Model script not found!", 404
    else:
        return redirect(url_for('login'))

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
