from flask import Flask
from story_generator.story_generator import story_bp

app = Flask(__name__)

app.register_blueprint(story_bp)

if __name__=="__main__":
    app.run(debug=True)