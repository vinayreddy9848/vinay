from flask import Flask
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session

flask_bcrypt = Bcrypt()
db = SQLAlchemy()


def create_app(config_name):
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:123456@localhost:5432/virtual_attendance_system"
    db.init_app(app)
    migrate = Migrate(app, db)
    flask_bcrypt.init_app(app)
    with app.app_context():
        db.create_all()
    Session(app)
    return app
