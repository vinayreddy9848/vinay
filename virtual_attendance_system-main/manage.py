import unittest
from flask_cors import CORS
from flask_script import Manager
import os
from app import blueprint
from app.main import create_app
from flask_migrate import Migrate, MigrateCommand

app = create_app(os.getenv('PROJECT_ENV') or 'dev')
app.register_blueprint(blueprint)
app.app_context().push()
manager = Manager(app)
CORS(app)

manager.add_command('db', MigrateCommand)



@manager.command
def run():
    env = os.getenv('PROJECT_ENV')
    if env is None:
        os.environ['PROJECT_ENV'] = 'dev'
    app.run(debug=True, host='0.0.0.0', port=5000)


@manager.command
def test():
    """Runs the unit tests."""
    tests = unittest.TestLoader().discover('app/test', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    return 1


if __name__ == '__main__':
    manager.run()
