from flask_cors import CORS
from flask_script import Manager
import os
from flask import session
from app import blueprint
from app.main import create_app
from flask_migrate import MigrateCommand
from keras_vggface.vggface import VGGFace
app = create_app(os.getenv('PROJECT_ENV') or 'dev')
app.register_blueprint(blueprint)
app.app_context().push()
manager = Manager(app)
CORS(app)

manager.add_command('db', MigrateCommand)

def load_vgg_resnet():
    # model = VGGFace(include_top=False,pooling='avg') # default : VGG16 , you can use model='resnet50' or 'senet50'
    return VGGFace(model='senet50', include_top=False, pooling='avg')


def load_models():
    model = load_vgg_resnet()
    return model

if __name__ == '__main__':
    env = os.getenv('PROJECT_ENV')
    if env is None:
        os.environ['PROJECT_ENV'] = 'dev'
    session["model"] = load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)