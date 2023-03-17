from flask import Blueprint
from flask_restx import Api
# from app.main import DevelopmentConfig
from .main.controller.main_controller import api as home_ns
from .main.controller.rest_main_controller import api as api_ns

blueprint = Blueprint('api', __name__)
basePath = '/facerecog'

api = Api(blueprint)
api.add_namespace(home_ns, basePath + '/home')
api.add_namespace(api_ns, basePath + '/api')