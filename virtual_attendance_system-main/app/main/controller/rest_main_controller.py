import os
from flask import render_template, make_response, request, flash, redirect
from flask_restx import Resource
from app.main.constants import file_constants as cnst
from app.main.dto import RestHomeDto
from app.main.utils import file_utils

api = RestHomeDto.api


@api.route('/')
class Home(Resource):
    def get(self):
        headers = {'Content-Type': 'application/json'}
        return make_response({"message": "API UNDER CONSTRUCTION"}, 200,
                             headers)

    def post(self):
        headers = {'Content-Type': 'application/json'}
        return make_response({"message": "API UNDER CONSTRUCTION"}, 200,
                             headers)
