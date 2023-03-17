import sys
from datetime import datetime
import io
import base64
from PIL import Image
from flask import Response, request, make_response, render_template
from flask_restx import Resource
from app.main.dto import HomeDto
from app.main.service.emp_service import EmployeeService
from app.main.service.video_service import register_employee, make_attendance, confirm_attendance, \
    display_attendance
from app.main.service.register_service import render_video as render_vid, register_employee as register_emp
import json
from flask_paginate import Pagination, get_page_args


api = HomeDto.api
apiModel = HomeDto.apiModel

@api.route('/register')
class Employee(Resource):
    def post(self):
        frameCount = request.form.get('frameCount')
        empId = request.form.get('empId')
        image = request.files['image'].stream.read()
        image = io.BytesIO(image)
        return make_response(register_employee(frameCount, empId, image, apiModel), 200)

    def get(self):
        empId = request.args.get('empId')
        response = EmployeeService.emp_details(empId)
        return make_response(response, 200)

@api.route('/create_emp')
class CreateEmployee(Resource):
    def post(self):
        name = request.form.get('name')
        empId = request.form.get('empId')
        return make_response(EmployeeService.register_employee(name, empId), 200)


@api.route('/attendance')
class EmployeeAttendance(Resource):
    def post(self):
        try:
            image = request.files['image'].stream.read()
            punch_type = request.form.get('punch')
            im = io.BytesIO(image)
            face_data = base64.b64encode(image)
            results = make_attendance(im, punch_type, face_data, apiModel)
            page, per_page, offset = get_page_args(page_parameter='page',per_page_parameter='per_page')
            paginated_results = results[offset: offset + per_page]
            pagination = Pagination(page=page, per_page=per_page, total=len(results))
            return make_response(make_attendance(im, punch_type, face_data, apiModel, paginated_results, pagination),200)
            #return make_response(make_attendance(im, punch_type, face_data, apiModel), 200)
        except:
            print(sys.exc_info())
            return None

    def put(self):
        id = request.form.get('id')
        confirm = request.form.get('confirm')
        results = confirm_attendance(id,confirm)
        page, per_page, offset = get_page_args(page_parameter='page',per_page_parameter='per_page')
        paginated_results = results[offset: offset + per_page]
        pagination = Pagination(page=page, per_page=per_page, total=len(results)) 
        return make_response(confirm_attendance(id, confirm, paginated_results, pagination), 200)
        #return make_response({'data': paginated_results, 'pagination': pagination}, 200)
        

    def get(self):
        return make_response(display_attendance(), 200)