import sys
import pandas as pd
from app.main.model.employee import Employee



class EmployeeService:
    @staticmethod
    def emp_details(empId):
        try:
            if not empId:
                response = []
                employees = Employee.query.all()
                for emp in employees:
                    response.append({"id": emp.id, "empId": emp.empId, "empName": emp.empName})
                return {"data": response, "message": "Employee details displayed.", "status": "Success",
                        "statusCode": 200}
            empId = int(empId)
            emp = Employee.query.filter_by(empId=empId).first()
            response = {"id": int(emp.id), "empName": emp.empName, "empId": emp.empId, "features": emp.features}
            if emp:
                return {"data": response, "message": "Employee details displayed.", "status": "Success", "statusCode": 200}
            else:
                return {"message": "Employee ID not Exists", "status": "Failure", "statusCode": 400}
        except:
            return {"message": str(sys.exc_info()[1]), "status": "Success", "statusCode": 500}

    @staticmethod
    def register_employee(name, empId):
        try:
            empId = int(empId)
            if EmployeeService.check_employee_exists(empId):
                return {"message": "Employee already exists", "status": "Failure", "statusCode": 401}
            emp=Employee()
            emp.save(name, empId)
            return {"message": "Employee Created Successfully", "status": "Success", "statusCode": 200}
        except:
            return {"message": str(sys.exc_info()[1]), "status": "Failure", "statusCode": 500}
    
    @staticmethod
    def check_employee_exists(empId):
        val = Employee.query.filter_by(empId=empId).first()
        return val
    




