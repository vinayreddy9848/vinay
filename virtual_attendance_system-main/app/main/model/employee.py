from sqlalchemy.dialects.postgresql import ARRAY

from app.main import db
from sqlalchemy.ext.mutable import Mutable

class MutableList(Mutable, list):
    def append(self, value):
        list.append(self, value)
        self.changed()

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, MutableList):
            if isinstance(value, list):
                return MutableList(value)
            return Mutable.coerce(key, value)
        else:
            return value

class Employee(db.Model):
    __tablename__ = 'employee'
    id = db.Column(db.Integer, primary_key=True)
    empName = db.Column(db.String, unique=True, nullable=False)
    empId = db.Column(db.Integer, nullable=False)
    features = db.Column(MutableList.as_mutable(ARRAY(db.Float)))

    def save(self, name, empId):
        self.empName = name
        self.empId = empId
        self.features = []
        db.session.add(self)
        db.session.commit()
    def update(self, features):
        self.features = features
        db.session.commit()


class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    empId = db.Column(db.String, nullable=False)
    time = db.Column(db.DateTime(timezone=False))
    punch = db.Column(db.String, nullable=False)
    confirmed = db.Column(db.Boolean, nullable=False)
    faceData = db.Column(db.String, nullable=True)

    def save(self, empId, time, punch, confirmed, faceData):
        self.empId = empId
        self.time = time
        self.punch = punch
        self.confirmed = confirmed
        self.faceData = faceData
        db.session.add(self)
        db.session.commit()
        db.session.flush()
        return self.id
    
    @staticmethod
    def get_paginated(page, per_page):
        return Attendance.query.paginate(page=page, per_page=per_page)
    @staticmethod
    def update():
        db.session.commit()

#class Employee():
    #select empId from employee where empName = 'self.empName';
    #select empName from employee where empId = 'self.empId';


