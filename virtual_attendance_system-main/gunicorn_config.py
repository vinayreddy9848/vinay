import multiprocessing
import threading, sys, traceback
from threading import Thread
import logging
import os
import errno

hostname = os.uname()[1]

def mkdir_path(path):
    try:
        os.makedirs(path, exist_ok=True)
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print(exc)
class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        mkdir_path(os.path.dirname(filename))

log_path = './logs/'
MakeFileHandler('./logs/')
logging.basicConfig(filename=log_path + '/' + hostname + '_server.log', level=logging.DEBUG, format='%(asctime)s %(name)s :%(message)s')
logger = logging.getLogger(__name__)
bind = "0.0.0.0:5000"
workers = 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2
accesslog = log_path + "/" + "" + hostname + "_all.log"  # STDOUT
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True