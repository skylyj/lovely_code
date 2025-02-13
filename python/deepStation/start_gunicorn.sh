#!/bin/bash
gunicorn --workers 3 --bind unix:myapp.sock -m 007 wsgi:app

