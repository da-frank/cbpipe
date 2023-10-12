import streamlit as st
from celery import Celery

app = Celery('tasks', broker='pyamqp://guest@rabbitmq//')

@st.cache_data
def get_running_tasks():
    return app.control.inspect().active()

st.title('Current Running Celery Tasks')

tasks = get_running_tasks()

if not tasks:
    st.write('No tasks currently running.')
else:
    for task in tasks:
        st.write(f'Task ID: {task}')
