from celery import Celery
import trainer

app = Celery('tasks', broker='pyamqp://guest@172.17.0.3//')

@app.task
def train(group):
    print(f"Training {group}")
    train(group)
