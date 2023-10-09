from celery import Celery
import trainer

app = Celery('tasks', broker='pyamqp://guest@rabbitmq//')

@app.task
def train(group, stemmer="cistem"):
    print(f"Training {group}")
    trainer.Trainer(stemmer=stemmer).train(group)
