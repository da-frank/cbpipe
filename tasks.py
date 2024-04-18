from celery import Celery
import trainer
import distilbert

app = Celery('tasks', broker='pyamqp://guest@rabbitmq//')

@app.task
def train(group, timestamp, stemmer="cistem"):
    print(f"Training {group} at {timestamp} with {stemmer} stemmer.")
    trainer.Trainer(stemmer=stemmer).train(group+"_"+timestamp)

@app.task
def train_distilbert(group, timestamp):
    print(f"Training {group} at {timestamp} with distilBERT.")
    distilbert.Distilbert().train(group, timestamp)