# import the modules
import time
import logging
from watchdog.observers import Observer, polling
from watchdog.events import PatternMatchingEventHandler
from celery import Celery

# Initialize celery app
app = Celery('tasks', broker='redis://localhost:6379/0')


@app.task
def train():
    print("Training")
    # TODO train model

# class overriding PatternMatchingEventhandler on_created method
class CustomHandler(PatternMatchingEventHandler):
    # on_created event handler
    def on_created(self, event):
        # Display the file created event TODO
        logging.info("File created: % s", event.src_path)
        # add training to celery queue

    # on_modified event handler
    def on_modified(self, event):
        # Display the file modified event TODO
        logging.info("File modified: % s", event.src_path)

    # on_deleted event handler
    def on_deleted(self, event):
        # Display the file deleted event TODO
        logging.info("File deleted: % s", event.src_path)

    # on_moved event handler
    def on_moved(self, event):
        # Display the file moved event TODO
        logging.info("File moved: % s to % s", event.src_path, event.dest_path)


if __name__ == "__main__":
    # Set the format for logging info
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Set format for displaying path
    path = "inputs/"
    # path = "/tmp/inputs"

# TODO Fix Path. Not possible to get notifications form mounted path on windows host.

    # initialize regex event handler to ignore ERROR.json files
    event_handler = CustomHandler(
        ignore_patterns=["ERROR.json"], ignore_directories=True)

    # Initialize Observer
    # observer = Observer()
    observer = polling.PollingObserver()
    observer.schedule(event_handler, path, recursive=True)

    # Start the observer
    observer.start()
    try:
        while True:
            # Set the thread sleep time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
