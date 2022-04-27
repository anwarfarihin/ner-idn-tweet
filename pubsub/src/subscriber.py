import logging
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1

from coder import Message


class PubsubSubcriber:

    def __init__(self, project_id, subscribtion_id, processor):
        self.project_id = project_id
        self.subscribtion_id = subscribtion_id
        self.processor = processor

    def start(self):
        subscriber = pubsub_v1.SubscriberClient()
        path = subscriber.subscription_path(
            self.project_id, self.subscribtion_id)

        stream_future = subscriber.subscribe(path, callback=self.callback)
        logging.info("Subscribing from {}".format(self.subscribtion_id))
        with subscriber:
            try:
                stream_future.result()
            except TimeoutError:
                stream_future.cancel()
                stream_future.result()

    def callback(self, message):
        msg = Message(message.data, message.attributes)
        result = self.processor.process(msg)
        # if result:
        #     message.ack()
