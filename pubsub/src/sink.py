import logging
import json

import requests
from coder import JsonCoder
from google.cloud.pubsublite.cloudpubsub import PublisherClient


class StdOutSink:
    def publish(self, data, attributes):
        logging.info("data {} | attributes {}".format(data, attributes))


class NSQSink:
    def __init__(self, nsqd_host, topic_name, coder=JsonCoder()):
        self.nsqd_host = nsqd_host
        self.topic_name = topic_name
        self.coder = coder
        self.topic_url = "http://{}/pub?topic={}".format(
            nsqd_host, self.topic_name)

    def publish(self, data, **kwargs):
        data_str = json.dumps(data)
        message = self.coder.encode(data_str, **kwargs)
        requests.request("POST", self.topic_url, data=message)


class NERSink:
    def __init__(self, topic_name):
        self.topic_name = topic_name

    def publish(self, data):
        with PublisherClient() as publisher_client:
            api_future = publisher_client.publish(
                self.topic_name,
                json.dumps(data).encode("utf-8")
            )
            message_id = api_future.result()
