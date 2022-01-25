import logging
import json

import requests
from coder import JsonCoder

class StdOutSink:
    def publish(self, data, attributes):
        logging.info("data {} | attributes {}".format(data, attributes))

class NSQSink:
    def __init__(self, nsqd_host, topic_name, coder=JsonCoder()):
        self.nsqd_host = nsqd_host
        self.topic_name = topic_name
        self.coder = coder
        self.topic_url = "http://{}/pub?topic={}".format(nsqd_host, self.topic_name)
    
    def publish(self, data, **kwargs):
        data_str = json.dumps(data)
        message = self.coder.encode(data_str, **kwargs)
        requests.request("POST", self.topic_url, data=message)