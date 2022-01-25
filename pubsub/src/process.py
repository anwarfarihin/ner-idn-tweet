from cmath import sin
import logging

class LogProcessor:

    def __init__(self, sink):
        self.sink = sink

    def process(self, message):
        self.sink.publish(message.data, message.attributes)
        return True