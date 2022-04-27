import os
import logging
import argparse
from sink import StdOutSink
from process import NERProcessor
from subscriber import PubsubSubcriber


logging.basicConfig(
    format='{%(asctime)s %(filename)s:%(lineno)d} %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscribtion_id", default="ner-analyser-test")
    parser.add_argument("--project_id", default="exalted-bonus-197703")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    project_id = args.project_id
    subscribtion_id = args.subscribtion_id

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pubsub\src\pubsub.json"

    sink = StdOutSink()
    proc = NERProcessor(sink=sink)
    subs = PubsubSubcriber(project_id, subscribtion_id, proc)

    subs.start()
    subs.stop()
