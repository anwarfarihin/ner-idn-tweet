import logging
import argparse
from sink import StdOutSink
from process import LogProcessor
from subscriber import PubsubSubcriber


logging.basicConfig(
    format='{%(asctime)s %(filename)s:%(lineno)d} %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscribtion_id", default="user-saver-staging")
    parser.add_argument("--project_id", default="exalted-bonus-197703")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    project_id = args.project_id
    subscribtion_id = args.subscribtion_id

    sink = StdOutSink()
    proc = LogProcessor(sink=sink)
    subs = PubsubSubcriber(project_id, subscribtion_id, proc)
    
    subs.start()
    subs.stop()
