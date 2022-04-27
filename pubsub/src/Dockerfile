FROM python:3.7

COPY requirements.txt /
RUN pip3 install -r requirements.txt
COPY models/ /

ADD . /app
WORKDIR /app
ENV GOOGLE_APPLICATION_CREDENTIALS "./pubsub.json"
CMD [ "python3", "main.py", "exalted-bonus-197703", "ner-analyzer", "raw-saver-production" ]