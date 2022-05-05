import json


class Message:
    data = None
    attributes = None

    def __init__(self, data, attributes):
        self.data = data
        self.attributes = attributes

    def __str__(self) -> str:
        return "data: {} attributes {}".format(self.data, self.attributes)


class JsonCoder:
    def decode(self, message):
        try:
            temp = json.loads(message)
            if "data" in temp and "attributes" in temp:
                return Message(temp["data"], temp["attributes"])
            else:
                return None
        except:
            return None

    def encode(self, data, **kwargs):
        temp = {"data": data, "attributes": kwargs}
        return json.dumps(temp).encode("UTF-8")
