import json


class ConfigParser:
    @staticmethod
    def read(config_file):
        return json.load(open(config_file))

    @staticmethod
    def parse_configs(*files):
        res = []
        for file_ in files:
            res.append(ConfigParser.read(file_))
        return res
