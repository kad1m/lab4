import csv
import io


class ReadFile:

    def __init__(self, path: str):
        self.path = path

    def get_data(self) -> list:
        stream = io.StringIO()
        with open(self.path, 'r') as f:
            data = f.read()
            for line in data.split('\n'):
                stream.write(line.replace('\t', ',') + '\n')
            stream.seek(0)
            reader = csv.DictReader(stream, delimiter=',')
            data = [row for row in reader]
            return data

