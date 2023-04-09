import csv
import io

stream = io.StringIO()

with open('data.txt', 'r') as f:
    data = f.read()
    for line in data.split('\n'):
        stream.write(line.replace('\t', ',') + '\n')

    stream.seek(0)
    reader = csv.DictReader(stream, delimiter=',')
    for row in reader:
        print(row)

