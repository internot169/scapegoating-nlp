import sys
import csv
import datadir

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

reader = csv.reader(open(r"" + datadir.ATK_DIR, 'r', encoding='utf-8'))
reuters_writer = csv.writer(open(r"" + datadir.REUTERS_DIR, 'w', encoding='utf-8'))
cnn_writer = csv.writer(open(r"" + datadir.CNN_DIR, 'w', encoding='utf-8'))
fox_writer = csv.writer(open(r"" + datadir.FOX_DIR, 'w', encoding='utf-8'))
nytimes_writer = csv.writer(open(r"" + datadir.NYTIMES_DIR, 'w', encoding='utf-8'))

def contains(input, keywords):
    for keyword in keywords:
        if (keyword not in input):
            return False
    return True

for row in reader:
    if(contains(row[6], datadir.KEYWORDS)):
        if (row[9] == 'Reuters'):
            reuters_writer.writerow(row)
        elif (row[9] == 'CNN'):
            cnn_writer.writerow(row)
        elif (row[9] == 'Fox News'):
            fox_writer.writerow(row)
        elif (row[9] == 'The New York Times'):
            nytimes_writer.writerow(row)
