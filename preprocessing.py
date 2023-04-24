import tarfile
import os


POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# os.makedirs(NEG_PATH)

# file = tarfile.open('lfw.tgz')
#
# file.extractall('./data')
#
# file.close()

for directory in os.listdir('data/lfw'):
    for file in os.listdir(os.path.join('data/lfw', directory)):
        EX_PATH = os.path.join('data/lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)
