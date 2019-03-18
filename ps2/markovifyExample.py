import markovify
import re
from urllib import request

def markovifyProcess(text):

    # Build the model.
    text_model = markovify.Text(text)
    # Print three randomly-generated sentences
    for i in range(3):
        print(text_model.make_sentence())

if __name__=="__main__":
    url='http://www.gutenberg.org/files/59075/59075-0.txt'
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    text = re.sub(r'[^\w\s]', '', raw)
    text=text[500:1000]
    markovifyProcess(text)
    url='http://www.gutenberg.org/files/23294/23294-0.txt'
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    text = re.sub(r'[^\w\s]', '', raw)
    text = text[500:1000]
    markovifyProcess(text)
