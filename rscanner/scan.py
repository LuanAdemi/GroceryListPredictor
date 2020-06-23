from rscanner.model import RNN
from rscanner.util import lineToTensor, categoryFromOutput

import torch
import string
import pytesseract
import re
import cv2

all_letters = string.ascii_letters + " .,;'" + "äÄüÜöÖ"
n_letters = len(all_letters)

n_hidden = 128

rnn = RNN(n_letters, n_hidden, 3)

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def loadModel(path):
    rnn.load_state_dict(torch.load(path))

def scan(filePath):
    gray_img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

    products = []

    kassenzettel = pytesseract.image_to_string(gray_img, lang='deu')

    # remove unwanted characters using regex
    regex = re.compile('[^a-zA-Z äÄüÜöÖ]')

    # crop the string using a keyword used by most receipts
    produkte = kassenzettel[:kassenzettel.upper().find("SUMME")]
    
    produkte = regex.sub('', produkte).upper()
    
    produkte = produkte.split()
    for i in range(len(produkte)):
        output = evaluate(lineToTensor(produkte[i]))
        category = categoryFromOutput(output)
        #print('%s (%d)' % (produkte[i], category))
        if category == 1:
            products.append(produkte[i])
        if category == 2:
            products[len(products)-1] += " " + produkte[i]

    return products