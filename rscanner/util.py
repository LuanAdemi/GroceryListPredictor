import string
import torch
import random

# create a list with all ascii letters
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# maps every ascii character to an individual number
def letterToIndex(letter):
    return all_letters.find(letter)

# creates a tensor resembeling a curtain character
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# creates a tensor resembeling a curtain word
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# retrives the catergory from the output tensor
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

# selects a random element from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]