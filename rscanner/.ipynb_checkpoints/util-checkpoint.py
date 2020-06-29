import string
import torch
import random

# create a list with all ascii letters
all_letters = string.ascii_letters + " .,;'" + "äÄüÜöÖ"
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
def lineToTensor(line, dim=-1):
    if dim == -1:
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][letterToIndex(letter)] = 1
    else:
        assert(dim >= len(line)),"The tensor dimension needs to be bigger the the line length!"
        tensor = torch.zeros(dim, 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def tensorToLine(tensor):
    tensor = tensor.numpy()
    line = ""
    for sub in tensor:
        for li, n in enumerate(sub):
            if n:
                line += all_letters[li]
                
    return line
            

# retrives the catergory from the output tensor
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

# selects a random element from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]