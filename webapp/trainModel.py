from webapp.modules import User, GroceryList
from webapp.precommender.knetworks import knetworks
import numpy as np
import os
import torch

device = torch.device("cuda")

filename = os.getcwd() + "/webapp/" + "allproducts.txt" #may not work for windows
with open(filename, "r") as file:
	f = file.read()
	products = f.split("\n")

data = [[gr_list.items.split(',') for gr_list in user.created] for user in User.query.all()]
vectors = [[np.zeros(len(products), dtype=np.int) for gr_list in user.created] for user in User.query.all()]
for i,x in enumerate(data):
    for j,y in enumerate(x):
        for k,f in enumerate(products):
            if f in y:
                vectors[i][j][k] = 1

vectors = np.array(vectors)
knet = knetworks(2, vectors, len(products),device)
knet.fit(7)
#knet.train(50,10)
#knet.save(os.getcwd() + "/webapp/precommender/" + "saves")
knet.load(os.getcwd() + "/webapp/precommender/" + "saves")