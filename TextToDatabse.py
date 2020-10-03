

from webapp import db, bcrypt
from webapp.modules import User, GroceryList
import os
import numpy as np
#user1 = User.query.filter_by(id=1).first()
for j in range(1,20):
    file = open(os.getcwd()+"/webapp/precommender/testdata/gu_"+str(j)+".txt")
    stri = file.read()
    lists = stri.split("\n")
    features = lists[0].split(",")
    gr_lists = [[int(float(a)) for a in l.split(",")] for l in lists[1:]]
    hashed_pw = bcrypt.generate_password_hash("SampleUser"+str(j)).decode("utf-8")
    user1 = User(username="SampleUser"+str(j),password=hashed_pw, email="sample@user"+str(j)+".com", no_lists=len(gr_lists))
    db.session.add(user1)
    filename = os.getcwd() + "/webapp/" + "allproducts2.txt" #may not work for windows
    with open(filename, "r") as file2:
        f2 = file2.read()
        products = f2.split("\n")

    for i,l in enumerate(gr_lists):
        items = ""
        cnt = 0
        for k,f in enumerate(products):
            if f in [st[0] + st[1:].lower() for st in features] and 1 == gr_lists[i][[st[0] + st[1:].lower() for st in features].index(f)]:
                items += f+","
                cnt += 1

        items = items[:-1]
        gr_l = GroceryList(name="MyList"+str(i+1), items=items, user=user1, num_items=cnt)
        db.session.add(gr_l)
    db.session.commit()
    file.close()