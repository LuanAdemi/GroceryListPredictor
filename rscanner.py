from rscanner.train import trainOnData
from rscanner.scan import scan, loadModel

import sys

if len(sys.argv) < 2:
    raise ValueError('Usage: rscanner <train|scan> (image)')

if sys.argv[1] == "train":
    trainOnData()
if sys.argv[1] == "scan":
    if len(sys.argv) < 3:
        raise ValueError('Usage: rscanner <train|scan> (image)')
    loadModel("rscanner/state_dicts/model.pt")
    print(scan(str(sys.argv[2])))