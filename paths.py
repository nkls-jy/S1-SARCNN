import os 

#train_dir = "/home/niklas/Documents/traindata_small"
train_dir = "/home/niklas/Documents/train_tiles"
valid_dir = ""
test_dir = "/home/niklas/Documents/test_tiles"

# create testfiles
list_testfiles = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.endswith(".tif")]

