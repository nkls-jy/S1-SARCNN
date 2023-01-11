import os 

train_dir = ""
valid_dir = ""
test_dir = "/home/niklas/Documents/testdata_big_noisy"

# create testfiles
list_testfiles = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.endswith(".tif")]

