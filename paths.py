import os 

#train_dir = "/home/niklas/Documents/SLC_train_tiles"
train_dir = "/home/niklas/Documents/Training_iterations_test/train_tiles"
#train_dir = "/home/niklas/Documents/Experiment_weights"
#train_dir = "/home/niklas/Documents/GRD_train_tiles"
#train_dir = "/home/niklas/Documents/traindata_big_cleaned"
valid_dir = ""
#test_dir = "/home/niklas/Documents/weight_test"
test_dir = "/home/niklas/Documents/Training_iterations_test/test_tiles"
#test_dir = "/home/niklas/Documents/dataForDL_Filtering/GRD_post"
#test_dir = "/home/niklas/Documents/SLC_test_tiles"
#test_dir = "/home/niklas/Documents/GRD_test_tiles"

# create testfiles
list_testfiles = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.endswith(".tif")]

