# Training files
#-----------------------------------
# classification.py - model training 
#   set_model.py - sets model and the loss function 
#   MedNISTDataset.py - inherits from torch.utils.data.Dataset with sligh modifications
#   getdata.py - reads data for training, validation and testing 
# Inference files
#-----------------------------------
# volume_quality.py - recieves volumes folder location, best model location and type, symetry and movement results file location, runs inference and updates the result file with the quality assesmet
# slice_to_png.py - recieves volumes folder location and creates slices folder for each volument with *.png for each slice
# evalulate.py - model evaluation
