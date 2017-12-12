# Checking loss values with plt
import pickle


file = open("saveSegnetValues.p",'rb')
object_file = pickle.load(file)
loss_train, loss_test = object_file

print( loss_train == loss_test)