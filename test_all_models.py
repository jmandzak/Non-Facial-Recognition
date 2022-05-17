# this is a simple testing script to test the accuracy of all models so you don't have to

import os

# call all appropriate models
for model in os.listdir('saved_models_h5'):
    print('**************************************************************************************')
    print(model)
    file_name = 'saved_models_h5/' + str(model)
    os.system(f'python test_model.py {file_name}')
    print('**************************************************************************************\n\n')