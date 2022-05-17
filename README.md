# Applying Facial Recognition Techniques to Non-Facial Data
This project is accompanied by the paper "Applying Facial Recognition Techniques to Non-Facial Data" that can be found as a pdf in this repository. For more details about the results of the project, please refer to this paper.

This project attempts to see if facial recognition techniques (mainly semi-hard-triplet-loss) can be used on things other than faces, in this case images of cities. The dataset of images of cities can be found on Kaggle.com at the link https://www.kaggle.com/datasets/cursiv/cities-dataset-10.

There are 2 main tasks this project aims to evaluate.

## Task 1:
This compares the classification accuracy of a traditional CNN that uses categorical cross entropy loss on the 10 cities vs a CNN that uses semi hard triplet loss. The triplet loss model then uses a linear SVM to achieve classification.

## Task 2:
This compares the zero shot learning accuracy of both of the previously described models. This is done by training the models on 8 of the 10 cities, then using the 2 that were held out to test. Testing is done by feeding pairs of the held out cities to each model, then computing the L2 distance of the two embeddings outputted by the model. If that distance falls under a certain threshold, they are classified as the same, otherwise they are different. The accuracy is then computed from this to compare.

# How to Run
The file `build.py` can be used to build the actual models. However, for others convenience, these models were already built and saved. There are 8 pre_trained models that can be found in `saved_models_h5/`. 

Any one of these models can be easily evaluated with the testing script `test_model.py`. Simply run the test script with the path of the model (including 'saved_models_h5/) to receive accuracy scores. 

If you wish to test all 8 models, simply run `python test_all_models.py`. This script conveniently tests all of the models in the saved models directory.

*NOTE- if you attempt to train extra models and place them into the saved models directory, please follow the naming convention. Specifically, 'holdout' needs to be in the name if you are training on only 8 cities, and either 'baseline' or 'triploss' needs to be in the model name to specify what kind of model you are training. There are no promises made that new models you add will work with the test script.

#
If you have any questions or comments, feel free to reach out to jmandzak99@gmail.com