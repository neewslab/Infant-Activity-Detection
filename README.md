# Infant-Activity-Detection

High level Description: Here we develop a multi-sensor framework for high resolution classification of infant activities. This involves identifying various activities that occur before onset of walking, including crawling, cruising, sitting, falling, and standing. This is achieved using a system involving multiple accelerometer sensors placed on the infant’s body. The collected data is preprocessed and fed to an Artificial Neural Network module to classify the different activities. A hierarchical binary classification strategy is used in order to ensure fine-grain classification of the infant activities. 

To use the code:

1. Import all the required machine learning and input-output libraries
2. Define how many times the data needs to be shuffled for k-fold validation in parameter K
3. Load the dataset from the local directory as a csv file
4. Shuffle the dataset for making it suitable for k-fold validation
5. Define the NN model with all its parameters, including loss function, optimizer, evaluation metric
6. Compile and run the model
7. Find the confusion matrix and accuracy using variables true_p and false_p
