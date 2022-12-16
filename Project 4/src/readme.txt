# This project is done using python3. Please download and install python3 if not already installed 

python3: https://www.python.org/downloads/ 



# install these dependencies given beloww 

pip3 install numpy
pip3 install argparse
pip3 install pandas
pip3 install scikit-learn
pip3 install imblearn
pip3 install keras
pip3 install seaborn
pip3 install tensorflow

# Specific instructions can be found here: 

numpy: https://numpy.org/install/
argparse: https://pypi.org/project/argparse/
pandas: https://pandas.pydata.org/docs/getting_started/install.html
scikit: https://scikit-learn.org/stable/install.html
imblearn: https://pypi.org/project/imblearn/
keras: https://pypi.org/project/keras/
seaborn: https://pypi.org/project/seaborn/
tensorflow: https://www.tensorflow.org/

Or you can just Run: 
pip3 install -r requirements.txt

That will install all the dependencies including tensorflow

# I have also included the Jupyter Notebook File in case if you don't want to download the tensorflow in src folder


# Running Individual Neural Networks

To Run Individual Neural Networks on Test Data to see the prediction use this format:  

python3 neural_nets.py classifierName

# classifierName is the name of the classifier to be used. It can be one of the following:

mlp: Runs the MLPClassifier 
ksm:  Runs the Keras Sequential Model Classifier
rf: Runs the Random Forest Classifier
all - Runs All Classifier and shows the comparison plot

example command: python3 neural_nets.py rf 
