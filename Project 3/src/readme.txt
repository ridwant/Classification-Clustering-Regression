# I wiil be using 2 late days from my 5 late days for this assignment

# This project is done using python3. Please download and install python3 if not already installed 
python3: https://www.python.org/downloads/ 



# install these dependencies given beloww 

pip3 install numpy
pip3 install argparse
pip3 install pandas
pip3 install scikit-learn
pip3 install imblearn

# Specific instructions can be found here: 

numpy: https://numpy.org/install/
argparse: https://pypi.org/project/argparse/
pandas: https://pandas.pydata.org/docs/getting_started/install.html
scikit: https://scikit-learn.org/stable/install.html
imblearn: https://pypi.org/project/imblearn/


Or you can just Run: 
pip3 install -r requirements.txt

That will install all the dependencies

# Running Classifiers

Now to Run the Classifiers use this format:  

python3 classifiers.py classifierName

# classifierName is the name of the classifier to be used. It can be one of the following:

knn: Runs the K-Nearest Neighbour Classifier 
dt:  Runs the Decision Tree Classifier
rf: Runs the Random Forest Classifier
other - Runs the AdaBoost Classifier
all - Runs the KNN, Decision Tree and Random Forest Classifier

example command: python3 classifiers.py other 


# Running Regressors

Now to Run the Regressors use this format:  

python3 regressors.py regressorName

# regressorName is the name of the regressor to be used. It can be one of the following:

knn: Runs the K-Nearest Neighbour Regressor 
dt:  Runs the Decision Tree Regressor
rf: Runs the Random Forest Regressor
other - Runs the VotingRegressor
all - Runs the KNN Regressor, Decision Tree Regressor, Random Forest Regressor, and VotingRegressor

example command: python3 regressors.py knn