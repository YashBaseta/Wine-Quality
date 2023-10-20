{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "dd1c5606",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "from warnings import filterwarnings\n",
    "filterwarnings(action='ignore')\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "ae5faa7a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
      "0            7.4              0.70         0.00             1.9      0.076   \n",
      "1            7.8              0.88         0.00             2.6      0.098   \n",
      "2            7.8              0.76         0.04             2.3      0.092   \n",
      "3           11.2              0.28         0.56             1.9      0.075   \n",
      "4            7.4              0.70         0.00             1.9      0.076   \n",
      "\n",
      "   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
      "0                 11.0                  34.0   0.9978  3.51       0.56   \n",
      "1                 25.0                  67.0   0.9968  3.20       0.68   \n",
      "2                 15.0                  54.0   0.9970  3.26       0.65   \n",
      "3                 17.0                  60.0   0.9980  3.16       0.58   \n",
      "4                 11.0                  34.0   0.9978  3.51       0.56   \n",
      "\n",
      "   alcohol  \n",
      "0      9.4  \n",
      "1      9.8  \n",
      "2      9.8  \n",
      "3      9.8  \n",
      "4      9.4  \n",
      "0    0\n",
      "1    0\n",
      "2    0\n",
      "3    0\n",
      "4    0\n",
      "Name: quality, dtype: int64\n",
      "(1599,) (1279,) (320,)\n",
      "Accuracy :  0.93125\n",
      "[0]\n",
      "Bad Quality Wine\n"
     ]
    }
   ],
   "source": [
    "# loading the dataset \n",
    "wine_dataset = pd.read_csv('winequality-red.csv')\n",
    "\n",
    "# separate the data and Label\n",
    "A = wine_dataset.drop('quality',axis=1)\n",
    "print(A.head())\n",
    "\n",
    "B = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)\n",
    "print(B.head())\n",
    "\n",
    "# Train & Test Split\n",
    "A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=3)\n",
    "print(B.shape, B_train.shape, B_test.shape)\n",
    "\n",
    "\n",
    "#Traing model using Random Forest Classifier\n",
    "model = RandomForestClassifier()\n",
    "model.fit(X_train, Y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed5059b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# accuracy on test data\n",
    "A_test_prediction = model.predict(A_test)\n",
    "test_data_accuracy = accuracy_score(A_test_prediction, B_test)\n",
    "print('Accuracy : ', test_data_accuracy)\n",
    "\n",
    "input_data = (4.25,2.5,0.99,12.1,18.71,1.0,22.0,1.9978,10.35,2.8,14.5)\n",
    "\n",
    "# changing the input data \n",
    "input_data_as_numpy_array = np.asarray(input_data)\n",
    "\n",
    "# reshape the data as we are predicting the label for only one instance\n",
    "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
    "\n",
    "prediction = model.predict(input_data_reshaped)\n",
    "print(prediction)\n",
    "\n",
    "if (prediction[0]==1):\n",
    "  print('Good Quality Wine')\n",
    "else:\n",
    "  print('Bad Quality Wine')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
