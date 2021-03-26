import sys, os
import numpy as np
import pandas as pd
import pylab as plt
from sklearn.ensemble import RandomForestClassifier

#os.chdir('C:\\Users\\wiggsj\\Documents\\Practice Codes\\Python-Projects\\Titanic - Machine Learning from Disaster')

train_data = pd.read_csv('data/train.csv')
test_data =  pd.read_csv('data/test.csv')

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
#output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

#don't ket this destory others' work.
#also let...

#print(train_data["Name"].unique())
train_data["Unique_names"] = train_data["Name"].unique().astype(str) # Convert data to str
first_names = train_data.Unique_names.apply(lambda name: name.split('.'))
train_data["forenames"] = np.array([name[1] for name in first_names]) # Take first letter

first_name_split = train_data.forenames.apply(lambda name: name.split())
nickname=np.empty_like(train_data.Unique_names)
for i in range(len(first_name_split)): nickname[i] = first_name_split[i][0]
new_nicknames = [s.replace("(", "") for s in nickname]

new_nicknames = [s.replace(")", "") for s in new_nicknames]
print(new_nicknames)

pd.Series(new_nicknames).value_counts().plot(kind='bar')
import pylab as plt
plt.tick_params(labelsize=4, rotation=90)
#plt.show()
print(pd.Series(new_nicknames).value_counts())

#print(first_name)

oneline_first_names = train_data.Name.apply(lambda name: name.split('.')[1].split()[0].strip()).value_counts()
print(oneline_first_names)
