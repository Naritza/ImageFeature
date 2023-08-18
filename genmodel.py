import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

carvectors_train = pickle.load(open('carvectors_train.pkl', 'rb'))
carvectors_test = pickle.load(open('carvectors_test.pkl', 'rb'))


X_train_data = [carvectors_train[0:8100] for  carvectors_train in carvectors_train]
X_test_data = [carvectors_test[0:8100] for carvectors_test in carvectors_test]

Y_train_data = [carvectors_train[-1] for carvectors_train in carvectors_train]
Y_test_data = [carvectors_test[-1] for carvectors_test in carvectors_test]

LE = LabelEncoder()
new_y_train= LE.fit_transform(Y_train_data)
new_y_test= LE.fit_transform(Y_test_data)
# //////////////////
# สร้าง object จาก model DecisionTree
clf = DecisionTreeClassifier(random_state=42)

# สร้าง object จาก modelXGBoost
# model = xgb.XGBClassifier(objective="multi:softmax",num_class=len(y_labelNum_train),random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
      
# ทำการรวม 2โมเดล ที่สร้างไว้มารวมด้วยกัน 
ensemble_model = VotingClassifier(estimators=[('DecisionTree', clf), ('RandomForest', model)], voting='hard',weights=[1,2])

# .fit() X_train_data เป็นการให้โมเดลมันเรียนรู้ข้อมูลที่เหมาะสมกับข้อมูล ส่วน Y_train_data เป็นคำตอบที่ควรจะได้จากการเรียนรู้
ensemble_model.fit(X_train_data, new_y_train) 

# ใช้ข้อมูลทดสอบ X_test_data เพื่อไว้ทำนายผลลัพธ์ที่ได้จากโมเดลนี้ของข้อมูลทดสอบ
y_pred = ensemble_model.predict(X_test_data) 

# y_labelNum_test เป็นข้อมูลที่ถูกต้องจริงของข้อมูลทดสอบ ส่วน y_pred เป็นข้อมูลคำนวณความแม่นยำของโมเดลที่ได้นายผลลัพธ์จากข้อมูลทดสอบ
# คือมันจะทำการเปรียบเทียบ โดยเอาข้อมูลที่ทำนาย y_pred ที่ได้จากการ predict ข้อมูลทดสอบ โดยเปรียบเทียบว่าจำนวนคำตอบที่ถูกต้องที่โมเดลทำนายตรงกับ
# y_labelNum_test เป็นกี่เปอร์เซ็นต์ของจำนวนข้อมูลทดสอบ
accuracy = accuracy_score(new_y_test, y_pred)
print("Accuracy: ", accuracy*100)
# /////////////

# # Train a classification model
# model = DecisionTreeClassifier()
# model.fit(X_train_data, Y_train_data)

# # Make predictions
# y_pred = model.predict(X_test_data)

# # Calculate confusion matrix
# cm = confusion_matrix(Y_test_data, y_pred)

# # Calculate accuracy from confusion matrix
# accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# Print accuracy
# print("Accuracy:", accuracy)
# ///////////////
# The confusion matrix method provides deeper insights into different types of predictions and misclassifications, allowing you to analyze your model's performance in more detail.
# Feature scaling (optional)

# ////////////
# from sklearn import metrics
# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier()

# # Train Decision Tree Classifer
# clf = clf.fit(X_train_data,new_y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(new_y_test)

# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(new_y_test, y_pred))


# //////////////
# Train a Decision Tree Classifier
# model = DecisionTreeClassifier(max_depth=None, random_state=42)
# model.fit(X_train_data, new_y_train)

# # Make predictions
# y_pred = model.predict(X_test_data)

# # Calculate accuracy
# accuracy = accuracy_score(new_y_test, y_pred)

# # Print accuracy
# print("Accuracy:", accuracy)

# model = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
# model.fit(X_train_data, new_y_train)

# # Make predictions
# y_pred = model.predict(X_test_data)

# # Calculate accuracy
# accuracy = accuracy_score(new_y_test, y_pred)

# # Print accuracy
# print("Accuracy:", accuracy)
# # /////////
# clf_gini = DecisionTreeClassifier(criterion='gini') #สร้างต้นไม้ตัดสินใจ
# clf_gini.fit(X_train_data,new_y_train) #เรียนรู้ข้อมูล
# y_model_prediction = clf_gini.predict(X_test_data)
# print(accuracy_score(new_y_test,y_model_prediction)*100)
# print(confusion_matrix(new_y_test,y_model_prediction))

# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# import pickle
# lm = LinearRegression(fit_intercept=True)
# lm.fit(X_train_data,new_y_train)

# linear_reg_model = 'linear_reg_model.pkl'
# pickle.dump(lm,open(linear_reg_model,'wb'))
# print('save model : done')

# rm = RandomForestRegressor(max_depth = 2,random_state = 0)
# rm.fit(X_train_data,new_y_train)

# rf_reg_model = 'randomforest_model.pkl'
# pickle.dump(rm,open(rf_reg_model,'wb'))
# print('save model:done')


clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train_data, new_y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_data)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(new_y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")