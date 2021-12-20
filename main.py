from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
x = np.loadtxt('./data/ecoli/ecoli_data.txt')
y = np.loadtxt('./data/ecoli/ecoli_id.txt')

result = []
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    clf = TabNetClassifier(
        n_d=44,
        n_a=44
    )  #TabNetRegressor()
    clf.fit(
      x_train, y_train,
        max_epochs=800
    )
    preds = clf.predict(x_test)
    result.append(np.sum(preds == y_test) / y_test.shape[0] * 100)
    print(result[-1])



result = pd.DataFrame(result)
result.to_csv("./result/result.csv")