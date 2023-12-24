import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("./runs/8296627103/val_preds.csv")

y_true = df['y']
y_pred = df['preds']

# Raw data
dct = {
    'y_true': y_true,
    'y_pred': y_pred
}
df = pd.DataFrame(dct)
# Remove NaNs
df = df.dropna()
# Pearson product-moment correlation coefficients
y_true = df['y_true']
y_pred = df['y_pred']
cor = np.corrcoef(y_true, y_pred)[0][1]
# Means
mean_true = np.mean(y_true)
mean_pred = np.mean(y_pred)
# Population variances
var_true = np.var(y_true)
var_pred = np.var(y_pred)
# Population standard deviations
sd_true = np.std(y_true)
sd_pred = np.std(y_pred)
# Calculate CCC
numerator = 2 * cor * sd_true * sd_pred
denominator = var_true + var_pred + (mean_true - mean_pred)**2
ccc = numerator / denominator

r2 = r2_score(y_true, y_pred)

error = y_true - y_pred

plt.hist(error, bins=25)
print("error mean", np.mean(error))
print("error std", np.std(error))
plt.show()


print(ccc)
print(r2)