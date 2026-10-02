import pandas as pd
pd.set_option('display.width', 120)

df = pd.read_csv('data/Mall_Customers.csv')
df.columns = ['CustomerID','Gender','Age','AnnualIncome','SpendingScore']
print("Shape:", df.shape)
print(df.dtypes)
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nGender counts:\n", df['Gender'].value_counts())
print("\nDuplicate rows:", df.duplicated().sum())
df.to_csv('data/mall_customers_clean.csv', index=False)
