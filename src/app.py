from utils import db_connect
engine = db_connect()

# your code here
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# Leer el CSV desde la carpeta data/raw
df = pd.read_csv('../data/raw/AB_NYC_2019.csv')

# Mostrar las primeras 5 filas de la tabla
df.head()

dimen = df.shape
print(dimen)

df.info()

# Eliminar duplicados

df.drop("host_id", axis = 1).duplicated().sum()


print(f"The number of duplicated Name records is: {df['name'].duplicated().sum()}")
print(f"The number of duplicated Host ID records is: {df['host_id'].duplicated().sum()}")
print(f"The number of duplicated ID records is: {df['id'].duplicated().sum()}")

if df.duplicated().sum():
    df = df.drop_duplicates()
print(df.shape)
df.head()

df.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)
df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Suponiendo que tu DataFrame se llama df y ya lo tienes cargado
fig, axis = plt.subplots(2, 3, figsize=(10, 7))

# Histogramas 
sns.histplot(ax = axis[0,0], data = df, x = "host_id")
sns.histplot(ax = axis[0,1], data = df, x = "neighbourhood_group").set_xticks([])
sns.histplot(ax = axis[0,2], data = df, x = "neighbourhood").set_xticks([])
sns.histplot(ax = axis[1,0], data = df, x = "room_type")
sns.histplot(ax = axis[1,1], data = df, x = "availability_365")
fig.delaxes(axis[1, 2])

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Configuramos la figura con 3 filas y 2 columnas
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15))

# Variable: Price
sns.histplot(df['price'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histograma de Price')
axes[0, 0].set_xlabel('Price')
axes[0, 0].set_ylabel('Frecuencia')

sns.boxplot(x=df['price'], ax=axes[0, 1])
axes[0, 1].set_title('Boxplot de Price')
axes[0, 1].set_xlabel('Price')

# Variable: Minimum Nights
sns.histplot(df['minimum_nights'], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histograma de Minimum Nights')
axes[1, 0].set_xlabel('Minimum Nights')
axes[1, 0].set_ylabel('Frecuencia')

sns.boxplot(x=df['minimum_nights'], ax=axes[1, 1])
axes[1, 1].set_title('Boxplot de Minimum Nights')
axes[1, 1].set_xlabel('Minimum Nights')

# Variable: Number of Reviews
sns.histplot(df['number_of_reviews'], bins=30, kde=True, ax=axes[2, 0])
axes[2, 0].set_title('Histograma de Number of Reviews')
axes[2, 0].set_xlabel('Number of Reviews')
axes[2, 0].set_ylabel('Frecuencia')

sns.boxplot(x=df['number_of_reviews'], ax=axes[2, 1])
axes[2, 1].set_title('Boxplot de Number of Reviews')
axes[2, 1].set_xlabel('Number of Reviews')

plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Gráfico de dispersión entre price, minimum_nights y number_of_reviews
fig, axis = plt.subplots(4, 2, figsize = (10, 16))

# Crear placas
sns.regplot(ax = axis[0, 0], data = df, x = "minimum_nights", y = "price")
sns.heatmap(df[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = df, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(df[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = df, x = "calculated_host_listings_count", y = "price").set(ylabel = None)
sns.heatmap(df[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])


plt.tight_layout()


plt.show()

fig, axis = plt.subplots(figsize = (5, 4))

sns.countplot(data = df, x = "room_type", hue = "neighbourhood_group")

plt.show()

# Factorizar los datos de tipo de habitación y barrio
df["room_type"] = pd.factorize(df["room_type"])[0]
df["neighbourhood_group"] = pd.factorize(df["neighbourhood_group"])[0]
df["neighbourhood"] = pd.factorize(df["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(df[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()

sns.pairplot(data = df)

df.describe()

fig, axes = plt.subplots(3, 3, figsize = (15, 15))

sns.boxplot(ax = axes[0, 0], data = df, y = "neighbourhood_group")
sns.boxplot(ax = axes[0, 1], data = df, y = "price")
sns.boxplot(ax = axes[0, 2], data = df, y = "minimum_nights")
sns.boxplot(ax = axes[1, 0], data = df, y = "number_of_reviews")
sns.boxplot(ax = axes[1, 1], data = df, y = "calculated_host_listings_count")
sns.boxplot(ax = axes[1, 2], data = df, y = "availability_365")
sns.boxplot(ax = axes[2, 0], data = df, y = "room_type")

plt.tight_layout()

plt.show()

# Estadísticas de price
price_stats = df["price"].describe()
price_stats

# IQR para Price

price_iqr = price_stats["75%"] - price_stats["25%"]
upper_limit = price_stats["75%"] + 1.5 * price_iqr
lower_limit = price_stats["25%"] - 1.5 * price_iqr

print(f"Los límites superior e inferior para encontrar valores atípicos son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(price_iqr, 2)}")

# Limpieza de outliers

df = df[df["price"] > 0]

count_0 = df[df["price"] == 0].shape[0]
count_1 = df[df["price"] == 1].shape[0]

print("Count of 0: ", count_0)
print("Count of 1: ", count_1)

nights_stats = df["minimum_nights"].describe()
nights_stats


# IQR for minimum_nights
nights_iqr = nights_stats["75%"] - nights_stats["25%"]

upper_limit = nights_stats["75%"] + 1.5 * nights_iqr
lower_limit = nights_stats["25%"] - 1.5 * nights_iqr

print(f"Los límites superior e inferior para encontrar valores atípicos son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(nights_iqr, 2)}")

# Limpieza de outliers

df = df[df["minimum_nights"] <= 15]

count_0 = df[df["minimum_nights"] == 0].shape[0]
count_1 = df[df["minimum_nights"] == 1].shape[0]
count_2 = df[df["minimum_nights"] == 2].shape[0]
count_3 = df[df["minimum_nights"] == 3].shape[0]
count_4 = df[df["minimum_nights"] == 4].shape[0]


print("Count of 0: ", count_0)
print("Count of 1: ", count_1)
print("Count of 2: ", count_2)
print("Count of 3: ", count_3)
print("Count of 4: ", count_4)


review_stats = df["number_of_reviews"].describe()
print(review_stats)

review_iqr = review_stats["75%"] - review_stats["25%"]

upper_limit = review_stats["75%"] + 1.5 * review_iqr
lower_limit = review_stats["25%"] - 1.5 * review_iqr

print(f"Los límites superior e inferior para encontrar valores atípicos son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(review_iqr, 2)}")


hostlist_stats = df["calculated_host_listings_count"].describe()
print(hostlist_stats)

hostlist_iqr = hostlist_stats["75%"] - hostlist_stats["25%"]

upper_limit = hostlist_stats["75%"] + 1.5 * hostlist_iqr
lower_limit = hostlist_stats["25%"] - 1.5 * hostlist_iqr

print(f"Los límites superior e inferior para encontrar valores atípicos son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(hostlist_iqr, 2)}")

count_04 = sum(1 for x in df["calculated_host_listings_count"] if x in range(0, 5))
count_1 = df[df["calculated_host_listings_count"] == 1].shape[0]
count_2 = df[df["calculated_host_listings_count"] == 2].shape[0]

print("Count of 0: ", count_04)
print("Count of 1: ", count_1)
print("Count of 2: ", count_2)

# Clean the outliers

df = df[df["calculated_host_listings_count"] > 4]

df.isnull().sum().sort_values(ascending = False)

from sklearn.preprocessing import MinMaxScaler

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(df[num_variables])
df_scal = pd.DataFrame(scal_features, index = df.index, columns = num_variables)
df_scal["price"] = df["price"]
df_scal.head()

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scal.drop("price", axis = 1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)