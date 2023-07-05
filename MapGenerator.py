import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

# Cargar los datos etiquetados desde el archivo CSV
dataframe = pd.read_csv(r'labeledcountries.csv')


# Cargar el mapa mundial
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Unir los datos etiquetados con el mapa mundial en base al nombre del país
world = world.merge(dataframe, left_on='name', right_on='Country')

# Mostrar el mapa coloreado por las etiquetas
fig, ax = plt.subplots(figsize=(12, 8))
world.plot(column='Label', categorical=True, legend=True, ax=ax)
ax.set_axis_off()

plt.show()
