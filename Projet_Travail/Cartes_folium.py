# -*- coding: utf-8 -*-
"""
@author: adrien
"""
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
#Info sur heatmap et les plugins : https://python-visualization.github.io/folium/plugins.html

elec_data_ecolo= pd.read_csv('Donnees_clean/contributions_coordonnees_insee/data_ecolo_coord_insee.csv',error_bad_lines=False)
elec_data_serv_pub=pd.read_csv('Donnees_clean/contributions_coordonnees_insee/data_serv_pub_coord_insee.csv',error_bad_lines=False)
elec_data_fisc=pd.read_csv('Donnees_clean/contributions_coordonnees_insee/data_fisc_coord_insee.csv',error_bad_lines=False)
elec_data_dem=pd.read_csv('Donnees_clean/contributions_coordonnees_insee/data_dem_coord_insee.csv',error_bad_lines=False)


#On remplace les coordonnées GPS par les coordonnées GPS précédentes dans la colonne : 
#elec_data_ecolo['Coordonnees'].fillna(method='pad', inplace=True)
elec_data_ecolo['Coordonnees']=elec_data_ecolo['Coordonnees'].str.split(',')
elec_data_serv_pub['Coordonnees']=elec_data_serv_pub['Coordonnees'].str.split(',')
elec_data_fisc['Coordonnees']=elec_data_fisc['Coordonnees'].str.split(',')
elec_data_dem['Coordonnees']=elec_data_dem['Coordonnees'].str.split(',')


All_coord=pd.concat([elec_data_ecolo.Coordonnees, elec_data_serv_pub.Coordonnees, elec_data_fisc.Coordonnees, elec_data_dem.Coordonnees], ignore_index=True)

#On convertit en list pour pouvoir afficher avec folium :
#CP_contrib_folium=elec_data_ecolo['Coordonnees']
CP_contrib_folium=All_coord
#.tolist()




#creating the map :
m = folium.Map(location=[47.088615, 2.637424],zoom_start=6)
#Markers :
#for point in range(0,2000):
#    folium.Marker(CP_contrib_folium[point]).add_to(m)

#clusters of markers : 
marker_cluster = folium.plugins.MarkerCluster().add_to(m)
for point in CP_contrib_folium:
    folium.Marker(point).add_to(marker_cluster)
#HeatMap:
HeatMap(CP_contrib_folium,radius=7,blur=1).add_to(m)

m.save('test.html')