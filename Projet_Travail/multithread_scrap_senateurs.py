# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:33:36 2020

@author: adrie
"""

from multiprocessing.dummy import Pool as ThreadPool

#Imports : 
import pandas as pd
import numpy as np
import datetime
import os
#Twitter intelligence tool to scrap data from twitter : 
#https://github.com/twintproject/twint
import twint
#Pour gérer une erreur : 
#Voir https://github.com/twintproject/twint/issues/166
import nest_asyncio
nest_asyncio.apply()


def scrap_twint_multi(user):
    
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    #compteurs :
    u_connu=0
    u_inconnu=0
    
    u_id=user
#Boucle for sur tous les ids utilisateurs de la base :
    #Définir le nom du fichier :
    #filename='Donnees/tweets_pol/'+str(u_id)+'.csv'
    filename='C:/Users/adrie/Documents/Centrale/G3/DATA/Projet integ/Projet_Travail/Donnees/twitter_parlementaires/tweets_senateurs/'+str(u_id)+'.csv'
    #Lancer la configuration : 
    c = twint.Config()
    #On défini l'user dans twint : 
    c.User_id=u_id
    
    
#Si nous avons déjà un fichier csv des tweets, nous n’analysons pas le compte Tweeter de la personnalité
    if os.path.exists(filename):
         print('déjà présent :', u_id )
         u_connu+=1
 #Sinon, nous récuperons la liste des tweets
    else:
        c.Output = filename
        #Langage :
        c.Lang='fr'
 #Nous ne souhaitons pas voir les tweets apparaitre dans la console Python
        c.Hide_output = True  
 #Nous n'affichons pas les hashtags
        c.Show_hashtags = False     
 #On stock dans un csv :
        c.Store_csv = True
#On scrappe seulement les tweets les plus populaires :
        #c.Popular_tweets=True
#On limite le nombre de tweets par utilisateur : (par tranches de 20)
        #c.Limit=5
#On exclu les tweets avec des liens :
        c.Links='exclude'

#Lancement de la recherche
#Test pour savoir si un utilisateur est connu ou non :
        try:
            print('scrapping : ', u_id)
            twint.run.Search(c)
            u_connu+=1
        except TypeError:
            print('utilisateur inconnu : ', u_id)
            u_inconnu+=1
            pass

    print("Utilisateurs connus : ", u_connu)
    print("Utilisateurs inconnus : ", u_inconnu)
    return True


#%%
df_deputes=pd.read_csv('C:/Users/adrie/Documents/Centrale/G3/DATA/Projet integ/Projet_Travail/Donnees/twitter_parlementaires/deputes.csv')
df_senateurs=pd.read_csv('C:/Users/adrie/Documents/Centrale/G3/DATA/Projet integ/Projet_Travail/Donnees/twitter_parlementaires/senateurs.csv')


user_list=np.array(df_senateurs.twitter_id)
user_list=np.array(user_list)


# Make the Pool of workers
pool = ThreadPool(5)
# Open the URLs in their own threads
# and return the results
results = pool.map(scrap_twint_multi, user_list)

# Close the pool and wait for the work to finish
pool.close()
pool.join()