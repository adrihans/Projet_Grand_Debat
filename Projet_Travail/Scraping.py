# -*- coding: utf-8 -*-
"""
@author: adrien
"""
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


#%%
def scrap_twint(user_list):
    #compteurs :
    u_connu=0
    u_inconnu=0
    
    
    print(user_list)
#Boucle for sur tous les ids utilisateurs de la base :
    for u_id in user_list: 
        #Définir le nom du fichier :
        #filename='Donnees/tweets_pol/'+str(u_id)+'.csv'
        filename='C:/Users/adrie/Documents/Centrale/G3/DATA/Projet integ/Projet_Travail/Donnees/tweets_pol/'+str(u_id)+'.csv'
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
            c.Popular_tweets=True
    #On limite le nombre de tweets par utilisateur : (par tranches de 20)
            c.Limit=5
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
#%% ----TEST------
#Initialisation des users à scrapper :
#user_list=['1264925822','1265204928','1265310602']

#scrap_twint(user_list)

#Lire le fichier obtenu : 
#u_id=user_list[2]
#df=pd.read_csv('C:/Users/adrie/Documents/Centrale/G3/DATA/Projet integ/Projet_Travail/Donnees/tweets_pol/'+str(u_id)+'.csv', error_bad_lines=False)

#%%
#dfprofiles_clean=pd.read_csv('C:/Users/adrie/Documents/Centrale/G3/DATA/Projet integ/Projet_Travail/Donnees_clean/twitter/utilisateurs/profiles_annotations.csv')
#user_list=np.array(dfprofiles_clean['FROM_USER_ID'])

#On ne le fait que sur les 100 premiers comptes :
#user_list=user_list[:1500]

#scrap_twint(user_list)
