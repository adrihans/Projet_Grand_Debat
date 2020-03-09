# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:52:36 2020

@author: adrie
"""

"""
Fichier Affichage_stats.py pour définir toutes les fonctions nécessaires à l'affichage des statistiques 
liées aux contributions. 

"""

#Imports:
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from textwrap import wrap
import seaborn as sns
#Palettes : https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/





"""
Fonction pour afficher les tailles des titres:
"""
def Len_titles(df_list, themes):
    """
    fonction Len_titles pour afficher la taille des titres des contributions
    @Inputs: 
        df_list
        themes
    @Outputs:
        None
    """
    #Concaténation des titres de tous les thèmes:
    All_contrib_title=pd.concat([df.title for df in df_list], ignore_index=True)
    All_contrib_title=All_contrib_title.astype('str')
    lengths=[]
    for i in All_contrib_title:
        lengths.append(len(i))
    fig, ax=plt.subplots(figsize=(15,5))
    ax = sns.countplot(lengths)

    #Ne mettre que des xticks tous les 5 :
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('Taille des titres')
    plt.show()
    print('Taille moyenne des titres: {moy:.2f} caractères'.format(moy=np.mean(lengths)))
    
    for df, theme in zip(df_list,themes):
        titles= df.title.astype('str')
        lengths=[]
        for i in titles:
            lengths.append(len(i))
        fig, ax=plt.subplots(figsize=(15,5))
        ax = sns.countplot(lengths)
        #Ne mettre que des xticks tous les 5 :
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.title(theme)
        plt.show()
        print('Taille moyenne des titres: {moy:.2f} caractères'.format(moy=np.mean(lengths)))
    return None


"""
Fonction pour supprimer toutes les colonnes qui ne sont pas nécéssaires :
"""

def questions_columns(raw_contrib):
    #si ecolo : pas id 
    #les autres : id
    return raw_contrib.drop(['reference', 'title', 'createdAt', 'publishedAt', 
                             'updatedAt','trashed', 'trashedStatus', 'authorId', 
                             'authorType', 'authorZipCode'],axis=1)

def stat_lengths(raw_contrib):
    lengths=[]
    for i in raw_contrib:
        for rep in raw_contrib[i].dropna().astype('str'):
            lengths.append(len(rep))
    
    fig, ax=plt.subplots(figsize=(15,5))
    plt.title('taille de chaque réponse aux questions : ')
    plt.hist(lengths,bins=70, range=(0, 500))
    plt.show()
    print('Taille moyenne des réponses aux questions: {moy:.2f}'.format(moy=np.mean(lengths)))
    k=280
    count_280 = len([i for i in lengths if i > k])/len(lengths)
    k=140
    count_140 = len([i for i in lengths if i > k])/len(lengths)
    k=3
    count_3 = len([i for i in lengths if i == k])/len(lengths)
    print("Part des contributions plus grandes que la taille d'un tweet (140): {Part140:.2f}%".format(Part140=count_140*100))
    print("Part des contributions plus grandes que la taille d'un tweet (280): {Part280:.2f}%".format(Part280=count_280*100))
    print("Part des contributions de la taille de 3 caractères: {count3:.2f}%".format(count3=count_3*100))
    return None


def stat_n_answers(raw_contrib):
    n_answers=raw_contrib.count(axis=1)
    fig,ax=plt.subplots(figsize=(15,5))
    ax=sns.countplot(n_answers)
    plt.title('Nombre de réponses pour chaque contributeur :')
    #--------Ajout des pourcentages : 
    ncount = len(n_answers)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    #----------

    plt.show()
    print('Nombre moyen de réponses : {moy:.2f}'.format(moy=np.mean(n_answers)))
    return None


"""
Compter les réponses uniques pour chaque question:
"""
def info_questions_fermees(raw_data):
    raw_data_questions=questions_columns(raw_data)
    for col_title in raw_data_questions:
        if raw_data_questions[col_title].unique().size<=5:
            print(col_title)
            print(raw_data_questions[col_title].unique().size)
            print(raw_data_questions[col_title].unique())
    return None
    
    
def Resultats_questions_fermees(raw_data):
    raw_data_questions=questions_columns(raw_data)
    for col_title in raw_data_questions:
        if raw_data_questions[col_title].unique().size<=5:
            fig, ax = plt.subplots(figsize=(18,6))
                      
            ax = sns.countplot(raw_data_questions[col_title], palette=sns.color_palette("muted"))
            #ax.set_yscale('log')
            ax.set_title(col_title)

            #------ Ajouter les densités : -----
            #https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
            ncount = len(raw_data_questions[col_title].dropna())
            for p in ax.patches:
                x=p.get_bbox().get_points()[:,0]
                y=p.get_bbox().get_points()[1,1]
                ax.annotate('{:.2f}%'.format(100.*y/ncount), (x.mean(), y), 
                        ha='center', va='bottom') # set the alignment of the text
            #----------
            plt.show()
    return None
            


#Fonction pour utiliser la liste:
#Titre 'Nombre de contributions par auteur et par thèmes, échelle logarithmique'
def n_contribs_par_auteur_list(df_list, themes):
    
    for df, theme in zip(df_list, themes):
        Tailles=df.groupby('authorId').size()
        fig, ax = plt.subplots(figsize=(15,5))
        ax=sns.countplot(Tailles)
        plt.title(theme)
        ax.set_yscale('log')
        plt.show()
        print('Maximum : ', Tailles.max())
        print('Nombre moyen : {moy:.2f}'.format(moy=Tailles.mean()))
        print("Part d'auteurs n'ayant fait qu'une seule contribution : {percent:.2f} % ".format(percent=len([i for i in Tailles if i == 1])/len(Tailles)*100) )
    return None

#Fonction pour afficher les types d'auteur:
def Types_auteurs(df_list, themes):
    #On fait une concaténation des types d'auteur de chaque thème:
    Types=pd.concat([df.authorType for df in df_list], ignore_index=True)
    #Initialisation de la figure:
    fig, ax = plt.subplots(figsize=(18,6))
    ax = sns.countplot(Types, palette=sns.color_palette("muted"))
    ax.set_yscale('log')
    ax.set_title("Type des auteurs pour tous les thèmes - échelle logarithmique")
    
    plt.show()


    for df, theme in zip(df_list, themes):
        Types= df.authorType
        fig, ax = plt.subplots(figsize=(18,6))
        ax = sns.countplot(Types, palette=sns.color_palette("muted"))
        ax.set_yscale('log')
        ax.set_title(theme)
        plt.show()
    return None

#Fonction pour afficher les auteurs ayant contribué dans plusieurs thèmes:
"""
Faire un dictionnaire pour chaque thème 

Associer chaque colonne au nombre de contribution pour chaque thème 

faire un count() pour n'avoir que les valeurs non nan 
"""
    
def dict_n_contribs(rawData):
    rawData=rawData.groupby('authorId').count()
    #informations importantes :
    author_n_contribs = pd.DataFrame({'authorId':rawData.index, 'n_contribs':rawData['title']})
    #Dictionnaire : 
    authors_n_contribs_d = author_n_contribs.set_index('authorId').to_dict()['n_contribs']
    return authors_n_contribs_d


def auteurs_plusieurs_themes(df_list, themes):
    All_authors=pd.concat([df.authorId for df in df_list], ignore_index=True)
    All_authors=All_authors.astype('str')
    print('nombre total de contributions : ', All_authors.shape[0])
    print("nombre d'auteurs uniques : ", All_authors.unique().shape[0])
    df_authors_theme=pd.DataFrame(data=All_authors.unique(), columns=['uniqueAuthorId'])
    
    for df, theme in zip(df_list,themes):
        df_authors_theme[theme] = df_authors_theme['uniqueAuthorId'].map(dict_n_contribs(df))
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax = sns.countplot(df_authors_theme.drop(['uniqueAuthorId'],axis=1).count(axis=1))
    
    #--------Ajout des pourcentages : 
    ncount = len(df_authors_theme.drop(['uniqueAuthorId'],axis=1))
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.2f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    #----------
    
    plt.title('Nombre de contribution à des themes differents par auteur : ')
    plt.show()
    return None



#Fonction pour obtenir les questions liées à chaque thème:
def questions_themes(df_list, themes, col_common):
    questions = pd.concat([pd.DataFrame({'old_name':df_list[i].columns,
                                         'df_id':i,
                                         'theme':themes[i]}) for i in range(len(df_list))])
    #On enlève les colonnes qui sont communes à chaque thème (et qui ne sont donc pas des questions)
    questions = questions[-questions["old_name"].isin(col_common)].reset_index(drop=True)
    #On donne un nouveau nom à chaque question:
    questions = questions.assign(new_name=(pd.Series(
        ['Q{}'.format(i) for i in range(1, questions.shape[0] + 1)])))
    #On enlève la référence 'bizarre' au début du nom de chaque question pour ne garder que le texte pertinent:
    questions = questions.assign(question=pd.Series(
        [name.split(' - ')[0] if theme=='La transition écologique' else name.split(' - ')[1] for name, theme in zip(questions.old_name, questions.theme)]))
    
    
    
    
    # On renomme les questions grâce au dictionnaire:
    dict_rename = {old:new for old, new in zip(questions.old_name,questions.new_name)}
    for df in df_list:
        df.rename(columns=dict_rename,inplace=True)
        
        
    questions['nbrow'] = questions.apply(lambda g: df_list[g.df_id].shape[0], axis=1)
    questions['nbnnull'] = questions.apply(lambda g: df_list[g.df_id].loc[:,g.new_name]\
                                       .notnull().sum(), axis=1)
    questions['nbunique'] = questions.apply(lambda g: df_list[g.df_id].loc[:,g.new_name]\
                                            .nunique(), axis=1)
    
    questions['nnull_rate'] = questions.nbnnull/questions.nbrow * 100
    questions['unique_rate'] = questions.nbunique/questions.nbnnull * 100
    
    questions['closed'] = questions['nbunique'] <= 3

    return questions



#Affichage des résultats aux questions fermées :
# Countplot of questions_df
def countplot_qdf(df_list, questions_df, suptitle):
    n = questions_df.shape[0]
    
    # If there is nothing to plot, we stop here
    if n==0:
        return
    
    # Numbers of rows and cols in the subplots
    ncols = 3
    nrows = (n+3)//ncols
    fig,ax = plt.subplots(nrows, ncols, figsize=(25,6*nrows))
    fig.tight_layout(pad=9, w_pad=10, h_pad=7)
    fig.suptitle(suptitle, size=30, fontweight='bold')
    # Hide exceeding subplots
    for i in range(n, ncols*nrows):
        ax.flatten()[i].axis('off')
        
    # Countplot for each question
    for index, row in questions_df.iterrows():
        plt.sca(ax.flatten()[index])
        # We add the sort_values argument to always have the same order: Oui, Non...
        xlabels = df_list[row.df_id].loc[:,row.new_name]
        xlabels = xlabels.value_counts().index.sort_values(ascending=False)
        axi = sns.countplot(x=row.new_name,
                           data=df_list[row.df_id],
                           order = xlabels)
        # Wrap long questions into lines
        axi.set_title("\n".join(wrap(row.new_name + '. ' + row.question, 60)))
        axi.set_xlabel('')
        # We also set a wrap here (for one very long answer...)
        axi.set_xticklabels(["\n".join(wrap(s, 17)) for s in xlabels])
        axi.set_ylabel('Nombre de réponses')
        add_frequencies(axi, row.nbnnull)
    return None

# Add frequencies to a countplot
# Source: https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
def add_frequencies(ax, ncount):
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f} %'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom', size='small', color='black', weight='bold')
    return None

def questions_fermees_themes(df_list, questions, themes):
    # Plotting questions, grouped by theme
    for i in range(len(themes)):
        countplot_qdf(df_list,questions[(questions.closed) & (questions.df_id == i)].reset_index(), themes[i])
    return None




"""
Affichage des résultats aux élections:
"""

def resultats_elections(df_list_elec, themes_elec):
    candidats=['LE PEN.exp', 'MÉLENCHON.exp', 'MACRON.exp', 'FILLON.exp', 'DUPONT-AIGNAN.exp', 'LASSALLE.exp', 'HAMON.exp', 'ASSELINEAU.exp', 'POUTOU.exp', 'ARTHAUD.exp', 'CHEMINADE.exp']
    #candidats=['LE PEN', 'MÉLENCHON', 'MACRON', 'FILLON', 'DUPONT-AIGNAN', 'LASSALLE', 'HAMON', 'ASSELINEAU', 'POUTOU', 'ARTHAUD', 'CHEMINADE']
    
    print('par exprimés et en considérant tous les contributeurs:')
    #Pour tous les themes:
    df_All_themes=pd.concat([df[candidats] for df in df_list_elec], ignore_index=True)
    df = pd.melt(df_All_themes)
    
    df2=df.groupby(['variable'],as_index=False).agg('sum',axis=1)\
    .sort_values('value',ascending=False)
    
    plt.figure(figsize=(15,5))
    ax=sns.barplot(x=df2.variable, y=df2.value)
    ncount=df2.value.sum()
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # Pour ajuster l'alignement du texte 
    plt.title('Tous les themes')
    plt.show()
    
    #Pour chaque theme:    
    
    for df,theme in zip(df_list_elec, themes_elec):
        resultats_raw_int=df[candidats] 
        #/ df.shape[0]
        df = pd.melt(resultats_raw_int)
        df2=df.groupby(['variable'],as_index=False).agg('sum',axis=1)\
        .sort_values('value',ascending=False)
    
    
        plt.figure(figsize=(15,5))
        ax=sns.barplot(x=df2.variable, y=df2.value)
        ncount=df2.value.sum()
        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                    ha='center', va='bottom') # Pour ajuster l'alignement du texte 
        plt.title(theme)
        plt.show()
        
    print('En considérant chaque contributeur unique: ')
    candidats=['LE PEN.exp', 'MÉLENCHON.exp', 'MACRON.exp', 'FILLON.exp', 'DUPONT-AIGNAN.exp', 'LASSALLE.exp', 'HAMON.exp', 'ASSELINEAU.exp', 'POUTOU.exp', 'ARTHAUD.exp', 'CHEMINADE.exp']
    candidats.append('authorId')
    df_All_themes=pd.concat([df[candidats] for df in df_list_elec], ignore_index=True)
    df_All_themes=df_All_themes.groupby('authorId',as_index=False).first()
    df_All_themes.drop(['authorId'],axis=1, inplace=True)
    df = pd.melt(df_All_themes)
    
    df2=df.groupby(['variable'],as_index=False).agg('sum',axis=1)\
    .sort_values('value',ascending=False)
    
    plt.figure(figsize=(15,5))
    ax=sns.barplot(x=df2.variable, y=df2.value)
    ncount=df2.value.sum()
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # Pour ajuster l'alignement du texte 
    plt.title('Tous les themes')
    plt.show()
    
    #Pour chaque theme:    
    candidats=['LE PEN.exp', 'MÉLENCHON.exp', 'MACRON.exp', 'FILLON.exp', 'DUPONT-AIGNAN.exp', 'LASSALLE.exp', 'HAMON.exp', 'ASSELINEAU.exp', 'POUTOU.exp', 'ARTHAUD.exp', 'CHEMINADE.exp']
    for df,theme in zip(df_list_elec, themes_elec):
        resultats_raw_int=df.groupby('authorId').first()[candidats] 
        #/ df.shape[0]
        df = pd.melt(resultats_raw_int)
        df2=df.groupby(['variable'],as_index=False).agg('sum',axis=1)\
        .sort_values('value',ascending=False)
    
    
        plt.figure(figsize=(15,5))
        ax=sns.barplot(x=df2.variable, y=df2.value)
        ncount=df2.value.sum()
        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                    ha='center', va='bottom') # Pour ajuster l'alignement du texte 
        plt.title(theme)
        plt.show()
        
    return None


#%% Nombre de caractères pour les questions ouvertes:

#Fonction pour appeler la fonction suivante avec seulement les questions ouvertes
def questions_ouvertes_themes(df_list, questions, themes):
    lengths=[]
    # Plotting questions, grouped by theme
    for i in range(len(themes)):
        lengths.extend(countplot_nb_chars(df_list,questions[(questions.closed == False) & (questions.df_id == i)].reset_index(), themes[i]))
    #Plotting pour tous les thèmes:
    fig, ax=plt.subplots(figsize=(15,5))
    plt.title('Tous les thèmes')
    plt.hist(lengths,bins=70, range=(0, 500))
    plt.show()
    print('Taille moyenne des réponses aux questions: {moy:.2f}'.format(moy=np.mean(lengths)))
    k=280
    count_280 = len([i for i in lengths if i > k])/len(lengths)
    k=140
    count_140 = len([i for i in lengths if i > k])/len(lengths)
    print("Part des contributions plus grandes que la taille d'un tweet (140): {Part140:.2f}%".format(Part140=count_140*100))
    print("Part des contributions plus grandes que la taille d'un tweet (280): {Part280:.2f}%".format(Part280=count_280*100))
    
    return None

#Fonction pour connaitre le nombre de caractères de chaque réponse:
def countplot_nb_chars(df_list, questions_df, suptitle):
    # Countplot for each question
    for index, row in questions_df.iterrows():
        lengths=[]
        for rep in df_list[row.df_id].loc[:,row.new_name].dropna().astype('str'):
            lengths.append(len(rep))
            
    fig, ax=plt.subplots(figsize=(15,5))
    plt.title(suptitle)
    plt.hist(lengths,bins=70, range=(0, 500))
    plt.show()
    print('Taille moyenne des réponses aux questions: {moy:.2f}'.format(moy=np.mean(lengths)))
    k=280
    count_280 = len([i for i in lengths if i > k])/len(lengths)
    k=140
    count_140 = len([i for i in lengths if i > k])/len(lengths)
    print("Part des contributions plus grandes que la taille d'un tweet (140): {Part140:.2f}%".format(Part140=count_140*100))
    print("Part des contributions plus grandes que la taille d'un tweet (280): {Part280:.2f}%".format(Part280=count_280*100))
    return lengths

