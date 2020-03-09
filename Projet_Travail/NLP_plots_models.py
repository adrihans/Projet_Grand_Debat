# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:13:38 2020

@author: adrie
"""
#%% Imports :

import pandas as pd
import numpy as np
import os
import re
import unidecode
import spacy.lang.fr
import spacy
import seaborn as sns


from sklearn.metrics import roc_auc_score
from sklearn import preprocessing



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF

from sklearn.feature_selection import chi2


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


import matplotlib.pyplot as plt
from wordcloud import WordCloud


#%% General :

def partis_pie_chart(data_partis):
    """
    fonction partis_pie_chart ayant pour but d'afficher le pie chart des différentes classes.
    @Inputs:
        - data_partis
    @Outputs:
        - None
    """
    #https://stackoverflow.com/questions/38337918/plot-pie-chart-and-table-of-pandas-dataframe
    fig=plt.figure(figsize=(7,7))
    data_partis.value_counts().plot(kind='pie',autopct='%1.1f%%',startangle=90, shadow=False, legend = False, fontsize=10)
    plt.show()
    return None

#%% Preprocessing :
#https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90



def clean_tweets(tweet):
    """
    fonction clean_tweets pour nettoyer les tweets et les textes n'ayant pas besoin de lemmatisation
    @Input:
        tweet, le texte à pré-traiter
    @output:
        tweet, le texte pré-traité
    """
    STOPWORDS = spacy.lang.fr.stop_words.STOP_WORDS
    #Gestion des accents :
    tweet=unidecode.unidecode(tweet)
    #Lowercases :
    tweet = tweet.lower()
    #Stopwords :
    tweet = ' '.join(word for word in tweet.split() if word not in STOPWORDS) 
    #Mention :
    tweet=re.sub(r'@[A-Za-z0-9]+','',tweet)
    #URL Links :
    tweet=re.sub('https?://[A-Za-z0-9./]+','',tweet)
    #Hashtags/numbers-non charcater :
    tweet=re.sub("[^a-zA-Z]", " ", tweet)  
    return tweet



#Lemmatization :
global spacy_nlp
spacy_nlp = spacy.load('fr_core_news_sm')
def clean_text_lemma(tweet):
    """
    Fonction clean_text_lemma utilisée pour le pré-traitement des textes lorsqu'une
    lemmatisation est nécessaire est justifiée.
    @Inputs:
        tweet (str), le texte à pré-traiter
    @Outputs:
        tweet (str), le texte pré-traité
    """
    #Import des stop words:
    STOPWORDS = spacy.lang.fr.stop_words.STOP_WORDS
    #Gestion des accents :
    tweet=unidecode.unidecode(tweet)
    #Lowercases :
    tweet = tweet.lower()
    #Stopwords :
    tweet = ' '.join(word for word in tweet.split() if word not in STOPWORDS) 

    #Mention :
    tweet=re.sub(r'@[A-Za-z0-9]+','',tweet)
    #URL Links :
    tweet=re.sub('https?://[A-Za-z0-9./]+','',tweet)
    #Hashtags/numbers-non charcater :
    tweet=re.sub("[^a-zA-Z]", " ", tweet)  
    #Lemmatization:
    tweet=spacy_nlp(tweet)
    tweet = ' '.join(word.lemma_ for word in tweet) 

    return tweet



#%% Affichage des résultats :

def classification_partis(y_test, y_pred, partis):
    """
    fonction classification_partis
    """
    fig=plt.figure(figsize=(20,7))
    n=len(partis)
    for i in range(n):
        plt.subplot(2,3,i+1)
        plt.title('Classification du parti : {parti}'.format(parti=partis[i]))
        sns.countplot(y_pred[y_test==partis[i]])
    plt.show()
    return True
    
def plotting_confusion_matrix(y_test,y_pred,partis):
    """
    fonction plotting_confusion_matrix ayant pour but d'afficher les matrices de confusion des classifications
    @Inputs:
        - y_test, les 'vrais' labels du test set
        - y_pred, les labels prédits du test set
        - partis, les labels présents dans la base
    @Output:
        None
    """
    #On initialise la figure:
    fig=plt.figure(figsize = (10,7))
    #On utilise l'implémentation de scikit-learn pour obtenir la matrice de confusion à afficher
    cm = confusion_matrix(y_test,y_pred, labels=partis)              
    # On normalise cette matrice de confusion
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #On affiche la matrice de confusion
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=partis, yticklabels=partis, cmap="Greens")
    #On fixe des labels pour les axes
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    #On affiche la figure
    plt.show()
    return None


#%% Affichage des corrélations :

def correlations_uni_bi_grams(features, labels, tfidf, N=2):
    """
    fonction correlations_uni_bi_grams permettant d'afficher les uni-grammes et 
    les bi-grammes les plus corrélés à chaque label
    
    @Inputs:
        - features
        - labels
        - tfidf
        - N, le nombre de N-grammes à afficher
    @Output:
        None
    """
    parties=labels.unique()
    # N: nombre d'unigrams et de bigrams à afficher :
    for party in parties:
        features_chi2 = chi2(features, labels == party)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("For the party : '{}':".format(party))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
    return None


#%% Binary case :
def models_test_binary(features, labels, CV=5):
    models = [
    SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None),
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(max_depth=5)
    ]
    #,AdaBoostClassifier()
    #MLPClassifier(alpha=1),
    #KNeighborsClassifier(3),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    
    
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
    
    
    
    partis=labels.unique()
    
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=0)
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_validate(model, features, labels, scoring=['accuracy','roc_auc'], cv=CV)      
        roc_auc_scores=accuracies['test_roc_auc']
        accuracy_scores=accuracies['test_accuracy']
        
        for fold_idx, accuracy in enumerate(accuracy_scores):
            roc_auc=roc_auc_scores[fold_idx]
            #On append des informations sur la Validation croisée :
            entries.append((model_name, fold_idx, accuracy, roc_auc ))
        #On fit le modèle:
        model.fit(X_train, y_train)
        #Prediction sur le modèle :
        y_pred = model.predict(X_test)
        #On affiche le nom du modèle actuel :
        print("-------------", model_name, "----------------------------------------")
        #Printing classification report :
        print(classification_report(y_test, y_pred, partis))
        #Affichage de la matrice de confusion :
        plotting_confusion_matrix(y_test,y_pred, partis)
        
        
        #Affichage de la classification de chaque parti :
        #classification_partis(y_test, y_pred)
        
    
    #Affichage graphique des scores CV de chaque modèle :
    fig=plt.figure(figsize=(15,10))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy', 'roc_auc'])
    plt.title('accuracy:')
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    fig=plt.figure(figsize=(15,10))
    plt.title('roc_auc:')
    sns.boxplot(x='model_name', y='roc_auc', data=cv_df)
    sns.stripplot(x='model_name', y='roc_auc', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    #Affichage du score CV sur chaque modèle :
    print('Accuracy:')
    print(cv_df.groupby('model_name').accuracy.mean())
    print('ROC AUC:')
    print(cv_df.groupby('model_name').roc_auc.mean())
    return True 

#%% Multiclass:
def multi_models_test(features, labels, CV=5):
    models = [
    SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None),
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(max_depth=5)
    ]
    #,AdaBoostClassifier()
    #MLPClassifier(alpha=1),
    #KNeighborsClassifier(3),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    
    
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
    
    partis=labels.unique()
    
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=0)
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_validate(model, features, labels, scoring=['accuracy'], cv=CV)      
        accuracy_scores=accuracies['test_accuracy']
        
        for fold_idx, accuracy in enumerate(accuracy_scores):
            #On append des informations sur la Validation croisée :
            entries.append((model_name, fold_idx, accuracy ))
        #On fit le modèle:
        model.fit(X_train, y_train)
        #Prediction sur le modèle :
        y_pred = model.predict(X_test)
        #On affiche le nom du modèle actuel :
        print("-------------", model_name, "----------------------------------------")
        #Printing classification report :
        print(classification_report(y_test, y_pred, partis))
        #Affichage de la matrice de confusion :
        plotting_confusion_matrix(y_test,y_pred, partis)
        
        
        
        #Affichage du score ROC-AUC:
        #lb = preprocessing.LabelBinarizer()
        #lb.fit(y_test)
        #y_test_encoded=lb.transform(y_test)
        #print('ROC-AUC', roc_auc_score(y_test_encoded, lb.transform(model.predict_proba(X_test))))
        
        
        #Affichage de la classification de chaque parti :
        #classification_partis(y_test, y_pred)
        
    
    #Affichage graphique des scores CV de chaque modèle :
    fig=plt.figure(figsize=(15,10))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    plt.title('accuracy:')
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    
    #Affichage du score CV sur chaque modèle :
    print('Accuracy:')
    print(cv_df.groupby('model_name').accuracy.mean())
    return True 


#%% Wordcloud : 
def wordcloud_decision(model, features,df):
    #On appelle la decision_function : 
    labels_decision_function=model.decision_function(features)
    #On cherche les labels tels que définis dans la modèle:
    Labels=model.classes_
    
    df2=df.reset_index()
    for i, label in enumerate(Labels):
        s=pd.Series(abs(labels_decision_function[:,i]))
        #On définit le texte:
        text=df2.texte[s <= np.sort(abs(labels_decision_function[:,i]))[100]]
        
        #On définit le nuage:
        cloud = WordCloud(stopwords=['oui', 'non', 'qu', 'est'], background_color='black',
                                  collocations=False,
                                  width=2500,
                                  height=1800
                                 ).generate(" ".join(text))
        
        plt.figure(figsize=(40,25))
        plt.axis('off')
        plt.title(label, size=50)
        plt.imshow(cloud)
    return None