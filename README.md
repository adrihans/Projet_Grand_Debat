# Projet_Grand_Debat

Ce projet a été conduit dans le cadre du projet d'intégration à réaliser au cours de la dernière année d'étude à l'Ecole Centrale de Lille.

Il a pour but d'étudier la représentativité des individus ayant participé au grand débat national par rapport à la population française. 

Pour obtenir des résultats, il a été nécessaire d'obtenir des données textuelles annotées en fonction du parti politique ou en plus largement en fonction de sa position par rapport à certains sujets sociétaux. 

Trois bases de données principales, en plus bien évidemment de la base de données en OpenData issue du grand débat national, ont été collectées, nettoyées et utilisées : 

- Une base de données twitter, associé en fonction du parti politique lors de l'élection présidentielle de 2017.
- Une base de données twitter concernant les parlementaires (députés et sénateurs) français, puisque leurs positions politiques étaient connues
- Une base de données, appelée "Entendre la France", qui contenait notamment la position par rapport au mouvement des gilets jaunes des contributeurs de ce grande débat "parallèle". 

L'apprentissage d'algorithmes classiques a été réalisé sur ces trois bases de données annotées, ce qui a permis ensuite de classifier toutes les contributions au grand débat. 

Ce projet étant un exemple typique d'exploitation de processus de NLP, plusieurs étapes et techniques ont été utilisées, comme par exemple : 
- Lemmatisation
- Suppression de Stop-words
- TF-IDF
- WordCloud
- ... 

Faute de temps pour réaliser ce projet, les techniques de word embeddings n'ont pas été utilisées pour ce projet. 

Ce projet ayant été réalisé avant tout comme un projet devant être évalué d'un point de vu scolaire, un [rapport](https://github.com/adrihans/Projet_Grand_Debat/blob/master/Rapport_final_HANS_Adrien.pdf) a été écrit, que vous pouvez retrouver [ici](https://github.com/adrihans/Projet_Grand_Debat/blob/master/Rapport_final_HANS_Adrien.pdf). 
