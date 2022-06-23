# BumbleBee <img src="https://user-images.githubusercontent.com/25641555/76114333-d7a63480-5fb3-11ea-96e1-8d2ff27c4a7f.png" width="128" height="200" /> 

Easy Twitter API Data Collection By python : https://github.com/moasgh/BumbleBee/tree/master/DataCollections/Twitter

Named Entity Recognition : https://github.com/moasgh/BumbleBee/tree/master/NER 

[Trends on Health in Social Media: Analysis using Twitter Topic Modeling](https://www.researchgate.net/profile/Mohsen_Asghari5/publication/331205903_Trends_on_Health_in_Social_Media_Analysis_using_Twitter_Topic_Modeling/links/5c75529e299bf1268d28248f/Trends-on-Health-in-Social-Media-Analysis-using-Twitter-Topic-Modeling.pdf)

There is a growing interest on social networks for topics related to Healthcare. In particular, on Twitter, millions of tweets related to healthcare can be found. These posts contain public opinions on health, and allow to understand how is the popular perception on topics such as medical diagnosis, medicines, facilities, and claims. In this paper we present an adaptive system designed using 5 layers. The system contains a combination of unsupervised and supervised algorithms to track the trends of health social media. As it is based on a word2vec model, it also captures the correlation of words based on the context, improving over time, enhancing the accuracy of predictions and tweet tracking. In this work we focused on United States data and use it to detect the trending topics of each state. These topics are followed including new social network contributions. The supervised algorithm implemented is a Convolutional Neural Network (CNN) in conjunction with the Word2Vect model to classify and label new tweets, assigning a feedback to the topic models. The results of this algorithm present an accuracy of 83.34%, precision of 83%, recall 84% and F-Score of 83.8% when evaluated. Our results are compared with two state of the art techniques demonstrating an advantage that can be leveraged for further improvements.  

[A topic modeling framework for spatio-temporal information management](https://www.sciencedirect.com/science/article/pii/S0306457320308359?casa_token=talkTmJVV14AAAAA:C6nfpEeQBd9-gl4ADV21MsZ1DbrAFPA5IGdWlAt_E5l9P5Ts1J9biQR04fLJ91QAGVb0qO6zaFM7)

Highlights
•
Propose a robust procedure to take a decision for selecting the best topic model. We design an adaptive framework to use gained knowledge for improving the result over time. For our case study we used four topic modeling techniques and report the result of the evaluation techniques.

•
Propose a neural network using transfer learning techniques to enhance the framework ability to detect unrelated messages over data streams existing in twitter. We focus our attention in healthcare to present examples.

•
Create automatic deep cleaning method to enhance the quality of data to perform better classification in outlier and topic detection.


[BINER: A low-cost biomedical named entity recognition](https://www.sciencedirect.com/science/article/pii/S0020025522003838)

A primary focus of the healthcare industry is to improve patient experience and quality of service. Practitioners and health workers are generating large volumes of text that are captured in Electronic Medical Records, clinical reports, and publications. Additionally, patients post millions of comments on social media related to healthcare, on diverse topics such as hospital services, disease symptoms, and drugs effects. Unifying various data sources can guide physicians and healthcare workers to avoid unnecessary, irrelevant information and expedite access to helpful information. The main challenge to creating Biomedical Natural Language Understanding is the lack of standard datasets and the extensive computational resources needed to develop different models. This paper proposes a model trained on low-tier GPU computers, producing comparable results to larger models like BioBERT. We propose BINER, a Biomedical Named Entity Recognition architecture using limited data and computational resources.


```
@inproceedings{asghari2018trends,
  title={Trends on Health in Social Media: Analysis using Twitter Topic Modeling},
  author={Asghari, Mohsen and Sierra-Sosa, Daniel and Elmaghraby, Adel},
  booktitle={2018 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT)},
  pages={558--563},
  year={2018},
  organization={IEEE}
}

@article{asghari2020topic,
  title={A topic modeling framework for spatio-temporal information management},
  author={Asghari, Mohsen and Sierra-Sosa, Daniel and Elmaghraby, Adel S},
  journal={Information Processing \& Management},
  pages={102340},
  year={2020},
  publisher={Elsevier}
}

@article{asghari2022biner,
  title={BINER: A low-cost biomedical named entity recognition},
  author={Asghari, Mohsen and Sierra-Sosa, Daniel and Elmaghraby, Adel S},
  journal={Information Sciences},
  volume={602},
  pages={184--200},
  year={2022},
  publisher={Elsevier}
}
```

Named Entity Regnition Datasets : https://github.com/moasgh/BumbleBee/tree/master/NER/datasets

Named Entity Regonition Models : https://github.com/moasgh/BumbleBee/tree/master/NER
