# How To Collect the Twitter Stream Collecttion?

This python code will help you to access twitter data more clean structured by any HashTag, CashTag, or ny other filter you have in your mind that it is valuable to be extract from twitter.


Using our code to collect twittes would be done in 3 easy step:

# Config and Create your twitter App 
Create an app by access the twitter portal through this link https://developer.twitter.com/en/apps
After creating the app under the "Keys and Tokens" Tab find these below information and update the Config.py 

```
    Consumer_Key  = '' #API key
    Consumer_Secret = '' #API secret key
    Access_Token = ''
    Access_Token_Secret = ''
```

# Selecte your DataBaseName And list the langiages you are going to collect 

```
    CONNECTION_STRING = "sqlite:///Twitter.db"
    LANG = ["en"]
```

# Start Collecting

run this code in your Terminal

```
 python TwitterREader.py
```

# Publications used this Code

How do I cite?

```
@inproceedings{asghari2018trends,
  title={Trends on Health in Social Media: Analysis using Twitter Topic Modeling},
  author={Asghari, Mohsen and Sierra-Sosa, Daniel and Elmaghraby, Adel},
  booktitle={2018 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT)},
  pages={558--563},
  year={2018},
  organization={IEEE}
}

```

[Trends on Health in Social Media: Analysis using Twitter Topic Modeling](https://www.researchgate.net/profile/Mohsen_Asghari5/publication/331205903_Trends_on_Health_in_Social_Media_Analysis_using_Twitter_Topic_Modeling/links/5c75529e299bf1268d28248f/Trends-on-Health-in-Social-Media-Analysis-using-Twitter-Topic-Modeling.pdf)