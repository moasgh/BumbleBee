How To Collect the Twitter Stream Collecttion?

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
