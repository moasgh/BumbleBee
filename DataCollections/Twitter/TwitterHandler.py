import Config
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import dataset
import re
import CTweets
import pandas as pd
import sys

#FIPS = pd.read_csv('FipsCode.csv')
#USCodes = {}
#statename = {}

# for index,row in FIPS.iterrows():
#     USCodes[row['Name'].lower()] = row['FIPS']
#     if statename.get(row['State'].lower()) is None:
#         statename[row['State'].lower()] = row['FIPS']

def in_usa(state):
    if state is None:
        return False
    state = state.lower().strip()
    if ',' in state:
        keys = state.split(',')
        if USCodes.get(keys[0].strip()) or statename.get(keys[0].strip()):
            print('is in US')
            return True
        if len(keys) > 1 and USCodes.get(keys[1].strip()) or statename.get(keys[1].strip()):
            print('is in US')
            return True

    if USCodes.get(state.strip()) or statename.get(state.strip()) or 'usa' in state \
            or 'united state' in state or 'united states' in state or 'u.s.' in state:
        print('is in US')
        return True
    return False

class StdoutListener(StreamListener):
    def __init__(self , connectionstring = None , lang = None , verbos = True , limit = -1):
        super().__init__()
        if connectionstring:
            self.Connection_String = connectionstring
        else:
            self.Connection_String = Config.Twitter.CONNECTION_STRING
        if lang:
            self.LANG = lang
        else:
            self.LANG = Config.Twitter.LANG
        self.verbos = verbos
        if limit:
            self.limit  = limit
        self.messagecount = 0 
        print(self.Connection_String)
        self.db = dataset.connect(self.Connection_String)
        print("Database is created")
    
    def on_status(self, status):
        
        table = self.db["tweets"]
        hashtags = status.entities['hashtags']
        description = status.user.description
        loc = status.user.location
        coordinate = status.coordinates
        name = status.user.screen_name
        user_created = status.user.created_at
        followers = status.user.followers_count
        id_str = status.id_str
        created = status.created_at
        retweets = status.retweet_count
        favorite_count = status.favorite_count
        is_truncated = False
        is_retweet = False
        # sysmbol indicies Start
        sis = -1 if not status.entities['symbols'] else status.entities['symbols'][0]["indices"][0]
        # sysmbol indicies End
        sie = -1 if not status.entities['symbols'] else status.entities['symbols'][0]["indices"][1]
        symbolval = 'nosymbol' if not status.entities['symbols'] else status.entities['symbols'][0]["text"]

        # Remove the tweets by keywords in AVOID_DICTIONARY and Check The Language
        
        if self.verbos: print(loc)
        if status.lang in self.LANG: #and in_usa(loc) is True:#\
                #re.search(Config.Twitter.AVOID_DICTIONARY, status.text.lower()) is None and in_usa(loc) is True:

            # Get the Coordinate
            if self.verbos: print(id_str, loc)
            if coordinate is not None:
                coordinate = json.dumps(coordinate)

            # Get The HashTags
            if hashtags is not None:
                hashtags = json.dumps(hashtags)

            # Get the Full Text of the Tweet
            if self.verbos: print("--------------------")
            if hasattr(status, 'retweeted_status'):
                is_retweet = True
                try:
                    is_truncated = False
                    text = status.retweeted_status.extended_tweet["full_text"]
                    try:
                        retweets = status.retweeted_status.extended_tweet["retweet_count"]
                    except:
                        retweets = 0

                    favorite_count = status.retweeted_status.extended_tweet['favorite_count']

                except:

                    if self.verbos: print('exception...')
                    try:
                        is_truncated = True
                        a = tweepy.api.get_status(id_str, tweet_mode='extended')
                        text = a._json['retweeted_status']['full_text']
                        retweets = status.retweeted_status.retweet_count
                        favorite_count = status.retweeted_status.favorite_count
                    except:
                        text = ''
            else:
                is_retweet = False
                is_truncated = False
                try:
                    text = status.extended_tweet["full_text"]
                    favorite_count = status.favorite_count
                except AttributeError:
                    text = status.text

            if self.verbos: print("full text: ", id_str, text)
            if self.verbos: print("--------------------")
            duplicate = table.find_one(text=CTweets.clean(text))
            if duplicate is None:
                if self.verbos: print("Inserting ...")
                table.insert(dict(user_description=description, user_location=loc, coordinates=coordinate,
                                  orig_text=text, text= CTweets.clean(text),
                                  user_name=name, user_created=user_created, user_followers=followers, id_str=id_str,
                                  created=created, retweet_count=retweets, hashtags=hashtags, isretweet=is_retweet,
                                  istruncated=is_truncated,favoritecount=favorite_count,symbolval=symbolval , sis = sis , sie = sie))
                
                self.messagecount+=1
                print('Progress : %d      '%self.messagecount , end = "\r")
                if self.limit > 0:
                    if self.messagecount == self.limit:
                        print('Total number of ' + str(self.messagecount) +" twitter collected.")
                        return False
            else:
                if self.verbos: print("Duplicate Found ...")
                retweets = duplicate['retweet_count']
                retweets = retweets + 1
                table.update(dict(text=CTweets.clean(text), retweet_count=retweets), ['text'])

    def on_error(self, status_code):
        if self.verbos: print(status_code)

class Collect():
    def __init__(self, API_Key , API_Secret_Key, Access_Token, Access_Token_Secret, DataBaseName, LANG , Filters , Limit , verbos = False ):
        self.config = Config.Twitter()
        self.config.Consumer_Key = API_Key
        self.config.Consumer_Secret = API_Secret_Key
        self.config.Access_Token = Access_Token
        self.config.Access_Token_Secret = Access_Token_Secret
        self.config.CONNECTION_STRING = "sqlite:///"+DataBaseName+".db"
        self.config.LANG = LANG
        self.filters = Filters
        self.verbos = verbos
        self.limit = Limit
    
    def start(self):
        listener = StdoutListener(self.config.CONNECTION_STRING , self.config.LANG , self.verbos , self.limit)
        auth = OAuthHandler(self.config.Consumer_Key, self.config.Consumer_Secret)
        auth.set_access_token(self.config.Access_Token, self.config.Access_Token_Secret)
        tweepy.api = tweepy.API(auth)
        print('API Connected to the twitter.')
        print('Collecting started...')
        stream = Stream(auth, listener)
        while True:
            try:
                stream.filter(
                    track=self.filters,
                    is_async=True
                )
            except Exception as ex:
                print(ex)
                break
                #continue

    def to_DataFrame(self):
        self.db = dataset.connect(self.config.CONNECTION_STRING)
        result = self.db.query('SELECT * FROM tweets')
        return pd.DataFrame.from_dict(result)

    def to_csv(self, path , where = None):
        self.db = dataset.connect(self.config.CONNECTION_STRING)
        if where:
            result = self.db.query('SELECT * FROM tweets where ' + where)
            pd.DataFrame.from_dict(result).to_csv(path)
        else:
            result = self.db.query('SELECT * FROM tweets')
            pd.DataFrame.from_dict(result).to_csv(path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        listener = StdoutListener()
        auth = OAuthHandler(Config.Twitter.Consumer_Key, Config.Twitter.Consumer_Secret)
        auth.set_access_token(Config.Twitter.Access_Token, Config.Twitter.Access_Token_Secret)
        tweepy.api = tweepy.API(auth)
        stream = Stream(auth, listener)

        ## https://github.com/tweepy/tweepy/issues/908 answer by diegoje
        ## for error: Connection broken: IncompleteRead(0 bytes read)
        while True:
            try:
                stream.filter(
                    track=["obama"],is_async=True
                )
            except Exception as ex:
                continue
