import dataset
import Config
import json

db = dataset.connect(Config.Twitter.CONNECTION_STRING)

class T_tweets:
    def query(self, f):
        return db.query("select * from tweets where " + f)

class T_sentiment:
    table_name = "sentiment"
    def __init__(self):
        self.table = db[self.table_name]
    def inser(self,id_tweet,blob_polarity,blob_subjectivity,vader_neg,vader_neu,vader_pos,vader_compund,tags,noun_phrase):
        if noun_phrase is not None:
            noun_phrase = json.dumps(noun_phrase)
        tags = json.dumps(tags)
        self.table.insert(dict(id_tweet=id_tweet,blob_polarity=blob_polarity,blob_subjectivity=blob_subjectivity,vader_neg=vader_neg,
        vader_neu=vader_neu,vader_pos=vader_pos,vader_compund=vader_compund,tags=tags,noun_phrase=noun_phrase))
    def hastweetid(self,id_tweet):
        result = self.table.find_one(id_tweet=id_tweet)
        if result is not None:
            return True
        return False