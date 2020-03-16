import dataset
import Config
import json

db = dataset.connect(Config.Twitter.CONNECTION_STRING)

class T_tweets:
    def query(self, f):
        return db.query("select * from tweets where " + f)