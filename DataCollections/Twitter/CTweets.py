import re


def clean(text):
    ctext = clean_urls(text)
    ctext = clean_references(ctext)
    ctext = clean_nonechars(ctext)
    ctext = ctext.strip()
    return ctext


def clean_urls(text):
    return re.sub('https?://[A-Za-z0-9./]+', '', text)


def clean_references(text):
    return re.sub('@[A-Za-z0-9]+', '', text)


def clean_nonechars(text):
    return re.sub('[^a-zA-Z0-9|\s|#]', '', text)


def clean_hashtags(text):
    return re.sub('#', '', text)


def clean_numbers(text):
    return re.sub('[0-9]+','',text)


def clean_retweets_char(text):
    return re.sub('rt ','',text)
