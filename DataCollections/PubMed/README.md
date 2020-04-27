# How To Collect the Abstract From Pubmed?

This python code will help you to access Abstrct on Pubmed.

# How get an API-Key

Please visit this page:

https://www.ncbi.nlm.nih.gov/books/NBK25497/

On December 1, 2018, NCBI will begin enforcing the use of API keys that will offer enhanced levels of supported access to the E-utilities. After that date, any site (IP address) posting more than 3 requests per second to the E-utilities without an API key will receive an error message. By including an API key, a site can post up to 10 requests per second by default. Higher rates are available by request (vog.hin.mln.ibcn@seitilitue). Users can obtain an API key now from the Settings page of their NCBI account (to create an account, visit http://www.ncbi.nlm.nih.gov/account/). After creating the key, users should include it in each E-utility request by assigning it to the new api_key parameter.


When you get you API_key replace that unique ID in __init__() function in PubMedHandler.py

```
    # replace the API Key Generated from PUBMED website 
    self.API_KEY = ""
```