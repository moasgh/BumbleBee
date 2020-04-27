from requests import get
from requests.exceptions import RequestException
from contextlib import closing
import xml.etree.ElementTree as xme
import time
import re
import pandas as pd
import calendar

def getrequest(url):
    try:
        with closing(get(url,stream=True)) as resp:
            if (is_good_response(resp)):
                return resp.content
            else:
                print("Bad Request")
                return None
    except RequestException as e:
        print("[Error] {0}:{1}".format(url , str(e)))
        return None

def is_good_response(resp):
    return (resp.status_code == 200)

def clean_html(txt):
    return re.sub(r'(<script(\s|\S)*?<\/script>)|(<style(\s|\S)*?<\/style>)|(<!--(\s|\S)*?-->)|(<\/?(\s|\S)*?>)','', txt)

class Collect():

    class MetaData():
        def __init__(self, pubmedid):
            self.Type = ''
            self.UI = ''
            self.PubmedID = pubmedid
            # Chemical Information
            self.NameOfSubstance = ''
            self.RegistryNumber = ''
            # MEshHeadingID will connect QualifierName to DescriptorName if exists
            self.MeshHeadingID = ''
            self.DescriptorName = ''
            self.MajorTopicYN = '' 
            self.QualifierName = ''
    
    class Article():
        def __init__(self, pubmedid):
            self.pubmedID = pubmedid
            self.ArticleTitle = ''
            self.AbstractText = ''
            self.PubYear = 0
            self.PubMonth = 0

    def __init__(self):
        self.MAINLINK = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.BASEURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.DB_SOURCE = "pubmed"
        self.KEYWORDS = ""
        # replace the API Key Generated from PUBMED website 
        self.API_KEY = ""
        self.RETMODE = "xml"
        self.RETTYPE = "abstract"
        self.Articles = []
        self.MetaDataList = []
        self.monthabbr_number = { a.lower():n for n,a in enumerate(calendar.month_abbr) }
    
    def getabstract(self,pubmedid):
        url = self.BASEURL + "?db="+self.DB_SOURCE+"&id="+str(pubmedid)+"&retmode="+self.RETMODE+"&rettype="+self.RETTYPE+"&api_key="+self.API_KEY
        Article = self.Article(pubmedid)
        _MetaDataList = []
        html_contnet = getrequest(url)
        if html_contnet is not None:
            root = xme.fromstring(html_contnet)
            # The format of AbstractText would be
            # <Abstract>
            # <AbstractText Label="OBJECTIVE">To assess the effects...</AbstractText>
            # <AbstractText Label="METHODS">Patients attending lung...</AbstractText>
            # <AbstractText Label="RESULTS">Twenty-five patients...</AbstractText>
            # <AbstractText Label="CONCLUSIONS">The findings suggest...</AbstractText>
            # </Abstract>
            # source Documentation : https://www.ncbi.nlm.nih.gov/books/NBK3828/
            Abstract = root.find('PubmedArticle/MedlineCitation/Article/Abstract')
            if Abstract is not None:
                AbstractText = ""
                for abstract_part in Abstract.findall('AbstractText'):
                    AbstractText += ('\n' if AbstractText else '') + abstract_part.text
                Article.AbstractText = AbstractText
                ArticleTitle = root.find('PubmedArticle/MedlineCitation/Article/ArticleTitle')
                Article.ArticleTitle = ArticleTitle.text if ArticleTitle is not None else ''
                PubDate = root.find('PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/PubDate')
                Article.PubYear = int(PubDate.find('Year').text) if PubDate.find('Year') is not None else 0
                Article.PubMonth = self.monthabbr_number[PubDate.find('Month').text.lower()] if PubDate.find('Month') is not None else 0

                ChemicalList = root.find('PubmedArticle/MedlineCitation/ChemicalList')
                if ChemicalList is not None:
                    for chemic in ChemicalList.findall('Chemical'):
                        metadata = self.MetaData(pubmedid)
                        metadata.Type = 'Chemical'
                        metadata.RegistryNumber = chemic.find('RegistryNumber').text if chemic.find('RegistryNumber') is not None else ''
                        NameOfSubstance = chemic.find('NameOfSubstance')
                        if NameOfSubstance is not None:
                            metadata.UI = NameOfSubstance.attrib['UI'] if 'UI' in NameOfSubstance.attrib else ''
                            metadata.NameOfSubstance = chemic.find('NameOfSubstance').text
                        _MetaDataList.append(metadata)
                MeshHeadingList = root.find('PubmedArticle/MedlineCitation/MeshHeadingList')
                if MeshHeadingList is not None:
                    MeshID = 1
                    for Mesh in MeshHeadingList.findall('MeshHeading'):
                        DescriptorName = Mesh.find('DescriptorName')
                        if DescriptorName is not None:
                            metadata = self.MetaData(pubmedid)
                            metadata.Type = 'DescriptorName'
                            metadata.MeshHeadingID = MeshID
                            metadata.DescriptorName = DescriptorName.text
                            metadata.UI = DescriptorName.attrib['UI'] if 'UI' in DescriptorName.attrib else ''
                            metadata.MajorTopicYN = DescriptorName.attrib['MajorTopicYN'] if 'MajorTopicYN' in DescriptorName.attrib else ''
                            _MetaDataList.append(metadata)
                        QualifierName = Mesh.find('QualifierName')
                        if QualifierName is not None:
                            metadata = self.MetaData(pubmedid)
                            metadata.Type = 'QualifierName'
                            metadata.MeshHeadingID = MeshID
                            metadata.QualifierName = QualifierName.text
                            metadata.UI = QualifierName.attrib['UI'] if 'UI' in QualifierName.attrib else ''
                            metadata.MajorTopicYN = QualifierName.attrib['MajorTopicYN'] if 'MajorTopicYN' in QualifierName.attrib else ''
                            _MetaDataList.append(metadata)
                        MeshID+=1
                self.Articles.append(Article)
                self.MetaDataList.extend(_MetaDataList)
            else:
                print('Abstract Empty')
            

    def start_in_range_pages(self, start , page , perpage):
        for i in range(start , page):
            url = self.MAINLINK+"?db="+self.DB_SOURCE+"&term="+self.KEYWORDS+"&retmax="+str(perpage)+"&retstart="+str(i*perpage)
            print(url)
            html_content = getrequest(url)
            if html_content:
                root = xme.fromstring(html_content)
                ids = []
                for id in root.iter('Id'):
                    ids.append(id.text)
                IDS = ",".join(ids)
                for id in IDS:
                    self.getabstract(id)
                    
    def to_dataframe(self):
        article_columns = [ a for a,b in self.Article(0).__dict__.items()]
        article_records = [[getattr(article , attrib) for attrib in article_columns] for article in self.Articles]
        metadata_columns = [a for a,b in self.MetaData(0).__dict__.items()]
        metadata_records = [[getattr(metadata,attrib) for attrib in metadata_columns] for metadata in self.MetaDataList]
        Articles_df = pd.DataFrame(data=article_records , columns = article_columns)
        MetaData_df = pd.DataFrame(data=metadata_records , columns = metadata_columns)
        return Articles_df, MetaData_df

    def to_csv(self, article_filename,metadata_filename):
        Article_df , MetaData_df = self.to_dataframe()
        Article_df.to_csv(article_filename)
        MetaData_df.to_csv(metadata_filename)


                
                