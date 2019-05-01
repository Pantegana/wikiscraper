import scrapy
from w3lib.html import remove_tags
import nltk
from nltk.corpus import stopwords
import re
from collections import OrderedDict
import pandas as pd

class ImageSpider(scrapy.Spider):
    summary_length = 10
    name = "text"
    allowed_domains = ["https://en.wikipedia.org/"]
    start_urls = [
        "https://en.wikipedia.org/wiki/Pitta",
        "https://en.wikipedia.org/wiki/Erwin_Rommel",
        "https://en.wikipedia.org/wiki/Isaac_Komnenos_(son_of_John_II)",
        "https://en.wikipedia.org/wiki/Ernst_J%C3%BCnger",
        "https://en.wikipedia.org/wiki/Tuvan_throat_singing"
        ]

    def parse(self, response):
        words = {}
        s1 = [self.processParagraph(p, words) for p in response.css("p").getall()]
        sentences = pd.concat(s1, ignore_index=True)
        sentences = sentences.drop_duplicates()
        sentences['score'] = sentences.apply(lambda row: self.scoreSentence(words, row['sentence']), axis=1)
        sentences = sentences.nlargest(self.summary_length, 'score')
        sentences = sentences.sort_index()

        with open("./results/{}.txt".format(response.css("title::text").get()), mode="w", encoding="utf-8") as file:
            sentences.apply(lambda row: file.write("{}\n".format(row['sentence'])), axis=1)
        file.close()
        
    def processParagraph(self, p, words):
        sentences = pd.DataFrame(columns=['sentence'])
        ps = nltk.PorterStemmer()
        stop_words = set(stopwords.words('english'))
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        text = remove_tags(p)
        text = re.sub(r'\[[0-9]*\]', '', text)  
        text = re.sub(r'\xa0', ' ', text)

        for word in [ps.stem(w) for w in nltk.sent_tokenize(text) if w not in stop_words]:
            if word not in words.keys():
                words[word] = 1
            else:
                words[word] += 1

        for s in sent_tokenizer.tokenize(text):
            s = s.strip()
            s = re.sub(r'[^0-9a-zA-Z]$', '.', s)
            if len(s) > 1 and len(s.split()) > 5:
                sentences = pd.concat(
                    [sentences, pd.DataFrame({'sentence' : s}, index=[0])], ignore_index=True)
        return sentences

    def scoreSentence(self, words, sentence):
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        score = 0
        for word, count in words.items():
            if word in tokenizer.tokenize(sentence.lower()):
                score += count
        return score / len(sentence)