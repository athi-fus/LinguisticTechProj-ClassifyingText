import scrapy


class AriclesSpider(scrapy.Spider):

    name = "article2"
    start_urls= ['https://www.vox.com/business-and-finance/archives/14',
                 'https://www.vox.com/culture/archives/14',
                 'https://www.vox.com/policy-and-politics/archives/14',
                 'https://www.vox.com/science-and-health/archives/14',
                 'https://www.vox.com/world/archives/14',
                 'https://www.vox.com/technology/archives/14', 'https://www.vox.com/energy-and-environment/archives/14']


    def parse(self, response):
        for link in response.css('h2.c-entry-box--compact__title a::attr(href)'):
            # link = "{}{}".format("https://english.elpais.com/", link.get())
            yield response.follow(link.get(), callback=self.parse_article)
            '''yield {
                    'title': article.css('a::text').get(),
                    'link': "{}{}".format("https://english.elpais.com/", article.css('a::attr(href)').get())
                     }'''


    def parse_article(self, response):
        arts = response.css('article.l-segment.l-main-content')

        for art in arts:
            try:
                yield {
                        'title' : art.css('h1.c-page-title ::text').get(), # <<<<<<<<<<<<<<<<<<<<<<<
                        'url': response.request.url,
                        'text' : ' '.join(art.css('div.c-entry-content  ::text').getall()).replace("\n",' ').replace("\r",' ').replace('  ', ' ')
                }
            except:
                continue


# cles.css('h2.c-entry-box--compact__title a::text').get()
# cles.css('h2.c-entry-box--compact__title a::attr(href)').get()
# response.css('h1.c-page-title ::text').get()
# response.css('div.c-entry-content  ::text').getall()
#  txt = ' '.join(response.css('div.c-entry-content  ::text').getall()).replace("\n",'').replace("  ", '')
''' a = response.css('article.l-segment.l-main-content')
 a.css('h1.c-page-title ::text').get()
 
 >>> response
<200 https://www.vox.com/>
>>> response.css('h2.c-entry-box--compact__title a::attr(href)').get()
'https://www.vox.com/future-perfect/22841852/covid-drugs-antibodies-fluvoxamine-molnupiravir-paxlovid'
>>> a = response.css('h2.c-entry-box--compact__title a::attr(href)')
>>> a.css('h1.c-page-title ::text').get()
>>> ar = a.css('article.l-segment.l-main-content')
>>> arcss('h1.c-page-title ::text').get()
Traceback (most recent call last):
  File "<console>", line 1, in <module>
NameError: name 'arcss' is not defined
>>> ar.css('h1.c-page-title ::text').get() 
>>> a = response.css('h2.c-entry-box--compact__title a::attr(href)')
>>> a.get()
'https://www.vox.com/future-perfect/22841852/covid-drugs-antibodies-fluvoxamine-molnupiravir-paxlovid'

'''