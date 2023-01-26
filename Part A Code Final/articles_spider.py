import scrapy


class ArticlesSpider(scrapy.Spider):

    name = "article"
    start_urls = ['https://english.elpais.com/society/9/', 'https://english.elpais.com/international/9/',
                  'https://english.elpais.com/culture/9/', 'https://english.elpais.com/sports/9/',
                  'https://english.elpais.com/science-tech/9/', 'https://english.elpais.com/usa/9/',
                  'https://english.elpais.com/economy-and-business/9/']



    def parse(self, response):
        for link in response.css('h2.c_t a::attr(href)'):
            link = "{}{}".format("https://english.elpais.com/", link.get())
            yield response.follow(link, callback=self.parse_article)
            '''yield {
                    'title': article.css('a::text').get(),
                    'link': "{}{}".format("https://english.elpais.com/", article.css('a::attr(href)').get())
                     }'''


    def parse_article(self, response):
        #url =response.css('div.w_rs_i #btn_share_link_97::attr(href)').get().split('#?')[0]
        arts = response.css('article.a._g._g-lg._g-o')

        for art in arts:
            try:
                yield {
                        'title': art.css(' h1.a_t::text').get(),
                        'url': art.css('a._btn.rs_l ::attr(href)').get().split('#?')[0],
                        'text': ' '.join(art.css('div.a_c.clearfix > p::text').getall()).replace("\n",' ').replace("\r",' ').replace("  ", '')
                }
            except:
                continue

'''def parse(self, response):
    page = response.url.split("/")[-2]
    filename = 'articles-%s.html' % page
    with open(filename, 'wb') as f:
        f.write(response.body)
    self.log('saved file %s' % filename)
'''
#articles.css('a::text').getall() #get the titles of the articles
#articles.css('a::attr(href)').getall() #get the links of the articles
#response.css('div.a_c.clearfix > p::text').getall()
#response.css('article.a._g._g-lg._g-o h1.a_t::text').get()
#b-t_a _df