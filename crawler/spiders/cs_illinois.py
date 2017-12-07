# -*- coding: utf-8 -*-

import scrapy
from scrapy.http import Request
from url2vec_crawler.utils import *
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from url2vec_crawler.items import PageItem, LinkItem

class CsIllinoisSpider(CrawlSpider):
    name = 'cs_illinois'
    allowed_domains = ['cs.illinois.edu']
    denied_domains = ["my.cs.illinois.edu", "slazebni.cs.illinois.edu"]
    start_urls = ['http://www.cs.illinois.edu/']

    def __init__(self, *args, **kwargs):
        super(CsIllinoisSpider, self).__init__( *args, **kwargs)
        self.crawled_urls = {}
        self.link_extractor = LinkExtractor(
            deny=(r'/news', r'/buy', r'/login', r'Login', r'/index.php'),
            allow=(r'^(http://|https://)?(www.)?(cs.illinois.edu)'),
            allow_domains=self.allowed_domains,
            deny_extensions=IGNORED_EXTENSIONS,
            deny_domains=self.denied_domains, unique=True
        )

    def parse(self, response):
        self.logger.info('parsing url %s', response.url)
        self.crawled_urls[response.url] = True

        if response.headers['Content-Type'].startswith('text/html'):
            links = []

            for link in self.link_extractor.extract_links(response):
                if is_valid(link.url):
                    links.append( LinkItem(src = response.url, text = link.text, dst = link.url) )
                    yield Request(link.url, self.parse)

            page = PageItem()
            page['url'] = response.url
            page['text'] = response.body
            page['depth'] = response.meta['depth']
            page['title'] = response.xpath('//title/text()').extract_first()
            page['outlinks'] = links
            yield page
