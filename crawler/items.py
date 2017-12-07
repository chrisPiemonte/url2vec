# -*- coding: utf-8 -*-

from scrapy.item import Item, Field

class PageItem(Item):
    url = Field()
    title = Field()
    text = Field()
    depth = Field()
    outlinks = Field()

class LinkItem(Item):
    src = Field()
    text = Field()
    dst = Field()
