# -*- coding: utf-8 -*-
import pandas as pd
import pymongo
from finreport import settings
from bson.json_util import dumps


def getPresaleList() -> list:
	client = pymongo.MongoClient(settings.MONGODB_URI)
	presale_list = client.properties.presale_list
	ref = presale_list.find({}).limit(5).sort("add_date", -1)
	return list(ref)

# print(getPresaleList())