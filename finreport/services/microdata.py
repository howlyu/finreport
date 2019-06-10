# -*- coding: utf-8 -*-
import pandas as pd
import QUANTAXIS as QA
import ffn
from datetime import date, timedelta
from pyecharts_v12.charts import Bar, Kline, Grid
from pyecharts_v12.options import global_options as gopts
from pyecharts_v12.options import series_options as sopts
import numpy as np

symbols = ['^gspc', 'wbai', 'tsla', 'estc', 'lite']
today = date.today()
startdate = (today - timedelta(days=365)).strftime("%Y-%m-%d")
enddate = (today - timedelta(days=1)).strftime("%Y-%m-%d")


def getWatchlist(source='qa'):
	if source == 'ffn':
		data = getDataFromFFN(','.join(symbols), startdate, enddate)
		perf = data.calc_stats()
		return perf.stats
	else:
		data = getDataFromQA([s.upper() for s in symbols], startdate, enddate)
		tdata = transQAToFfn(data)
		tdata.columns = map(str.lower, tdata.columns)
		perf = ffn.calc_stats(tdata)
		# print(perf.stats)
		return perf.stats


def getDataFromQA(symbols, startdate, enddate):
	return QA.QA_fetch_stock_day_adv(symbols, startdate, enddate)


def getDataFromFFN(symbols, startdate, enddate):
	return ffn.get(symbols, start=startdate, end=enddate)


def getSymbolList():
	return [s.replace("^", "") for s in symbols]


'''
	Transform QAData to FFN data 
	return dataframe
'''


def transQAToFfn(QAData):
	data = QAData.to_pd().close.reset_index().set_index("date")
	frame = {}
	symbols = data.code.unique()
	for s in symbols:
		frame.update({s: data[data.code == s].close})
	return pd.DataFrame(frame)


def formatToFloat(arr):
	res = []
	for row in arr:
		res.append(["{0:.2f}".format(r) for r in row])
	return res


def getKlineBySymbol(symbol="TSLA") -> Grid:
	data = getDataFromQA(symbol.upper(), startdate=startdate, enddate=enddate).to_pd()
	data = data.reset_index().set_index("date")
	datetime = [i.strftime("%Y-%m-%d") for i in data.index]
	ohlc = formatToFloat(np.array(data.loc[:, ['open', 'close', 'low', 'high']]))
	vol = [v for v in data.volume]

	kline = Kline() \
		.add_xaxis(datetime) \
		.add_yaxis("kline",
	               ohlc,
	               itemstyle_opts=sopts.ItemStyleOpts(
		               color="#ec0000",
		               # color0="#00da3c",
		               border_color="#8A0000",
		               # border_color0="#008F28",
	               ),
	               markline_opts=sopts.MarkLineOpts(
		               data=[sopts.MarkLineItem(type_="max", value_dim="close"),
		                     sopts.MarkLineItem(type_="min", value_dim="close")]
	               ),
	               markpoint_opts=sopts.MarkPointOpts(
		               data=[sopts.MarkPointItem(type_="max", value_dim="close"),
		                     sopts.MarkPointItem(type_="min", value_dim="close")]
	               ),
	               ) \
		.set_global_opts(yaxis_opts=gopts.AxisOpts(is_scale=True),
	                     xaxis_opts=gopts.AxisOpts(is_scale=True),
	                     legend_opts=gopts.LegendOpts(is_show=False),
                         title_opts=gopts.TitleOpts(title="{} ({} ~ {})".format(symbol.upper(), startdate, enddate))
	                     )

	bar = Bar(init_opts=gopts.InitOpts(height="50px")) \
		.add_xaxis(datetime) \
		.add_yaxis("volume", vol) \
		.set_series_opts(
			label_opts=sopts.LabelOpts(is_show=False),

		)\
		.set_global_opts(datazoom_opts=gopts.DataZoomOpts(xaxis_index=[0, 1],
	                                                      is_show=True,
	                                                      range_start=75,
	                                                      range_end=100, ),
	                     legend_opts=gopts.LegendOpts(is_show=False),
	                     yaxis_opts=gopts.AxisOpts(is_show=False))

	grid = Grid()#init_opts=gopts.InitOpts(width="800px", height="600px")
	grid.add(bar, grid_opts=gopts.GridOpts(pos_top="85%"))
	grid.add(kline, grid_opts=gopts.GridOpts(pos_bottom="20%"))

	return grid

# print(getWatchlist())
