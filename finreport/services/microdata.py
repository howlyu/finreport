# -*- coding: utf-8 -*-
import pandas as pd
import QUANTAXIS as QA
import ffn
from datetime import date, timedelta
from pyecharts_v12.charts import Bar, Kline, Grid
from pyecharts_v12.options import global_options as gopts
from pyecharts_v12.options import series_options as sopts
import numpy as np
import talib
from .. import settings

# symbols = [ 'wbai', 'tsla', 'estc', 'cldr']#'^gspc',
symbols = [i.strip() for i in settings.SYMBOLS.split(",")]
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
		tdata = transQAToFFN(data)
		tdata.columns = map(str.lower, tdata.columns)
		perf = ffn.calc_stats(tdata)
		# print(perf.stats)
		return perf.stats


def getDataFromQA(symbols, startdate, enddate):
	return QA.QA_fetch_stock_day_adv(symbols, startdate, enddate).to_pd().reset_index().set_index("date")


def getDataFromFFN(symbols, startdate, enddate):
	return ffn.get(symbols, start=startdate, end=enddate)


def getSymbolList():
	return ffn.utils.clean_tickers(symbols)


def transQAToFFN(QAData):
	'''
		Transform QAData to FFN data
		return dataframe
	'''
	frame = {}
	symbols = QAData.code.unique()
	for s in symbols:
		frame.update({s: QAData[QAData.code == s].close})
	return pd.DataFrame(frame)


def formatToFloat(arr):
	res = []
	for row in arr:
		res.append(["{0:.2f}".format(r) for r in row])
	return res


def getKlineBySymbol(symbol="TSLA") -> Grid:
	data = getDataFromQA(symbol.upper(), startdate=startdate, enddate=enddate)
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

def checklistStats(symbol="TSLA"):
	startdate = (today - timedelta(days=365)).strftime("%Y-%m-%d")
	# today_90d = (today - timedelta(days=90)).strftime("%Y-%m-%d")
	data = getDataFromQA(symbol.upper(), startdate=startdate, enddate=enddate)

	# tdata = data.loc[:, ['open', 'close', 'low', 'high']]
	desc = [
		('1_1', 'price', '(current - lowest)/lowest = (10%, 15%) '),
		('1_2', 'price', '(current - highest)/highest = (10%, 15%)'),
		('1_3', 'price', 'W52 highest < current'),
		('1_4', 'price', 'MA(18) > MA(22)'),
		('1_5', 'price', 'MA(250) > current in BULL | MA(250) < current in BEAR'),
		('2_1', 'volume', 'MA(22) : current'),
	]
	close = np.array(data.close)
	volume = np.array([float(v) for v in data.volume])
	current_close, min_close, max_close = close[-1], min(close), max(close)
	current_vol, min_vol, max_vol = volume[-1], min(volume), max(volume)
	volume_sma = talib.SMA(volume)
	close_sma18 = talib.SMA(close, timeperiod=18)
	close_sma22 = talib.SMA(close, timeperiod=22)
	close_sma250 = talib.SMA(close, timeperiod=250)

	cp11 = "{0:.2%}".format((current_close - min_close) / min_close)
	cp12 = "{0:.2%}".format((current_close - max_close) / max_close)
	cp13 = "W52 highest = {0:.2f}, current = {1:.2f}, delta={2:.02f}".format(max_close, close[-1], max_close - close[-1])
	cp14 = "MA(18) = {0:.2f}, MA(22) = {1:.2f}, delta={2:.02f}".format(close_sma18[-1], close_sma22[-1], close_sma18[-1] - close_sma22[-1] )
	cp15 = "MA(250) = {0:.2f}, current = {1:.2f}, delta={2:.02f}".format(close_sma250[-1], close[-1], close_sma250[-1] - close[-1])
	cp21 = "MA(22) = {0:,.0f}, current = {1:,.0f}, delta = {2:.2%}".format(volume_sma[-1], volume[-1], (volume[-1] - volume_sma[-1])/volume_sma[-1])


	result = {
		'1_1' : cp11,
		'1_2' : cp12,
		'1_3' : cp13,
		'1_4' : cp14,
		'1_5' : cp15,
		'2_1' : cp21,
	}
	return result

def checklistStatsAll() -> dict:
	return {s:checklistStats(s) for s in getSymbolList()}

# print(talib.EMA([1,2,3,4,4,5,12,12]))