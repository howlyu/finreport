# -*- coding: utf-8 -*-
import os
from datetime import date, timedelta

import QUANTAXIS as QA
import ffn
import numpy as np
import pandas as pd
import talib
from dotenv import load_dotenv
from pyecharts_v12.charts import Bar, Kline, Grid, Line
from pyecharts_v12.options import global_options as gopts
from pyecharts_v12.options import series_options as sopts
from talib import MA_Type


def getSymbolListFromSetting() -> list:
	load_dotenv(override=True)
	return [i.strip() for i in os.getenv('SYMBOLS', default='wbai, tsla').split(",")]


def getDateFromSetting():
	load_dotenv(override=True)
	return (date.today() - timedelta(days=int(os.getenv('START_INTERVAL', default='365')))).strftime("%Y-%m-%d"), \
	       (date.today() - timedelta(days=int(os.getenv('END_INTERVAL', default='1')))).strftime("%Y-%m-%d")


def getWatchlist(source='qa'):
	'''
	get stats from all the symbols
	:param source:
	:return: stats
	'''
	startdate, enddate = getDateFromSetting()
	if source == 'ffn':
		data = getDataFromFFN(','.join(getSymbolListFromSetting()), startdate, enddate)
		perf = data.calc_stats()
		return perf.stats
	else:
		data = getDataFromQA([s.upper() for s in getSymbolListFromSetting()], startdate=startdate, enddate=enddate)
		tdata = transQAToFFN(data)
		tdata.columns = map(str.lower, tdata.columns)
		perf = ffn.calc_stats(tdata)
		return perf.stats


def getDataFromQA(symbols, startdate, enddate):
	return QA.QA_fetch_stock_day_adv(symbols, startdate, enddate).to_pd().reset_index().set_index("date")


def getDataFromFFN(symbols, startdate, enddate):
	return ffn.get(symbols, start=startdate, end=enddate)


def getSymbolList():
	return ffn.utils.clean_tickers(getSymbolListFromSetting())


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
		res.append(["{0:.2f}".format(r) for r in row if r != "nan"])
	return res


def getKlineBySymbol(symbol="TSLA") -> Grid:
	'''
	generate the kline chart for each symbol
	:param symbol:
	:return: Grid Chart
	'''
	startdate, enddate = getDateFromSetting()
	data = getDataFromQA(symbol.upper(), startdate=startdate, enddate=enddate)
	datetime = [i.strftime("%Y-%m-%d") for i in data.index]
	ohlc = formatToFloat(np.array(data.loc[:, ['open', 'close', 'low', 'high']]))
	vol = [v for v in data.volume]
	sma18_close = ["{0:.2f}".format(r) for r in np.array(talib.SMA(data.close, timeperiod=18)) if r != "nan"]
	sma22_close = ["{0:.2f}".format(r) for r in np.array(talib.SMA(data.close, timeperiod=22)) if r != "nan"]
	# print(talib.bolling)
	upper, middle, lower = talib.BBANDS(data.close, timeperiod=20, matype=MA_Type.T3)
	bbands_upper_close = ["{0:.2f}".format(r) for r in np.array(upper) if r != "nan"]
	bbands_middle_close = ["{0:.2f}".format(r) for r in np.array(middle) if r != "nan"]
	bbands_lower_close = ["{0:.2f}".format(r) for r in np.array(lower) if r != "nan"]

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

	sma_line = Line() \
		.add_xaxis(datetime) \
		.add_yaxis("sma18",
	               sma18_close,
	               is_connect_nones=False,
	               is_symbol_show=False,
	               is_smooth=True,
	               linestyle_opts=gopts.LineStyleOpts(width=2,
	                                                  color='#ff4500'),
	               ) \
		.add_yaxis("sma22",
	               sma22_close,
	               is_connect_nones=False,
	               is_symbol_show=False,
	               is_smooth=True,
	               linestyle_opts=gopts.LineStyleOpts(width=2,
	                                                  color='#338AFF'
	                                                  ),
	               )

	bbands_line = Line() \
		.add_xaxis(datetime) \
		.add_yaxis("upper",
	               bbands_upper_close,
	               is_connect_nones=False,
	               is_symbol_show=False,
	               is_smooth=True,
	               linestyle_opts=gopts.LineStyleOpts(width=2,
	                                                  color='#ff4500'),
	               ) \
		.add_yaxis("middle",
	               bbands_middle_close,
	               is_connect_nones=False,
	               is_symbol_show=False,
	               is_smooth=True,
	               linestyle_opts=gopts.LineStyleOpts(width=2,
	                                                  color='#338AFF'
	                                                  ),
	               ) \
		.add_yaxis("lower",
	               bbands_lower_close,
	               is_connect_nones=False,
	               is_symbol_show=False,
	               is_smooth=True,
	               linestyle_opts=gopts.LineStyleOpts(width=2,
	                                                  color='#ff4500'
	                                                  ),
	               )

	kline.overlap(bbands_line)

	bar = Bar(init_opts=gopts.InitOpts(height="50px")) \
		.add_xaxis(datetime) \
		.add_yaxis("volume", vol) \
		.set_series_opts(
		label_opts=sopts.LabelOpts(is_show=False),

	) \
		.set_global_opts(datazoom_opts=gopts.DataZoomOpts(xaxis_index=[0, 1],
	                                                      is_show=True,
	                                                      range_start=75,
	                                                      range_end=100, ),
	                     legend_opts=gopts.LegendOpts(is_show=False),
	                     yaxis_opts=gopts.AxisOpts(is_show=False))

	grid = Grid()  # init_opts=gopts.InitOpts(width="800px", height="600px")
	grid.add(bar, grid_opts=gopts.GridOpts(pos_top="85%"))
	grid.add(kline, grid_opts=gopts.GridOpts(pos_bottom="20%"))

	return grid


def checklistStats(symbol="TSLA", start=(date.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
                   end=(date.today() - timedelta(days=1)).strftime("%Y-%m-%d")) -> pd.DataFrame:
	'''
	check all the point by symbol, then get the statistical data
	:param symbol:
	:return:DataFrame
	'''
	startdate, enddate = getDateFromSetting()
	data = getDataFromQA(symbol.upper(), startdate=startdate, enddate=enddate)  # chart data
	data_cp = getDataFromQA(symbol.upper(), startdate=start, enddate=end)  # chart data

	desc = [
		('1_1', 'price', '(current - lowest)/lowest = (10%, 15%) '),
		('1_2', 'price', '(current - highest)/highest = (10%, 15%)'),
		('1_3', 'price', 'W52 highest < current'),
		('1_4', 'price', 'MA(18) > MA(22)'),
		('1_5', 'price', 'MA(250) > current in BULL | MA(250) < current in BEAR'),
		('2_1', 'volume', 'MA(22) : current'),
	]
	df = pd.DataFrame(columns=['id','signal','desc','delta'],
	                  index=['11','12','13','14','15','21'])
	close = np.array(data.close)
	close_cp = np.array(data_cp.close)
	volume_cp = np.array([float(v) for v in data_cp.volume])
	current_close_cp, min_close_cp, max_close_cp = close_cp[-1], min(close_cp), max(close_cp)
	current_vol_cp, min_vol, max_vol = volume_cp[-1], min(volume_cp), max(volume_cp)
	volume_ema = talib.EMA(volume_cp)
	close_ema18 = talib.EMA(close, timeperiod=18)
	close_ema22 = talib.EMA(close, timeperiod=22)
	close_ema250 = talib.EMA(close, timeperiod=250)

	delta11 = (current_close_cp - min_close_cp) / min_close_cp
	cp11 = "current={0:.2f}, min_close={1:.2f}".format(current_close_cp, min_close_cp)
	delta12 = (current_close_cp - max_close_cp) / max_close_cp
	cp12 = "current={0:.2f}, max_close={1:.2f}".format(current_close_cp, max_close_cp)
	delta13 = max(close) - close[-1]
	cp13 = "W52 highest = {0:.2f}, current = {1:.2f}".format(max(close), close[-1])
	delta14 = close_ema18[-1] - close_ema22[-1]
	cp14 = "EMA(18) = {0:.2f}, EMA(22) = {1:.2f}".format(close_ema18[-1], close_ema22[-1])
	delta15 = close_ema250[-1] - close[-1]
	cp15 = "EMA(250) = {0:.2f}, current = {1:.2f}".format(close_ema250[-1], close[-1])
	delta21 = (volume_cp[-1] - volume_ema[-1]) / volume_ema[-1]
	cp21 = "EMA(22) = {0:,.0f}, current = {1:,.0f} ".format(volume_ema[-1], volume_cp[-1])

	df.loc['11'] = ['11', delta11 >= 0.1, cp11, "delta={:.2%}".format(delta11)]
	df.loc['12'] = ['12', delta12 <= -0.1, cp12, "delta={:.2%}".format(delta12)]
	df.loc['13'] = ['13', delta13 <= 0, cp13, "delta={:.02f}".format(delta13)]
	df.loc['14'] = ['14', delta14 >= 0, cp14, "delta={:.02f}".format(delta14)]
	df.loc['15'] = ['15', delta15 <= 0, cp15, "delta={:.02f}".format(delta15)]
	df.loc['21'] = ['21', delta21 >= 0.2, cp21, "delta={:.2%}".format(delta21)]

	return df
