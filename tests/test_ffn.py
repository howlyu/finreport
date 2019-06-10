import ffn as fn
import QUANTAXIS as QA
import pandas as pd


symbols = ['WBAI', 'TSLA']
startdate = "2019-05-01"
enddate = "2019-05-30"
data = QA.QA_fetch_stock_day_adv(symbols, startdate, enddate)

dn = data.to_pd().close.reset_index().set_index("date")


frame = {}
for s in symbols:
	frame.update({s : dn[dn.code == s].close})

df = pd.DataFrame(frame)

perf = fn.calc_stats(df)

print(perf.stats)

