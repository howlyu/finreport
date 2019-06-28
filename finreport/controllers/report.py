# -*- coding: utf-8 -*-
from flask import render_template
from flask import Blueprint
from jinja2 import Markup
import json

from finreport.services import microdata

blueprint = Blueprint('report', __name__, url_prefix='/report')


@blueprint.route('/')
def finReport():
	return render_template('reports/report.html', watchlist=microdata.getWatchlist(source='qa'),
	                       symbols=microdata.getSymbolList())

@blueprint.route("klineChart/<string:symbol>")
def getKlineChart(symbol=None):
	return microdata.getKlineBySymbol(symbol).dump_options()

@blueprint.route("checkpointStat/<string:symbol>/<string:start>/<string:end>")
def showCheckPoint(symbol=None, start=None, end=None):
	# return json.dumps(microdata.checklistStats(symbol=symbol, start=start, end=end))
	return render_template("reports/report_checklist.html",
	                       symbol=symbol,
	                       start=start,
	                       end=end,
	                       result=microdata.checklistStats(symbol=symbol, start=start, end=end))