# -*- coding: utf-8 -*-
from flask import render_template
from flask import Blueprint
from jinja2 import Markup

from finreport.services import microdata

blueprint = Blueprint('report', __name__, url_prefix='/report')


@blueprint.route('/')
def finReport():
	return render_template('reports/report.html', watchlist=microdata.getWatchlist(source='qa'),
	                       symbols=microdata.getSymbolList(),
	                       barchart=Markup(microdata.getKlineBySymbol().render_embed()))

@blueprint.route("klineChart/<symbol>")
def getKlineChart(symbol=None):
	return microdata.getKlineBySymbol(symbol).dump_options()