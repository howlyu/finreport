# -*- coding: utf-8 -*-
from flask import render_template
from flask import Blueprint

from ..services import properties

buleprint = Blueprint("properties", __name__, url_prefix="/properties")

@buleprint.route("/")
def presale_list():
	return  render_template("properties/properties.html",
	                        presale_list = properties.getPresaleList())