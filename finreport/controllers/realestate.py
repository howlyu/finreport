# -*- coding: utf-8 -*-
from flask import render_template
from flask import Blueprint

from ..services import realestate

buleprint = Blueprint("realestate", __name__, url_prefix="/realestate")

@buleprint.route("/")
def presale_list():
	return  render_template("realestate/realesate.html",
	                        presale_list = realestate.getPresaleList())