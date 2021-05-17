import numpy as np
import datetime as dt
from collections import namedtuple

Activity = namedtuple("Activity", ["name", "start", "end", "tags"])
work = object()

AM=0
PM=12
log={dt.date(2019,9,24):[
	Activity(name="paper", start=dt.time(11+AM,0), end=dt.time(12+AM,00), tags=[work]),
	Activity(name="lunch", start=dt.time(12+AM,0), end=dt.time(12+AM,30), tags=[]),
	Activity(name="coffee", start=dt.time(1+PM,0), end=dt.time(2+PM,0), tags=[]),
	Activity(name="talking with nico", start=dt.time(2+PM,0), end=dt.time(3+PM,0), tags=[]),
	Activity(name="talking with binghan", start=dt.time(3+PM,0), end=dt.time(3+PM,20), tags=[]),
	Activity(name="writing this program", start=dt.time(3+PM,20), end=dt.time(3+PM,40), tags=[])
	]}