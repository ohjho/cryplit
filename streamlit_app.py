import streamlit as st
import os, sys, ccxt
import numpy as np
import pandas as pd
from datetime import timedelta

from toolbox.st_utils import show_logo, get_timeframe_params, show_plotly
from toolbox.plotly_utils import plotly_ohlc_chart, get_moving_average_col, \
							add_Scatter, add_Scatter_Event, add_color_event_ohlc
from toolbox.ta_utils import add_moving_average, add_MACD, add_AD, add_OBV, add_RSI, \
							add_ADX, add_Impulse, add_ATR, add_avg_penetration, \
							market_classification, efficiency_ratio
# from strategies.macd_divergence import detect_macd_divergence
# from strategies.kangaroo_tails import detect_kangaroo_tails
# from strategies.vol_breakout import detect_vol_breakout, detect_volatility_contraction, \
#                                 detect_low_vol_pullback, detect_VCP

#--------------ccxt functions -----------------#
def timeseries_date_limit(df, start_date = None, end_date = None):
	from datetime import timedelta
	df = df[df.index > pd.Timestamp(start_date)] if start_date else df
	df = df[df.index < pd.Timestamp(end_date + timedelta(days=1))] if end_date else df
	return df

def get_periods_per_day(interval):
	unit_per_day_dict = {
		'm': 60 * 24, 'h': 24, 'd': 1, 'w': 1/7, 'M': 1/28
	}
	unit = interval[-1] # assume last character is always a string
	assert unit in unit_per_day_dict, f'unit {unit} is not recognized: {unit_per_day_dict.keys()}'
	num_interval = int(interval.replace(unit,''))
	return unit_per_day_dict[unit]/num_interval

# @st.cache
def ccxt_get_ohlcv(exchange, symbol:str, period:str, start_date, end_date):
	''' returns a Pandas DF of Open, High, Low, Close, Volume using the ccxt library

	Reference: http://techflare.blog/how-to-get-ohlcv-data-for-your-exchange-with-ccxt-library/
	'''
	from datetime import datetime, date, timedelta
	import calendar
	assert exchange.has['fetchOHLCV'], f'{exchange.name} does not support fetchOHLCV'
	assert symbol in exchange.load_markets(), f'{symbol} is not a valid market for {exchange.name}'
	assert period in exchange.timeframes, f'{period} is not a supported timeframe: {exchange.timeframes}'
	days_num = (end_date - start_date).days + 1
	limit = int(days_num * get_periods_per_day(interval = period))+1
	datelist = [start_date + timedelta(days=x) for x in range(days_num)]
	datelist = [date.strftime("%Y%m%d") for date in datelist]

	ohlcv = []
	if any(['d' in period, 'w' in period, 'M' in period]):
		print(f'making single call to {exchange.name}, limit: {limit}')
		ohlcv = exchange.fetchOHLCV(symbol=symbol, timeframe = period, limit = limit,
				since = calendar.timegm(datetime.strptime(datelist[0],"%Y%m%d").utctimetuple())*1000
				)
	else:
		print(f'getting intraday candles for: {datelist}\nlimit: {limit}')
		for d in datelist: #TODO: use tqdm?
			ohlcv.extend(
				exchange.fetchOHLCV(symbol=symbol, timeframe = period, limit = limit,
					since = calendar.timegm(datetime.strptime(d,"%Y%m%d").utctimetuple())*1000
					)
			)
	df = pd.DataFrame(ohlcv, columns = ['Time','Open','High','Low','Close','Volume'])
	df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['Time']]
	df['Open'] = df['Open'].astype(np.float64)
	df['High'] = df['High'].astype(np.float64)
	df['Low'] = df['Low'].astype(np.float64)
	df['Close'] = df['Close'].astype(np.float64)
	df['Volume'] = df['Volume'].astype(np.float64)
	df.set_index('Time', inplace=True)
	df = df[~df.index.duplicated(keep='first')]
	return timeseries_date_limit(df, start_date = start_date, end_date = end_date)

def get_currencies(ccxt_markets):
	ccy = [m.split('/')[0] for m in ccxt_markets]
	return sorted(list(set(ccy)))
#------------end of ccxt functions -----------------#

def add_indicators(data, st_asset):
	cols = st_asset.columns(3)
	with cols[0]:
		st.write('#### the moving averages')
		ma_type = st.selectbox('moving average type', options = ['', 'ema', 'sma', 'vwap'])
		periods = st.text_input('moving average periods (comma separated)', value = '22,11')
		if ma_type:
			for p in periods.split(','):
				data = add_moving_average(data, period = int(p), type = ma_type,price_col='Close')

		st.write('#### MACD')
		do_MACD = st.checkbox('Show MACD?', value = True)
		if do_MACD:
			data = add_MACD(data,
					fast = st.number_input('fast', value = 12),
					slow = st.number_input('slow', value = 26),
					signal = st.number_input('signal', value = 9),
					price_col="Close" )
	with cols[1]:
		st.write('#### True Range Related')
		atr_period = int(st.number_input('Average True Range Period', value = 13))
		atr_ema = st.checkbox('use EMA for ATR', value = True)
		show_ATR = st.checkbox('show ATR?', value = False)
		if ma_type:
			st.write('##### ATR Channels')
			atr_ma_name = st.selectbox('select moving average for ATR channel',
							options = [''] + get_moving_average_col(data.columns))
			atr_channels = st.text_input('Channel Lines (comma separated)', value = "1,2,3") \
							if atr_ma_name else None
			fill_channels = st.checkbox('Fill Channels with color', value = False) \
							if atr_ma_name else None
		else:
			atr_ma_name = None
		data = add_ATR(data, period = atr_period, use_ema = atr_ema,
					channel_dict = {atr_ma_name: [float(c) for c in atr_channels.split(',')]} \
						if atr_ma_name else None
					)
	with cols[2]:
		st.write('#### oscillator')
		do_RSI = st.checkbox('RSI')
		data = add_RSI(data, n = st.number_input('RSI period', value = 13)) if do_RSI else data
		tup_RSI_hilo = st.text_input('RSI chart high and low line (comma separated):', value = '70,30').split(',') \
						if do_RSI else None
		tup_RSI_hilo = [int(i) for i in tup_RSI_hilo] if tup_RSI_hilo else None
		if do_RSI:
			data_over_hilo_pct = sum(
				((data['RSI']> tup_RSI_hilo[0]) | (data['RSI']< tup_RSI_hilo[1])))/ len(data)
			st.info(f"""
			{round(data_over_hilo_pct * 100, 2)}% within hilo\n
			5% of peaks and valley should be within hilo
			""")
		st.write('#### volume-based indicators')
		data = add_AD(data) if st.checkbox('Show Advance/ Decline') else data
		data = add_OBV(data)  if st.checkbox('Show On Balance Volume') else data
		st.write(f'##### Directional System')
		do_ADX = st.checkbox('Show ADX')
		data = add_ADX(data, period = st.number_input("ADX period", value = 13)) \
				if do_ADX else data
	return data

def test_ccxt():
	l_col, c_col, r_col = st.columns(3)
	url_params = st.experimental_get_query_params()
	l_exchanges = ccxt.exchanges
	exchange_name = r_col.selectbox('select your exchange',
					options = l_exchanges,
					index = l_exchanges.index('binance'))
	if not exchange_name:
		return None
	exchange = getattr(ccxt, exchange_name)()

	# Getting Symbol (currency pairs)
	l_ccy = get_currencies(exchange.load_markets())
	url_symbol = url_params['symbol'][0].upper().split('/') if 'symbol' in url_params else None
	if url_symbol:
		assert len(url_symbol)==2, f'expected currency pair separated by /, you provided: {url_params["symbol"]}'
	ccy = l_col.selectbox(f'Currency ({len(l_ccy)})', l_ccy,
			index = l_ccy.index(url_symbol[0] if url_symbol else 'BTC'),
			help = f'see [most active pairs](https://www.investing.com/crypto/top-pairs) or [by market cap](https://coinmarketcap.com/all/views/all/)')
	base_ccy = c_col.selectbox('Base Currency', l_ccy,
				index = l_ccy.index(url_symbol[1] if url_symbol else 'USDT'),
				help = f'use [stablecoins](https://coinmarketcap.com/view/stablecoin/) to relate to "physical" currencies')
	symbol = f'{ccy}/{base_ccy}'.upper()

	# setting timeframe
	timeframe_params = get_timeframe_params(
						st_asset = st.sidebar.expander('timeframe', expanded = True),
						default_tenor='6m', default_interval = '1d',
						data_buffer_tenor = '3m',
						l_interval_options = list(exchange.timeframes))
	if 'm' in timeframe_params['interval'] or 'h' in timeframe_params['interval']:
		timeframe_params['data_start_date'] = timeframe_params['start_date'] + timedelta(days = -2)

	st.experimental_set_query_params(symbol = symbol,
		bar = timeframe_params['interval'],
		td = timeframe_params['end_date'],
		period = timeframe_params['tenor']
		)

	chart_height = st.sidebar.slider('Chart Height', value = 1200,
						min_value = 800, max_value = 1600, step = 100)

	if symbol:
		df = ccxt_get_ohlcv(exchange = exchange,
				symbol=symbol,
				period=timeframe_params['interval'],
				start_date=timeframe_params['data_start_date'],
				end_date= timeframe_params['end_date'])

		df = add_indicators(data = df, st_asset= st.expander('indicators'))
		df = timeseries_date_limit(df, start_date = timeframe_params['start_date'])
		fig = plotly_ohlc_chart(df = df, vol_col = 'Volume')
		show_plotly(fig, height = chart_height,
			title=f'{symbol} ({timeframe_params["interval"]}) chart , exchange: {exchange.name}')

		with st.expander('view ohlcv dataframe'):
			st.write(df)
			st.download_button('download dataframe as CSV',
				data = df.to_csv().encode('utf-8'),
				file_name = f'{symbol.replace("/","")}_{timeframe_params["interval"]}.csv',
				mime="text/csv"
				)

def Main():
	st.set_page_config(
		layout = 'wide',
		page_title = 'Cryplit',
		page_icon = 'asset/app_logo_aubergine.png',
		initial_sidebar_state = 'expanded'
		)
	show_logo( str_color = 'aubergine')
	with st.sidebar.expander("cryplit"):
		st.info(f'''
		[information symmetry](https://en.wikipedia.org/wiki/Information_asymmetry) for all

		*	[project page](https://github.com/seekingvega/cryplit)
		*	data by [ccxt](https://github.com/ccxt)
		''')
	test_ccxt()
	# app_dict = {
	# 	"DESC": Desc,
	# 	"GP": GP,
	# 	"RT": RT,
	# 	"ATR": ATR,
	# 	"BETA": BETA,
	# 	"MBRS": MBRS,
	# 	"HK-DVD": HK_DVD,
	# 	"login": login,
	# }
	#
	# app_sw = st.sidebar.selectbox('select app', options = [''] + list(app_dict.keys()))
	# if app_sw:
	# 	app_func = app_dict[app_sw]
	# 	app_func()

if __name__ == '__main__':
	Main()
