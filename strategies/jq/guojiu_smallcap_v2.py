# -*- coding: utf-8 -*-
"""国九小市值策略【年化100.5%|回撤25.6%】

【移植说明】
    本文件由 scripts/port_joinquant.py 自动生成：
    **策略主体与原版逐行一致**，只替换了文件头的 import。
        from jqdata import *     -> from joinquant import *
        from jqfactor import *   -> 删除（未使用）
        from six import BytesIO  -> from io import BytesIO
    数据与撮合的近似见 docs/聚宽策略移植报告.md。

    原文件：添加399101可选-Clone.py
    来源  ：https://www.joinquant.com/post/47791
"""
from io import BytesIO  # noqa: F401
from joinquant import *  # noqa: F401,F403

# 克隆自聚宽文章：https://www.joinquant.com/post/53707
# 标题：国九小市值改进：十一年百倍收益，低回撤，贴近实盘
# 作者：Alpha_壶铃

# 克隆自聚宽文章：https://www.joinquant.com/post/47791
# 标题：国九小市值策略【年化100.5%|回撤25.6%】
# 作者：zycash

#enable_profile()
#本策略为www.joinquant.com/post/47346的改进版本
#根据国九条，筛选股票
#导入函数库

import numpy as np
import pandas as pd
from datetime import time
import pickle as pickle
import datetime as dt
from dateutil.relativedelta import relativedelta

#初始化函数 
def initialize(context):
    # 开启防未来函数
    set_option('avoid_future_data', True)
    # 设定基准
    set_benchmark('000985.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 将滑点设置为0
    set_slippage(PriceRelatedSlippage(6/1000))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=2.5/10000, close_commission=2.5/10000, close_today_commission=0, min_commission=5),type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'debug')
    #初始化全局变量 bool
    g.trading_signal = True  # 是否为可交易日
    g.filter_audit = False  # 是否筛选审计意见
    g.adjust_num = False  # 是否调整持仓数量
    g.index = '399101.XSHE'  # [移植改动] 微盘股指数 tiny_index.csv 本库没有，改用代码里自带的 399101 分支 #  如需使用中小板综指，用'399101.XSHE (ZXZZ)'替换
    #全局变量list
    g.hold_list = [] #当前持仓的全部股票    
    g.yesterday_HL_list = [] #记录持仓中昨日涨停的股票
    g.target_list = []
    g.limitup_stocks = []   # 记录涨停的股票避免再次买入
    #全局变量float/str
    g.min_mv = 10  # 股票最小市值要求
    g.max_mv = 1e8  # 股票最大市值要求
    g.stock_num = 15 # 持股数量
    #g.stoploss_market = -0.05  # 市场趋势止损参数
    g.etf = '511880.XSHG'  # 空仓月份持有银华日利ETF
    # 获取当前日期
    current_date = context.current_dt.date()
    # 计算从当前日期开始的未来14天
    end_date = current_date + dt.timedelta(days=14)
    # 获取未来14天内的交易日列表
    trade_days = get_trade_days(start_date=current_date, end_date=end_date)
    g.trading_day = trade_days[0] #设置调仓日，策略运行第一天建仓，后续在cut_loss函数设置
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_daily(check_limit_up, time='14:00', reference_security='399101.XSHE') #检查持仓中的涨停股是否需要卖出
    run_daily(time_choosing, time='10:00') # 止损函数
    run_daily(check_remain_amount, '10:05')
    run_daily(close_account, '14:50')
    run_daily(position_adjustment,'10:01')
    run_weekly(print_trade_info, 2, time='15:10', reference_security='000300.XSHG')

#1-1 准备股票池
def prepare_stock_list(context):
    #获取已持有列表
    g.hold_list= []
    g.limitup_stocks = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)  #读入持仓股
    #获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close','high_limit','low_limit'], count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []
    #判断今天是否为账户资金再平衡的日期
    #g.trading_signal = today_is_between(context)

#1-2 选股模块
def get_stock_list(context):
    final_list = []
    initial_list = filter_stocks(context, get_index_stocks('000985.XSHG'))
    #添加热门行业过滤
    #initial_list = list(set(initial_list) & set(get_hot_industry_stock(context)))
    # 国九更新：过滤近一年净利润为负且营业收入小于1亿的
    # 国九更新：过滤近一年期末净资产为负的 (经查询没有为负数的，所以直接pass这条)
    # 国九更新：过滤近一年审计建议无法出具或者为负面建议的 (经过净利润等筛选，审计意见几乎不会存在异常)
    q = query(
        valuation.code,
    ).filter(
        valuation.code.in_(initial_list),
        valuation.market_cap.between(g.min_mv,g.max_mv),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8
    ).order_by(valuation.market_cap.asc()).limit(g.stock_num*3)
    df = get_fundamentals(q)
    if g.filter_audit is True:
        # 如果筛选审计意见会大幅度增加回测时常
        before_audit_filter = len(df)
        df['audit'] = df['code'].apply(lambda x: filter_audit(context, x))
        df_audit = df[df['audit'] == True]
        log.info('去除掉了存在审计问题的股票{}只'.format(len(df)-before_audit_filter))
    final_list = list(df.code)
    if len(final_list) == 0:
        # 由于有时候选股条件苛刻，所以会没有股票入选，这时买入银华日利ETF
        log.info('无适合股票，买入ETF')
        return [g.etf]
    else:
        return final_list
#取得下个月的第一个交易日
def get_next_trading_day(context):
    # 获取当前日期
    current_date = context.current_dt.date()
    # 计算下个月的第一天
    next_month_first_day = (current_date + relativedelta(months=1)).replace(day=1)
    # 计算从下个月的第一天开始的未来30天
    end_date = next_month_first_day + dt.timedelta(days=30)
    # 获取从下个月的第一天到未来30天内的交易日列表
    trade_days = get_trade_days(start_date=next_month_first_day, end_date=end_date)
    # 返回第一个交易日
    return trade_days[0]

#1-3 整体调整持仓
def position_adjustment(context):
    today = context.current_dt.date()
    #一般情况下trading_signal为True，如果前面有大跌，则为False，这种情况下等待指数站上M10，重置为True。
    if g.trading_signal and (today == g.trading_day):
        g.target_list = get_stock_list(context)[:g.stock_num]
        log.info(str(g.target_list))
        sell_list = [stock for stock in g.hold_list if stock not in g.target_list and stock not in g.yesterday_HL_list]
        hold_list = [stock for stock in g.hold_list if stock in g.target_list or stock in g.yesterday_HL_list]
        log.info("已持有[%s]" % (str(hold_list)))
        log.info("卖出[%s]" % (str(sell_list)))
            
        sell_positions = [context.portfolio.positions[stock] for stock in sell_list]
        for position in sell_positions:
            close_position(position)
        buy_security(context, g.target_list)
            
        for position in list(context.portfolio.positions.values()):
            stock = position.security
        #建仓/调仓之后，设定调仓日为下周第二个交易日。在满仓状态（trading_signal 为 True）下，每周调仓一次。
        g.trading_day = get_next_trading_day(context)
    #当前一天大跌，cut_loss函数已将 trading_signal设为 False
    elif g.trading_signal == False: 
        buy_security(context, [g.etf])
        log.info('MA指示指数大跌，持有银华日利ETF')
        # 获取当前日期
        current_date = context.current_dt.date()
        # 计算从当前日期开始的未来14天
        end_date = current_date + dt.timedelta(days=14)
        # 获取未来14天内的交易日列表
        trade_days = get_trade_days(start_date=current_date, end_date=end_date)
        # 设定调仓日为下一个交易日，在空仓状态（trading_signal 为 False）下，每天的cut_loss函数都会检测是否满足建仓条件。
        g.trading_day = trade_days[1]  #第0个交易日为当天

#1-4 调整昨日涨停股票
def check_limit_up(context):
    now_time = context.current_dt
    if g.yesterday_HL_list != []:
        #对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
        for stock in g.yesterday_HL_list:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close','high_limit'], skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0,0] <    current_data.iloc[0,1]:
                log.info("[%s]涨停打开，卖出" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
                g.limitup_stocks.append(stock)
            else:
                log.info("[%s]涨停，继续持有" % (stock))

#1-5 如果昨天有股票卖出或者买入失败，剩余的金额今天早上买入
def check_remain_amount(context):
    if g.trading_signal:
        g.hold_list= []
        for position in list(context.portfolio.positions.values()):
            stock = position.security
            g.hold_list.append(stock)
        if len(g.hold_list) < g.stock_num:
            # 计算需要买入的股票数量
            num_stocks_to_buy = min(len(g.limitup_stocks), g.stock_num - len(context.portfolio.positions))
            target_list = [stock for stock in g.target_list if stock not in g.limitup_stocks][:num_stocks_to_buy]
            log.info('有余额可用'+str(round((context.portfolio.cash),2))+'元。买入'+ str(target_list))
            buy_security(context,target_list)
    else :
        log.info('有余额可用'+str(round((context.portfolio.cash),2))+'元。买入'+ str(g.etf))
        buy_security(context,[g.etf])
        
#1-7 择时函数
def time_choosing(context):
    yesterday = context.previous_date#.strftime('%Y-%m-%d')
    if g.index == '800007.choice' :
        #g.trading_signal = True
        current_positions = context.portfolio.positions
        #读入微盘指数数据
        filename = 'tiny_index.csv'
        # 使用 read_file 函数读取文件内容
        file_content = read_file(filename)
        # 使用 BytesIO 将文件内容转换为文件对象
        df_tiny_index = pd.read_csv(BytesIO(file_content), parse_dates=['date'], index_col='date')
    
        # 使用检测到的编码读取文件
        #df_tiny_index.set_index(df_tiny_index.columns[0], inplace=True)
        df_tiny_index.dropna(inplace=True)
        # 指数前一日涨跌幅
        change_pct = df_tiny_index.loc[yesterday,'change_pct']#取得昨日指数跌幅
        ma10 = df_tiny_index.loc[yesterday,'M10']
    elif g.index == '399101.XSHE':
        df_tiny_index = get_price(security='399101.XSHE', end_date=yesterday, frequency='daily', fields=['close'], count=10, panel=False)
        ma10 = df_tiny_index['close'].mean()
    #当昨日close 大于 昨日 MA10，则交易标志设为 True
    if df_tiny_index.loc[yesterday,'close'] >= ma10:
        g.trading_signal = True
        log.debug('大盘安全，可交易')
    # 市场大跌止损或特定期间止损（当指数位于M10之下，特定时间出发止损）
    elif is_between(context) : #and 'change_pct/100 <= g.stoploss_market:
        g.trading_signal = False #微盘股大跌，清仓，停止交易,如果前面打开了标志，这里会重新关上，对应在MA10之上暴跌的情况。
        #g.cut_loss_signal = True #止损状态打开
        log.debug("大盘警戒区域，清仓！！")
        #log.debug("大盘惨跌,平均降幅{:.2%}，清仓！！".format(change_pct/100))
        close_account(context)
        print(g.trading_signal)

#1-8 动态调仓代码
def adjust_stock_num(context):
    """Empty function to complete next edition"""
    return result
    

#2 过滤各种股票
def filter_stocks(context, stock_list):
    current_data = get_current_data()
    filtered_stocks = []
    stock_list = [stock for stock in stock_list if not (
            (current_data[stock].day_open == current_data[stock].high_limit) or  # 涨停开盘
            (current_data[stock].day_open == current_data[stock].low_limit) or  # 跌停开盘
            current_data[stock].paused or  # 停牌
            current_data[stock].is_st or  # ST
            ('ST' in current_data[stock].name) or
            ('*' in current_data[stock].name) or
            ('退' in current_data[stock].name) or
            (stock.startswith('30')) or  # 创业
            (stock.startswith('68')) or  # 科创
            (stock.startswith('8')) or  # 北交
            (stock.startswith('4'))    # 北交
            #or (stock.startswith('30'))   # 创业板
        )]
    # 次新股过滤
    for stock in stock_list:
        start_date = get_security_info(stock).start_date
        if context.previous_date - start_date < timedelta(days=375):
            continue
        filtered_stocks.append(stock)
    return filtered_stocks

#2.1 筛选审计意见
def filter_audit(context, code):
    # 获取审计意见，近三年内如果有不合格(report_type为2、3、4、5)的审计意见则返回False，否则返回True
    lstd = context.previous_date
    last_year = (lstd.replace(year=lstd.year - 3, month=1, day=1)).strftime('%Y-%m-%d')
    q=query(finance.STK_AUDIT_OPINION).filter(finance.STK_AUDIT_OPINION.code==code,finance.STK_AUDIT_OPINION.pub_date>=last_year)
    df=finance.run_query(q)
    df['report_type'] = df['report_type'].astype(str)
    contains_nums = df['report_type'].str.contains(r'2|3|4|5')
    return not contains_nums.any()

#3-1 交易模块-自定义下单
def order_target_value_(security, value):
    if value == 0:
        pass
        #log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)

#3-2 交易模块-开仓
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False

#3-3 交易模块-平仓
def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False

#3-4 买入模块
def buy_security(context,target_list):
    #调仓买入
    position_count = len(context.portfolio.positions)
    target_num = len(target_list)
    if target_num > position_count:
        value = context.portfolio.cash / (target_num - position_count)
        for stock in target_list:
            if context.portfolio.positions[stock].total_amount == 0:
            #if stock not in context.portfolio.positions:
                if open_position(stock, value):
                    log.info("买入[%s]（%s元）" % (stock,value))
                    if len(context.portfolio.positions) == target_num:
                        break

#4-1 判断今天是否跳过月份
def is_between(context):
    # 获取当前日期
    current_date = context.current_dt.date()
    # 获取当前年份
    year = current_date.year
    # 定义四个时间段
    periods = [
        (datetime.datetime(year, 1, 1), datetime.datetime(year, 1, 31)),
        (datetime.datetime(year, 4, 1), datetime.datetime(year, 4, 30)),
        (datetime.datetime(year, 6, 1), datetime.datetime(year, 6, 30)),
        (datetime.datetime(year, 12, 15), datetime.datetime(year, 12, 31))
    ]
    # 判断当前日期是否在任何一个时间段内
    for start, end in periods:
        if start.date() <= current_date <= end.date():
            return True
    return False

#4-2 清仓后次日资金可转
def close_account(context):
    if g.trading_signal == False:
        if len(g.hold_list) != 0 and g.hold_list != [g.etf]:
            for stock in g.hold_list:
                position = context.portfolio.positions[stock]
                close_position(position)
                log.info("卖出[%s]" % (stock))

def print_trade_info(context):
    #打印当天成交记录
    for position in list(context.portfolio.positions.values()):
        securities=position.security
#        cost=position.avg_cost
#        price=position.price
#        ret=100*(price/cost-1)
#        value=position.value
#        amount=position.total_amount    
        print(' {} {}'.format(securities,get_security_info(securities).display_name))
#        print('成本价:{}'.format(format(cost,'.2f')))
#        print('现价:{}'.format(price))
#        print('收益率:{}%'.format(format(ret,'.2f')))
#        print('持仓(股):{}'.format(amount))
#        print('市值:{}'.format(format(value,'.2f')))
#        print('———————————————————————————————————————分割线1————————————————————————————————————————')
    print("当日收盘账户可用资金 %d" % context.portfolio.available_cash)
    print('共持有股票:{}支'.format(len(context.portfolio.positions)))
    print('———————————————————————————————————————分割线2————————————————————————————————————————')    
#获取市场热门行业    ,not used
def get_hot_industry_stock(context, count=10, number=24):
    end_date = context.previous_date
    by_date = get_trade_days(end_date=end_date,count=count+20)[0]
    stock_list = get_all_securities(date=by_date).index.tolist()  # count+20个交易日之前就已经上市的
    df_close = get_price(stock_list,end_date=end_date, count=count+20, fields='close', panel=False).pivot(index='time', values='close', columns='code')
    df_bias = df_close.iloc[20:] > df_close.rolling(20).mean().iloc[20:]  # C > MA20
    df_industries = get_industries('sw_l1', date=end_date)
    df_industries['code'] = list(df_industries.index)
    df=pd.DataFrame()
    columns = set(df_bias.columns)
    for idx, row in df_industries.iterrows():
        ind_stocks = set(get_industry_stocks(idx, date=end_date))  # 行业成分股
        ind_avail_stocks = list(columns & ind_stocks) # 成分股 在df_bias表中存在的
        if ind_avail_stocks:
            # 计算该行业成分股C>MA20的百分比，技巧：df_bias[ind_avail_stocks]
            df[row['code']] = (100*(df_bias[ind_avail_stocks].sum(axis=1))/len(ind_avail_stocks)).astype(int)
    df.sort_index(ascending=False, inplace=True)
    sr = df.iloc[0:count,:].mean().sort_values(ascending=False, inplace=False)
    sr = list(sr[0:number].index)
    stocks_set = set()
    for s in sr:
        ind_stocks = set(get_industry_stocks(s, date=end_date))
        stocks_set.update(ind_stocks)
        # get_industry_stocks(sr, date=end_date)
    return list(stocks_set)
