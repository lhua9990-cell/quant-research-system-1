import pymongo
import pandas as pd
from datetime import datetime, timedelta
from core.utils import metric
import argparse

def buy_stock(stock_code, amount, price):
    global capital
    global position
    shares = int(amount / (price * 100)) * 100
    cost = shares * price
    if shares == 0: # amount limit
        return 0
    if cost <= capital:
        capital -= cost
        position[stock_code] = {'shares':shares, 'buy_price':price}

def rebuy_stock(stock_code, amount, price):
    global capital
    global position
    shares = int(amount / (price * 100)) * 100
    if shares == 0: # price is too high to buy
        return 0
    cost = shares * price
    if cost <= capital:
        capital -= cost
        total_shares = position[stock_code]['shares'] + shares
        new_buy_price = (position[stock_code]['shares']*position[stock_code]['buy_price'] + shares*price)/total_shares
        position[stock_code]['buy_price'] = new_buy_price
        position[stock_code]['shares'] = total_shares
    return 0


def sell_stock(stock_code, price):
    global capital
    global position
    shares = position[stock_code]['shares']
    capital += shares * price
    position.pop(stock_code, 0)

@metric
def main():

    global capital
    global position
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["quant_database"]
    collection = db["A_stock_daily"]

    for capital in range(1000000,2000000,1000000):
        print(capital)
        initial_money = capital

        start_date = datetime(2023, 3, 1)
        end_date = datetime.now()

        limit_amount = 50000
        stop_profit = 0.2
        position = {}
        position_history = []
        capital_history = []
        print_year = start_date.year
        print(print_year)

        # statistics of win probability
        total_cnt = 0
        win_cnt = 0
        lose_cnt = 0

        while start_date <= end_date:
            str_start_date = str(start_date).split(' ')[0]
            print('new day {}'.format(start_date))
            if start_date.year != print_year:
                print(start_date.year)
                print_year = start_date.year

            # first sell then buy
            for stock_code, info in position.copy().items():
                stock_data = collection.find_one({"code": stock_code, "date": str_start_date})
                if stock_data:
                    buy_price = info['buy_price']
                    current_price = stock_data['open']
                    profit = (current_price - buy_price) / buy_price
                    if profit >= stop_profit or profit <= stop_profit * -0.5:
                        sell_stock(stock_code, current_price)
                        #print(start_date,)

            # buy
            cursor = collection.find({'date': str_start_date})
            data = pd.DataFrame(list(cursor))
            if not data.empty:
                data = data[data['amount'] > 1e9]  # Filter stocks with 'amount' greater than 1 billion
                #data['amount_diff'] = data['amount'] - data['amount'].shift(1)

                data = data[(data['isST'] == 0) & (data['tradestatus'] == 1)]
                data = data[data['pctChg'] > 0]
                #print('first search')
                #print(data)
                # find the stocks whose previous volume is smaller than 10billion but pctChg > 0
                query = {
                    "date": str(start_date-timedelta(days=1)).split(' ')[0],
                    "code": {"$in": data['code'].tolist()},
                    "pctChg": {"$gt": 0},
                    "amount": {"$lt": 1e9},
                }
                pool_cursor = collection.find(query)
                pool_data = pd.DataFrame(list(pool_cursor))
                if not pool_data.empty:
                    #print('second search')
                    #print(pool_data)
                    #break
                    pool_data = pool_data.sort_values(by='pctChg', ascending=False)
                    total_cnt = total_cnt + len(pool_data)

                for index, row in pool_data.iterrows():
                    stock_code = row['code']
                    stock_price = row['close']
                    if stock_code not in position:
                        buy_stock(stock_code, limit_amount, stock_price)
                    if stock_code in position:
                        rebuy_stock(stock_code, limit_amount, stock_price)

                    # for statistics
                    high_price_cursor = collection.find({"datetime": {"$gt": start_date, "$lte": end_date}, 'code': stock_code, 'high':{'$gte':stock_price*(1+stop_profit)}},
                                                 {'_id': 0, 'datetime': 1}).sort([("datetime", 1)]).limit(1)
                    high_price_list = list(high_price_cursor)
                    if high_price_list:
                        upper_date = list(high_price_list)[0]['datetime']


                    else:
                        upper_date = end_date
                    print(upper_date)
                    low_price = collection.find({"datetime": {"$gt": start_date, "$lte": upper_date}, 'code': stock_code, 'low':{'$lte':stock_price*(1-stop_profit*0.5)}},
                                                 {'_id': 0, 'datetime': 1}).sort([("datetime", 1)]).limit(1)
                    low_price_list = list(low_price)
                    if low_price_list:
                        lose_cnt = lose_cnt+1
                        print(low_price_list[0]['datetime'])
                    if not low_price_list: # which means we don't find the stop loss point before we stop profit.
                        win_cnt = win_cnt+1

            # record the status
            total_capital = 0
            for stock_code, info in position.items():
                stock_data = collection.find_one({"code": stock_code, "date": str_start_date})
                if stock_data:
                    current_price = stock_data['close']
                    total_capital = total_capital + info['shares']*current_price
                else:
                    total_capital = capital_history[-1][1]
                    break

            total_capital = total_capital + capital

            capital_history.append((start_date,total_capital))
            position_history.append((start_date, position))
            start_date += timedelta(days=1)

        # Calculate annualized return
        print(capital_history[-1][1])
        total_return = (capital_history[-1][1] - initial_money) / float(initial_money)
        years = (datetime.now() - datetime(2023, 1, 1)).days / 250
        annualized_return = (1 + total_return) ** (1 / years) - 1
        df = pd.DataFrame(capital_history, columns=['date','value'])
        df.to_csv('capital_history_test.csv')

        df2 = pd.DataFrame(position_history)
        df2.to_csv('position_history_test.csv')
        print(annualized_return)
        print(win_cnt,lose_cnt)
        print('strategy win probability: {}%'.format(float(win_cnt)/total_cnt))


if __name__ == '__main__':
    main()