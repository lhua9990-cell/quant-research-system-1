# get data from http://baostock.com/baostock/index.php/A%E8%82%A1K%E7%BA%BF%E6%95%B0%E6%8D%AE
# update to 2023.3.22.

import os.path
from jqdatasdk import *
from datetime import *
import datetime
import pandas as pd

import logging

import baostock as bs
import pandas as pd

data_path = './data/'

def main():

    logging.basicConfig(filename='update_A_stock_data.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    auth('13547211969','Since1491@')
    # auth('13661732733','bWc-44f-Vuq-FJK')
    # auth('15221252658', 'Since1991')
    # assert (get_query_count()['total'] > 100000)

    today = str(datetime.datetime.now()).split(' ')[0]
    stock_table = get_all_securities(types=['stock'])
    stock_table.to_csv('./data/stock_table_original.csv')

    # log in baostock web
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    for i in range( len(stock_table)):

        logging.info(i)
        logging.info('{} is in process.'.format(stock_table['display_name'][i]))

        start_date = str(stock_table['start_date'][i]).split(' ')[0]

        stock_num = stock_table.index[i].split('.')[0]
        trade_exchage = 'sh.' if stock_table.index[i][-1]=='G' else 'sz.'

        if not os.path.isfile(data_path + stock_table.index[i].split('.')[0] + '.csv'):
            # file now existed!
            print(stock_num)
            try:
                rs = bs.query_history_k_data_plus(trade_exchage +stock_num,
                                                  "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST",
                                                  start_date=start_date, end_date=today,
                                                  frequency="d", adjustflag="2")
            except:
                logging.info('{} is in error when obtain data from baostock!'.format(stock_table['display_name'][i]))
                print('{} is in error when obtain data from baostock!'.format(stock_table['display_name'][i]))
                break

            data_list = []
            while (rs.error_code == '0') & rs.next():
                # 获取一条记录，将记录合并在一起
                data_list.append(rs.get_row_data())
            df = pd.DataFrame(data_list, columns=rs.fields)
            if len(df) > 0:
                df.to_csv(data_path + stock_table.index[i].split('.')[0] + '.csv')
                print(i)
                print('{} is done.'.format(stock_table['display_name'][i]))
        #break


if __name__ == '__main__':
    main()