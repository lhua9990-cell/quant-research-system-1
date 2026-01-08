import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    df = df.astype({
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'preclose': float,
        'volume': float,
        'amount': float,
        'turn': float,
        'pctChg': float,
        'tradestatus': int,
        'isST': int
    })
    return df.to_dict('records')

def store_data_in_mongodb(data, collection):
    collection.insert_many(data)

def main():
    data_directory = "./data/"  # 替换为您的数据目录路径
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["quant_database"]  # 替换为您的数据库名称
    collection = db["A_stock_daily"]  # 替换为您的集合名称

    # for file in os.listdir(data_directory):
    #     if file.endswith(".csv"):
    #         file_path = os.path.join(data_directory, file)
    #         data = process_csv_file(file_path)
    #         store_data_in_mongodb(data, collection)
    #
    # collection.create_index([("date", 1), ("code", 1)])


    # 查询所有文档
    documents = collection.find()
    # 遍历文档，将字符串日期转换为日期类型，并更新文档
    for doc in documents:
        # 假设字符串日期存储在名为"date"的字段中
        str_date = doc["date"]
        datetime_obj = datetime.strptime(str_date, "%Y-%m-%d")  # 根据实际的日期格式修改

        # 更新文档，添加一个名为"datetime"的新字段，并将其设置为转换后的日期
        collection.update_one({"_id": doc["_id"]}, {"$set": {"datetime": datetime_obj}})

    # 验证日期已成功更新
    sample_document = collection.find_one()
    print(sample_document)

if __name__ == "__main__":
    main()
