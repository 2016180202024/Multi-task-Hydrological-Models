import numpy as np
import datetime as dt
import pandas as pd


def date_to_year(date_array):
    year_array = []
    for date in date_array:
        if isinstance(date, np.datetime64):
            year_array.append(pd.to_datetime(date).date().year)
        elif isinstance(date, dt.datetime) or isinstance(date, dt.date):
            year_array.append(date.year)
        elif isinstance(date, str):
            year_array.append(date[0:4])
        else:
            raise Exception
    return np.array(year_array)


def date_to_doy(date: dt.date):
    d1 = date.toordinal()  # 获取该日期在公历中的序数
    d0 = dt.date(date.year, 1, 1).toordinal()  # 这一年1月1日的序数
    return d1 - d0 + 1


def date_array_to_doy(date_list: np.array(dt.date)):
    shape = date_list.shape
    date_list = date_list.flatten()
    doy = np.empty(date_list.shape[0])
    index = 0
    for date in date_list:
        d1 = date.toordinal()  # 获取该日期在公历中的序数
        d0 = dt.date(date.year, 1, 1).toordinal()  # 这一年1月1日的序数
        doy[index] = d1 - d0 + 1
        index += 1
    return doy.reshape(shape)


def doy_array_to_month(doy_list: np.ndarray):
    shape = doy_list.shape
    doy_list = doy_list.flatten()
    month_list = np.empty(doy_list.shape[0], dtype=int)
    index = 0
    for doy in doy_list:
        base = dt.date(1950, 1, 1)
        time = base + dt.timedelta(doy - 1)
        month = time.month
        month_list[index] = int(month)
        index += 1
    return month_list.reshape(shape)


def get_days_in_year(year: int):
    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year+1, 1, 1)
    days = (end_date - start_date).days
    return days


# startDate: 2020-01-01, endDate: 2024-01-01, isNoLeapYear: True (去掉闰年的多一天)
def get_date_range(startDate, endDate, isNoLeapYear=False, freq='D'):
    date_list = pd.date_range(start=startDate, end=endDate, freq=freq).to_series()
    if isNoLeapYear:
        date_list = date_list[~((date_list.dt.month == 2) & (date_list.dt.day == 29))]
    return date_list
