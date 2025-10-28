from typing import Literal, List
from WindPy import w
import pandas as pd
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
from tqdm import tqdm
import os


def period_to_date(year: int, period: Literal['q1', 'h1', 'q2', 'y1']) -> datetime.date:
    """万得参数转化为日期"""
    period_dict = {
        'q1': (3, 31),
        'h1': (6, 30),
        'q3': (9, 30),
        'y1': (12, 31)
    }
    return datetime.date(year, period_dict[period][0], period_dict[period][1])

def get_data(code: str, year: int, period: Literal['q1', 'h1', 'q2', 'y1']) -> pd.DataFrame:
    """获取数据"""
    _err, shareholder_df = w.wset(
        "top10shareholders",
        f"windcode={code}; year={year}; period={period}", usedf=True
    )
    date = period_to_date(year, period)
    if len(shareholder_df) != 0:
        announce_date = w.wss(
            code,
            "stm_issuingdate",
            f"rptDate={date.year}{date.month:02d}{date.day:02d}"
        )[1].iloc[0, 0]
        shareholder_df['code'] = code
        shareholder_df['end_date'] = period_to_date(year, period)
        shareholder_df['ann_date'] = announce_date
    return shareholder_df

def build_label_for_single_stock(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    对股东数据打标签
    :param data:
    :return:
    """
    # 股东性质
    data = check_state_own(data)
    state_own = int(data['state_own'].astype(bool).any())

    data = check_foreign_capital(data)
    foreign_capital = int(data['foreign_own'].astype(bool).any())

    data = check_domestic_capital(data)
    domestic_capital = int(data['domestic_own'].astype(bool).any())

    # 股权结构
    threshold = {
        'concentration_top1': 0.5,  # 第一大股东股权集中阈值
        'separation_top1': 0.2,     # 第一大股东股权分散阈值
        'concentration_all10': 0.7, # 前十大股东股权集中阈值
        'separation_all10': 0.3,    # 前十大股东股权分散阈值
        'z_index_concentration': 5, # z指数表明第一大股东股权集中阈值
        'z_index_separation': 2,    # z指数表明股权相对制衡阈值
        'balance_index': 1          # 股权制衡度阈值
    }
    top1_concentration = 1 if data.loc[0, 'ratio'] > threshold['concentration_top1'] else 0
    top1_separation = 1 if data.loc[0, 'ratio'] < threshold['separation_top1'] else 0
    all10_concentration = 1 if data.loc[:, 'ratio'].sum() > threshold['concentration_all10'] else 0
    all10_separation = 1 if data.loc[:, 'ratio'].sum() < threshold['separation_all10'] else 0

    z_index = data.loc[0, 'ratio'] / data.loc[1, 'ratio']  # z指数
    z_index_concentration = 1 if z_index > threshold['z_index_concentration'] else 0    # 股权集中-Z指数
    z_index_separation = 1 if z_index < threshold['z_index_separation'] else 0          # 股权分散-Z指数

    balance_index = data.loc[0, 'ratio'] / data.loc[1:, 'ratio'].sum()  # 股权制衡度
    equity_balance = 1 if balance_index > threshold['balance_index'] else 0  # 股权制衡-股权制衡度

    return data, {
        'code': data.loc[0, 'code'],
        'ann_date': data.loc[0, 'ann_date'],
        'end_date': data.loc[0, 'end_date'],
        '国资': state_own,
        '外资': foreign_capital,
        '内资': domestic_capital,
        '股权集中_第一大股东': top1_concentration,
        '股权分散_第一大股东': top1_separation,
        '股权集中_前十大股东': all10_concentration,
        '股权分散_前十大股东': all10_separation,
        'Z指数': z_index,
        '股权集中_Z指数': z_index_concentration,
        '股权分散_Z指数': z_index_separation,
        '股权制衡度': balance_index,
        '股权制衡': equity_balance
    }

def build_label(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对给定日期的所有股票打标签"""
    data_with_label = []
    label = []
    if len(data) < 10:
        return pd.DataFrame(), pd.DataFrame()
    for code in data['code'].unique():
        single_data = data.loc[data['code'] == code, :]
        single_data = single_data.sort_values('holder_pct', ascending=False, ignore_index=True)
        if len(single_data) < 10:
            continue
        try:
            single_data_with_label, label_dict = build_label_for_single_stock(single_data)
            data_with_label.append(single_data_with_label)
            label.append(label_dict)
        except KeyError:
            print(single_data)
    try:
        return pd.concat(data_with_label), pd.concat(label)
    except KeyError as e:
        print(f'出现错误: {e}, 对应的数据为: \n{data}')
        return pd.DataFrame(), pd.DataFrame()

def check_list_contain(
        data: pd.DataFrame,
        white_list: list[str],
        new_column: str,
        check_column: str = 'name'
) -> pd.DataFrame:
    """检查股东名称中是否包含列表字段"""
    pattern = '|'.join(white_list)
    data[new_column] = data[check_column].str.contains(pattern)
    return data

def check_state_own(data: pd.DataFrame) -> pd.DataFrame:
    """检查是否为国资持股"""
    state_own_list = [
        "国有资产监督委员会",
        "中央汇金资产管理有限责任公司", "中央汇金投资有限责任公司",
        "全国社保基金",
        "中国证券金融股份有限公司",
        "财政局", "财政部", "人民政府",
        "城投控股", "国有资本经营管理有限公司", "市投资", "省投资"
    ]
    return check_list_contain(data, state_own_list, 'state_own')

def check_foreign_capital(data: pd.DataFrame) -> pd.DataFrame:
    """检查是否为外资持股"""
    foreign_capital_list = [
        "香港中央結算有限公司", "香港中央结算有限公司",
        "瑞士联合银行集团", "瑞士嘉盛银行有限公司",
        "汇丰", "花旗", "渣打", "法国巴黎银行",
        "高盛国际", "Goldman Sachs", "GOLDMAN SACHS",
        "贝莱德", "BLACKROCK", "Blackrock",
        "摩根大通证券有限公司", "摩根士丹利国际股份有限公司", "MORGAN", "Morgan",
        "美林证券", "MERRILL LYNCH", "Merrill Lynch",
        "巴克莱", "BARCLAYS", "Barclays",
        "国际金融公司",
        "阿布达比投资局", "科威特政府投资局", "淡马锡富敦投资有限公司",
        "LIMITED", "Limited"
    ]
    return check_list_contain(data, foreign_capital_list, 'foreign_capital')

def check_domestic_capital(data: pd.DataFrame) -> pd.DataFrame:
    """检查是否为内资持股"""
    filter_list = [
        "交易性开放式指数证券投资基金", "沪深300", "中证500", "中证1000"
    ]
    domestic_capital_list = [
        "保险产品", "保险责任有限公司",
        "股票型证券投资基金", "混合型证券投资基金", "股票型发起式证券投资基金", "混合型发起式证券投资基金",
        "证券股份有限公司", "证券有限公司", "资产管理计划"
    ]
    data = check_list_contain(data, filter_list, 'ETF')
    data = check_list_contain(data, domestic_capital_list, 'domestic_capital')
    data['domestic_capital'] = np.where(
        data['ETF'] == 1,
        0,
        data['domestic_capital']
    )
    return data


class AsyncDataProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=16)

    async def async_data_fetch(
            self,
            params: tuple[str, int, Literal['q1', 'h1', 'q2', 'y1']]
    ) -> pd.DataFrame:
        """将数据获取包装为异步版本"""
        loop = asyncio.get_event_loop()

        data = await loop.run_in_executor(
            self.executor,
            get_data,
            params[0], params[1], params[2]
        )
        return data

    async def async_build_label(
            self,
            params: tuple[str, int, Literal['q1', 'h1', 'q2', 'y1']],
            pbar: tqdm
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """将数据处理包装为异步版本"""
        data = await self.async_data_fetch(params)
        label = build_label(data)

        pbar.update(1)
        date_str = f"{params[0]} {params[1]} {params[2]}"
        pbar.set_description(f"处理: {date_str}")

        return label

    async def process_batch(
            self,
            param_list: List[tuple[str, int, Literal['q1', 'h1', 'q2', 'y1']]]
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """并发处理数据请求"""
        with tqdm(total=len(param_list)) as pbar:
            tasks = [
                self.async_build_label(param, pbar) for param in param_list
            ]
            results = await asyncio.gather(*tasks)
        return results


def generate_param_list(
        start_date: datetime.date, end_date: datetime.date
) -> List[tuple[str, int, Literal['q1', 'h1', 'q2', 'y1']]]:
    """生成参数列表"""
    date_to_period: dict[tuple[int, int], Literal['q1', 'h1', 'q2', 'y1']] = {
        (3, 31): 'q1',
        (6, 30): 'h1',
        (9, 30): 'q2',
        (12, 31): 'y1'
    }
    date_index = pd.date_range(start_date, end_date)
    param_list: List[tuple[str, int, Literal['q1', 'h1', 'q2', 'y1']]] = []
    for date in date_index:
        if (
            date.month == 3 and date.day == 31
        ) or (
            date.month == 6 and date.day == 30
        ) or (
            date.month == 9 and date.day == 30
        ) or (
            date.month == 12 and date.day == 31
        ):
            _err, code_df = w.wset(
                "sectorconstituent",
                f"date={date.year}-{date.month:02d}-{date.day:02d}; windcode=000985.CSI",
                usedf=True
            )
            for code in code_df['wind_code']:
                param = (code, date.year, date_to_period[(date.month, date.day)])
                param_list.append(param)
    return param_list

async def main(cache_path: str, start_date: datetime.date):
    """异步主函数"""
    try:
        labeled_df_cache = pd.read_parquet(os.path.join(cache_path, 'labeled_shareholder.parquet'))
        label_df_cache = pd.read_parquet(os.path.join(cache_path, 'shareholder_label.parquet'))
        labeled_df_cache['ann_date'] = pd.to_datetime(labeled_df_cache['ann_date'], format='%Y%m%d')
        labeled_df_cache['end_date'] = pd.to_datetime(labeled_df_cache['end_date'], format='%Y%m%d')
        label_df_cache['ann_date'] = pd.to_datetime(label_df_cache['ann_date'], format='%Y%m%d')
        label_df_cache['end_date'] = pd.to_datetime(label_df_cache['end_date'], format='%Y%m%d')

        start_date = label_df_cache['end_date'].max()
    except FileNotFoundError:
        labeled_df_cache = pd.DataFrame()
        label_df_cache = pd.DataFrame()
        start_date = start_date
    end_date = datetime.date.today()

    processor = AsyncDataProcessor()

    param_list = generate_param_list(start_date, end_date)
    results = await processor.process_batch(param_list=param_list)

    labeled_df = [labeled_df_cache] + [tp[0] for tp in results]
    label_df = [label_df_cache] + [tp[1] for tp in results]

    labeled_df = (
        pd.concat(labeled_df)
            .drop_duplicates(subset=['code', 'ann_date', 'holder_name'])
            .sort_values(by=['code', 'ann_date'], ignore_index=True)
    )
    labeled_df.to_parquet(os.path.join(cache_path, 'labeled_shareholder.parquet'))
    label_df = (
        pd.concat(label_df)
            .drop_duplicates(subset=['code', 'ann_date'])
            .sort_values(by=['code', 'ann_date'], ignore_index=True)
    )
    label_df.to_parquet(os.path.join(cache_path, 'shareholder_label.parquet'))

    return 0

if __name__ == '__main__':
    w.start()
    asyncio.run(
        main(
            cache_path=r'D:\QuantData\ShareholderData',
            start_date=datetime.date(2010, 1, 1)
        )
    )
