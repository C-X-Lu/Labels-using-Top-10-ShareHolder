import os
import akshare as ak
import pandas as pd
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
from tqdm import tqdm
from typing import List


def code_plus_exchange(code: str) -> str|None:
    """为股票代码加上交易所代码以适用于akshare的接口"""
    if code[:2] == '00' or code[:2] == '30':
        return f"sz{code}"
    elif code[:2] == '60' or code[:2] == '68':
        return f"sh{code}"
    elif code[:2] == '92':
        return f"bj{code}"
    else:
        return None

def get_data(code: str, end_date: str) -> pd.DataFrame:
    """使用akshare的stock_gdfx_top_10_em接口获取数据"""
    code_with_exchange = code_plus_exchange(code)
    if code_with_exchange is None:
        return pd.DataFrame(
            columns=['code', 'end_date', 'holder_name', 'hold_num', 'holder_pct', 'change', 'change_ratio']
        )
    try:
        shareholder_df = ak.stock_gdfx_top_10_em(symbol=code_with_exchange, date=end_date)
    except ValueError:
        return pd.DataFrame(
            columns=['code', 'end_date', 'holder_name', 'hold_num', 'holder_pct', 'change', 'change_ratio']
        )
    if len(shareholder_df) != 0:
        shareholder_df.rename(
            columns={
                '股东名称': 'holder_name',
                '持股数': 'hold_num',
                '占总股本持股比例': 'holder_pct',
                '增减': 'change',
                '变动比率': 'change_ratio'
            }, inplace=True
        )
        shareholder_df['code'] = code
        shareholder_df['end_date'] = end_date
        shareholder_df['end_date'] = pd.to_datetime(shareholder_df['end_date'], format='%Y%m%d')
        return shareholder_df
    else:
        return pd.DataFrame(
            columns=['code', 'end_date', 'holder_name', 'hold_num', 'holder_pct', 'change', 'change_ratio']
        )

def build_label(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对股东数据打标签
    :param data:
    :return:
    """
    if len(data) == 0:
        return data, pd.DataFrame(columns=[
            'code', 'ann_date', 'end_date', '国资', '外资', '内资',
            '股权集中_第一大股东', '股权分散_第一大股东', '股权集中_前十大股东', '股权分散_前十大股东',
            'Z指数', '股权集中_Z指数', '股权分散_Z指数',
            '股权制衡度', '股权制衡'
        ])
    # 股东性质
    data = check_state_own(data)
    state_own = int(data['state_own'].astype(bool).any())

    data = check_foreign_capital(data)
    foreign_capital = int(data['foreign_capital'].astype(bool).any())

    data = check_domestic_capital(data)
    domestic_capital = int(data['domestic_capital'].astype(bool).any())

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
    top1_concentration = 1 if data.loc[0, 'holder_pct'] > threshold['concentration_top1'] else 0
    top1_separation = 1 if data.loc[0, 'holder_pct'] < threshold['separation_top1'] else 0
    all10_concentration = 1 if data.loc[:, 'holder_pct'].sum() > threshold['concentration_all10'] else 0
    all10_separation = 1 if data.loc[:, 'holder_pct'].sum() < threshold['separation_all10'] else 0

    z_index = data.loc[0, 'holder_pct'] / data.loc[1, 'holder_pct']  # z指数
    z_index_concentration = 1 if z_index > threshold['z_index_concentration'] else 0    # 股权集中-Z指数
    z_index_separation = 1 if z_index < threshold['z_index_separation'] else 0          # 股权分散-Z指数

    balance_index = data.loc[0, 'holder_pct'] / data.loc[1:, 'holder_pct'].sum()  # 股权制衡度
    equity_balance = 1 if balance_index > threshold['balance_index'] else 0  # 股权制衡-股权制衡度

    return data, pd.DataFrame({
        'code': data.loc[0, 'code'],
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
    }, index=[0])

def check_list_contain(
        data: pd.DataFrame,
        white_list: list[str],
        new_column: str,
        check_column: str = 'holder_name'
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

def fetch_code_list(
        date: datetime.date
) -> pd.DataFrame:
    """使用akshare的stock_yjbb_em接口获取对应截止日期的股票列表"""
    end_date_str: str = f"{date.year}{date.month:02d}{date.day:02d}"
    code_df = ak.stock_yjbb_em(date=end_date_str)
    return code_df

def generate_param_list_and_announce_date(
        data: pd.DataFrame, end_date: datetime.date
) -> tuple[List[tuple[str, str]], pd.DataFrame]:
    """生成参数列表和公告披露日期DataFrame"""
    data.rename(columns={
        '股票代码': 'code',
        '最新公告日期': 'ann_date',
    }, inplace=True)
    data['ann_date'] = pd.to_datetime(data['ann_date'], format='%Y-%m-%d')
    data['end_data'] = end_date
    code_list = data['code'].tolist()
    end_date_str = f"{end_date.year}{end_date.month:02d}{end_date.day:02d}"
    param_list = [(code, end_date_str) for code in code_list]
    return param_list, data

def generate_date_list(
        start_date: datetime.date, end_date: datetime.date
) -> List[datetime.date]:
    """生成日期列表"""
    date_index = pd.date_range(start_date, end_date)
    date_list = []
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
            date_list.append(date)
    return date_list

class AsyncDataProcessorForParamGenerating:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=64)

    async def async_data_fetch(
            self,
            date: datetime.date
    ) -> pd.DataFrame:
        """将参数和公告日期数据获取包装为异步版本"""
        loop = asyncio.get_event_loop()

        raw_code_list_data = await loop.run_in_executor(
            self.executor,
            fetch_code_list,
            date
        )
        return raw_code_list_data

    async def async_basic_process(
            self,
            end_date: datetime.date,
            pbar: tqdm
    ) -> tuple[List[tuple[str, str]], pd.DataFrame]:
        """将数据处理包装为异步版本"""
        data = await self.async_data_fetch(end_date)
        param_list, announce_df = generate_param_list_and_announce_date(data, end_date)

        pbar.update(1)
        date_str = f"{end_date.year}{end_date.month:02d}{end_date.day:02d}"
        pbar.set_description(f"处理: {date_str}")

        return param_list, announce_df

    async def process_batch(
            self,
            date_list: List[datetime.date]
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """并发处理数据请求"""
        with tqdm(total=len(date_list)) as pbar:
            tasks = [
                self.async_basic_process(date, pbar) for date in date_list
            ]
            results = await asyncio.gather(*tasks)
        return results


async def main_param_and_announce_date(
        cache_path: str, start_date: datetime.date, end_date: datetime.date
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """异步主函数1"""
    try:
        label_df_cache = pd.read_parquet(os.path.join(cache_path, 'shareholder_label.parquet'))
        label_df_cache['end_date'] = pd.to_datetime(label_df_cache['end_date'], format='%Y%m%d')
        start_date = max(start_date, label_df_cache['end_date'].max())
    except FileNotFoundError:
        pass

    processor = AsyncDataProcessorForParamGenerating()
    results = await processor.process_batch(generate_date_list(start_date, end_date))

    param_list_list = [t[0] for t in results]
    param_list = []
    for p_l in param_list_list:
        param_list += p_l

    announce_df_list = [t[1] for t in results]
    announce_df = pd.concat(announce_df_list, ignore_index=True)

    return param_list, announce_df


class AsyncDataProcessorForFetchAndLabelShareholderInfo:
    """获取和标签化十大股东数据的异步类"""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=128)

    async def async_data_fetch(
            self,
            params: tuple[str, str]
    ) -> pd.DataFrame:
        """将数据获取包装为异步版本"""
        loop = asyncio.get_event_loop()

        data = await loop.run_in_executor(
            self.executor,
            get_data,
            params[0], params[1]
        )
        return data

    async def async_build_label(
            self,
            params: tuple[str, str],
            pbar: tqdm
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """将数据处理包装为异步版本"""
        data = await self.async_data_fetch(params)
        label = build_label(data)

        pbar.update(1)
        date_str = f"{params[0]} {params[1]}"
        pbar.set_description(f"处理: {date_str}")

        return label

    async def process_batch(
            self,
            param_list: List[tuple[str, str]]
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """并发处理数据请求"""
        with tqdm(total=len(param_list)) as pbar:
            tasks = [
                self.async_build_label(param, pbar) for param in param_list
            ]
            results = await asyncio.gather(*tasks)
        return results


async def main_shareholder(
        cache_path: str,
        param_list: List[tuple[str, str]],
        announce_df: pd.DataFrame
):
    """异步主函数2"""
    try:
        labeled_df_cache = pd.read_parquet(os.path.join(cache_path, 'labeled_shareholder.parquet'))
        label_df_cache = pd.read_parquet(os.path.join(cache_path, 'shareholder_label.parquet'))
        labeled_df_cache['ann_date'] = pd.to_datetime(labeled_df_cache['ann_date'], format='%Y%m%d')
        labeled_df_cache['end_date'] = pd.to_datetime(labeled_df_cache['end_date'], format='%Y%m%d')
        label_df_cache['ann_date'] = pd.to_datetime(label_df_cache['ann_date'], format='%Y%m%d')
        label_df_cache['end_date'] = pd.to_datetime(label_df_cache['end_date'], format='%Y%m%d')

    except FileNotFoundError:
        labeled_df_cache = pd.DataFrame()
        label_df_cache = pd.DataFrame()

    processor = AsyncDataProcessorForFetchAndLabelShareholderInfo()

    results = await processor.process_batch(param_list=param_list)

    labeled_df = [labeled_df_cache] + [tp[0] for tp in results]
    label_df = [label_df_cache] + [tp[1] for tp in results]

    labeled_df = (
        pd.concat(labeled_df)
            .drop_duplicates(subset=['code', 'ann_date', 'holder_name'])
            .sort_values(by=['code', 'ann_date'], ignore_index=True)
    )
    labeled_df = pd.merge(labeled_df, announce_df, on=['code', 'end_date'], how='inner')
    labeled_df.to_parquet(os.path.join(cache_path, 'labeled_shareholder.parquet'))
    label_df = (
        pd.concat(label_df)
            .drop_duplicates(subset=['code', 'ann_date'])
            .sort_values(by=['code', 'ann_date'], ignore_index=True)
    )
    label_df = pd.merge(label_df, announce_df, on=['code', 'end_date'], how='inner')
    label_df.to_parquet(os.path.join(cache_path, 'shareholder_label.parquet'))

    return 0


if __name__ == '__main__':
    tu = asyncio.run(
        main_param_and_announce_date(
            cache_path=r'D:\QuantData\ShareholderData',
            start_date=datetime.date(2015, 1, 1),
            end_date=datetime.date.today()
        )
    )
    asyncio.run(
        main_shareholder(
            cache_path=r'D:\QuantData\ShareholderData',
            param_list=tu[0],
            announce_df=tu[1]
        )
    )
