# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   log.py
@Time    :   2021/09/02 23:17:22
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''


"""
功能：
1. 分 info 和 error 两个logger
2. 可以按日期、大小、分割时间点分割日志
3. 报错可追溯
4. 日志格式：修复显示文件名称、行号、函数名等问题
"""
import os
import sys
import time
import datetime
import copy

from loguru import logger


class CustRotator:
    """
    自定义 Rotator 类
    """
    def __init__(self, *, size, at):
        now = datetime.datetime.now()
        self._size_limit = size
        self._time_limit = now.replace(
            hour=at.hour, 
            minute=at.minute, 
            second=at.second
        )

        if now >= self._time_limit:
            # The current time is already past the target time so it would rotate already.
            # Add one day to prevent an immediate rotation.
            self._time_limit += datetime.timedelta(days=1)

    def should_rotate(self, message, file):
        file.seek(0, 2)
        if file.tell() + len(message) > self._size_limit:
            return True
        if message.record["time"].timestamp() > self._time_limit.timestamp():
            self._time_limit += datetime.timedelta(days=1)
            return True
        return False


class CustFilter:
    """
    级别名称	严重度值	记录器法
    TRACE	    5	    logger.trace()
    DEBUG	    10	    logger.debug()
    INFO	    20	    logger.info()
    SUCCESS	    25	    logger.success()
    WARNING	    30	    logger.warning()
    ERROR	    40	    logger.error()
    CRITICAL	50	    logger.critical()

    1. info 单独存文件
    2. warning 以上合并存文件
    """

    def __init__(self, level):
        self.level = level

    def __call__(self, record):
        levelno = logger.level(self.level).no
        # info
        if 20 <= levelno < 30:
            return record["level"].no in [20, 25]
        else:
            return record["level"].no >= levelno


def cust_caller(is_filename=True, is_lineno=False, is_func_name=False):
    """
    查看函数调用的情况
    :param func:
    :return:
    """

    def decorate(func):
        def wrapper(*args, **kwargs):
            f = sys._getframe()
            filename = f.f_back.f_code.co_filename
            if '/' in filename:
                fname = filename.split('/')[-1]
            elif "\\" in filename:
                fname = filename.split("\\")[-1]
            else:
                fname = filename
            lineno = f.f_back.f_lineno
            fun_name = f.f_back.f_code.co_name
            b_lst = [is_filename, is_lineno, is_func_name]
            v_lst = [fname, lineno, fun_name]
            bv_lst = [str(v) for b, v in zip(b_lst, v_lst) if b]
            if any(b_lst):
                fs = ' - '.join(bv_lst)
                args = (args[0], f"{fs} | {args[1]}")
            func(*args, **kwargs)

        return wrapper

    return decorate


class Logger:
    """
    日志文件配置
    """
    def __init__(
            self,
            file_path=None,
            level='INFO',
            format=None,
            colorize=False
    ):
        # 刷新格式
        logger.remove()
        # 示例化
        self.logger = copy.deepcopy(logger)
        # 按照 大小、 日期来分割日志
        rotator = CustRotator(size=5e+8, at=datetime.time(0, 0, 0))
        # info 单独存文件; warning 以上合并存文件
        level_filter = CustFilter(level)
        # format
        if not format:
            # 时间日期格式：https://loguru.readthedocs.io/en/stable/api/logger.html#time
            format = '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}'

        # 数据流显示
        self.logger.add(
            sys.stdout,
            format=format,
            filter=level_filter,
            colorize=colorize,
            level=level,
        )
        # 文件存储
        if file_path:
            self.logger.add(
                file_path,
                encoding='utf-8',
                rotation=rotator.should_rotate,
                enqueue=True,
                filter=level_filter,
                level=level,
                format=format
            )

    def __new__(cls, *args, **kwargs):
        """ 单例模式 """
        return super().__new__(cls)

    @cust_caller(is_filename=True, is_lineno=False, is_func_name=False)
    def info(self, msg):
        return self.logger.info(msg)

    @cust_caller(is_filename=True, is_lineno=True, is_func_name=True)
    def warn(self, msg):
        return self.logger.warning(msg)

    @cust_caller(is_filename=True, is_lineno=True, is_func_name=True)
    def error(self, msg):
        return self.logger.error(msg)

    def debug(self, msg):
        return self.logger.debug(msg)

if __name__ == '__main__':
    #
    t = time.strftime('%Y_%m_%d')
    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #
    log_info = Logger(f"logs/test_{t}.log", level='INFO')
    # log_error = Logger(f"logs/poi_error_{t}.log", level='WARNING')
    # log_1 = Logger()
    log_info.info('test !!!')
    log_info.info('hello !!!')
    log_info.warn('warn !!!')
    # log_error.warn('warn !!!')
    print('main over')

