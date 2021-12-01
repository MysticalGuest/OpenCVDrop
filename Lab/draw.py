"""
@version: v1.0
@author: Mystery Guest
@license: Apache Licence
@contact: mysticalguest@163.com
@site: 
@software: PyCharm
@file: draw.py
@time: 2021/11/19 21:00
"""
import numpy as np
import matplotlib.pyplot as mp


def draw_histogram(x_data, y_data):
    # 设置中文字体
    mp.rcParams['font.sans-serif'] = ['SimHei']
    # mp.rcParams['axes.unicode_minus'] = False

    quantity = np.array(y_data)
    mp.figure('液滴统计', facecolor='lightgray')
    mp.title('液滴统计', fontsize=16)
    mp.xlabel('直径', fontsize=14)
    mp.ylabel('数量', fontsize=14)
    mp.tick_params(labelsize=10)
    mp.grid(linestyle=':', axis='y')
    x = np.arange(len(x_data))
    a = mp.bar(x - 0.2, quantity, 0.4, color='dodgerblue', label='液滴数量', align='center')
    # 设置标签
    for i in a:
        h = i.get_height()
        mp.text(i.get_x() + i.get_width() / 2, h, '%d' % int(h), ha='center', va='bottom')
    mp.xticks(x, x_data)

    mp.legend()
    mp.show()


if __name__ == '__main__':
    x1_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y1_data = [45, 46, 12, 45, 121, 65, 45, 60, 11, 56, 34, 54]
    draw_histogram(x1_data, y1_data)