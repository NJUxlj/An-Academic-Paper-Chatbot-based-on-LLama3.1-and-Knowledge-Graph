'''
基于脚本的多轮对话系统
'''

import json
import random, re, os
import pandas as pd


class DialogSystem:
    def __init__(self):
        self.load()
        
        
    def load(self):
        # 加载场景
        self.node_id_to_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        
        
        










if __name__ == '__main__':
    pass
