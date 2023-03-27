import os
import re
import json
import configparser
from util import readjson, check_commit, get_full_msg, logjson
import sys
sys.setrecursionlimit(20000)


config = configparser.ConfigParser()
config.read("warning_elimination/config.ini")

data_root_path = config['PATH']['data_root_path']
repository_dataset_path = f'{data_root_path}/c_repos'
tmp_repository_path = f'{data_root_path}/tmp_repository'
infer_out_dir = f'{data_root_path}/infer_out_all'

local_project_path = config['PATH']['local_project_path']
repo_url_path = f'{local_project_path}/log/c_repos_sorted'
fixed_pair_log_file_path = f'{local_project_path}/log/fixed_pair_all.txt'
gitdiff_log_file_path = f'{local_project_path}/log/fixed_gitdiff.txt'
fixed_json_log_file_path = f'{local_project_path}/log/fixed_json.txt'
gitdiff_score_path_file = f'{local_project_path}/log/gitdiff_score.txt'
false_positive_path = f'{local_project_path}/log/false_positive.txt'

infer_path = config['PATH']['infer_path']

fixed_json_all = readjson(fixed_json_log_file_path)


all_dict = {}

infer_out_all_dir = os.listdir(infer_out_dir)
for dir in infer_out_all_dir:
    reports = os.listdir(f'{infer_out_dir}/{dir}')
    for report in reports:
        if "json"in report.split(".")[-1]:
            report = f'{infer_out_dir}/{dir}/{report}'
            with open(report) as f:
                content = f.read().replace("\n","")
                if content != "[]":
                    report = json.loads(content)
                    for i in report:
                        dir_hash = f'{dir}/{i["hash"]}'
                        all_dict[dir_hash] = i

for fixed_json in fixed_json_all:
    dir = fixed_json.split("/")[0]
    fixed_json = fixed_json_all[fixed_json][0]
    dir_hash = f'{dir}/{fixed_json["hash"]}'
    if hash in all_dict:
        all_dict.pop(hash)

for item in all_dict:
    logjson(item, all_dict[item], false_positive_path)
