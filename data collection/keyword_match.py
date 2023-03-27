import os
import shutil
import pydriller
import re
import json
import networkx as nx
import configparser
from util import readjson, check_commit, get_full_msg, logjson


config = configparser.ConfigParser()
config.read("warning_elimination/config.ini")

data_root_path = config['PATH']['data_root_path']
repository_dataset_path = f'{data_root_path}/c_repos'
tmp_repository_path = f'{data_root_path}/tmp_repository'
infer_out_dir = f'{data_root_path}/infer_out_all'

local_project_path = config['PATH']['local_project_path']
repo_url_path = f'{local_project_path}/log/c_repos_sorted'
fixed_pair_log_file_path = f'{local_project_path}/log/fixed_pair_all.txt'
commit_message_log_file_path = f'{local_project_path}/log/commit_msg.txt'
fixed_json_log_file_path = f'{local_project_path}/log/fixed_json.txt'
keyword_score_file_path = f'{local_project_path}/log/keyword_score.txt'

infer_path = config['PATH']['infer_path']

bug_type_all = ['DEAD_STORE','NULL_DEREFERENCE',"UNINITIALIZED_VALUE","RESOURCE_LEAK"]
bug_type_match = {}
bug_keyword_match = {}
bug_common_match = {}
for bug_type in bug_type_all:
    bug_type_match[bug_type] = 0
    bug_keyword_match[bug_type] = 0
    bug_common_match[bug_type] = 0




def get_fix_keyword(fix_json):
    bug_type = fix_json['bug_type']
    bug_type_keyword = []
    bug_tmp_keyword = []
    bug_keyword = []
    if bug_type == "DEAD_STORE":
        bug_type_keyword.extend(['dead', 'store', 'never', "unused"])
        pattern = '\&([a-zA-z]+)'
        res = re.findall(pattern, fix_json['qualifier'])
        bug_tmp_keyword.extend(res)
    if bug_type == "NULL_DEREFERENCE":
        bug_type_keyword.extend(['null', 'dereferen'])
        pattern = '`(.*?)`'
        res = re.findall(pattern, fix_json['qualifier'])
        bug_tmp_keyword.extend(res)
    if bug_type == "UNINITIALIZED_VALUE":
        bug_type_keyword.extend(['initial'])
        pattern = 'from (.*?)[\[| ]'
        res = re.findall(pattern, fix_json['qualifier'])
        bug_tmp_keyword.extend(res)
    if bug_type == "RESOURCE_LEAK":
        bug_type_keyword.extend(['resource', 'leak', 'release', 'free'])
        pattern = '`(.*?)`'
        res = re.findall(pattern, fix_json['qualifier'])
        bug_tmp_keyword.extend(res)
    for keyword in bug_tmp_keyword:
        keyword = keyword.replace('()','')
        keyword = keyword.replace('*','')
        bug_keyword.append(keyword)
    return bug_type_keyword, bug_keyword

common_keyword = ['fix','solve',"warning","bug",'problem']

with open(commit_message_log_file_path) as f:
    commit_msg_all = f.readlines()

with open(fixed_json_log_file_path) as f:
    fixed_json_all = f.readlines()

l = len(commit_msg_all)
for i in range(l):
    commit_msg_dict = json.loads(commit_msg_all[i])
    commit_msg = list(commit_msg_dict.values())[0].lower()
    fixed_json = fixed_json_all[i]
    fixed_json = json.loads(fixed_json)
    for id in fixed_json:
        fix = fixed_json[id][0]
        flag = 0
        bug_type = fix['bug_type']
        if bug_type not in bug_type_all:
            continue
        bug_type_keyword, bug_keyword = get_fix_keyword(fix)
        for keyword in bug_type_keyword:
            if keyword in commit_msg:
                logjson(id,"3",keyword_score_file_path)
                bug_type_match[bug_type] += 1
                flag = 1
                break
        if flag == 0:
            for keyword in bug_keyword:
                pattern = f'[^a-zA-Z0-9]({keyword})[^a-zA-Z0-9]'
                res = re.findall(pattern,commit_msg)
                if len(res)>0:
                    logjson(id,"2",keyword_score_file_path)
                    bug_keyword_match[bug_type] += 1
                    flag = 1
                    break
        if flag == 0:
            for keyword in common_keyword:
                if keyword in commit_msg:
                    logjson(id,"1",keyword_score_file_path)
                    bug_common_match[bug_type] += 1
                    flag = 1
                    break
        if flag == 0:
            logjson(id,"0",keyword_score_file_path)

