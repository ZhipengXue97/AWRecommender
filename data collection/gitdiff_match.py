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

infer_path = config['PATH']['infer_path']

with open(gitdiff_log_file_path)as f:
    gitdiff_all = f.readlines()
fixed_json_all = readjson(fixed_json_log_file_path)


def get_match_score(fix, gitdiff):
    score = 0
    key_path = ""
    bug_type = fix['bug_type']
    qualifier = fix['qualifier']
    path = fix['file']
    key_path_list = path.split("/")[2::]
    for word in key_path_list:
        key_path += word
        key_path += "/"
    key_path = key_path.strip("/")
    pattern = r'[1-9]\d*'
    key_line_num = re.findall(pattern, qualifier)

    if bug_type == "DEAD_STORE":
        DEAD_STORE_pattern = '\&([a-zA-z]+)'
        variable_name = re.findall(DEAD_STORE_pattern, qualifier)[0]
        key_line_num = fix['line']
        if key_path in gitdiff:
            added_lines = gitdiff[key_path]["added"]
            for line in added_lines:
                line_num = line[0]
                line_content = line[1]
                pattern = f'[^a-zA-Z0-9]({variable_name})[^a-zA-Z0-9]'
                if line_num >= key_line_num:
                    res = re.findall(pattern, line_content)
                    score = max(score, 1)
                    if len(res)>0:
                        if "=" in line_content:
                            left_line_content = line_content.split("=")[0]
                            res = re.findall(pattern, line_content)
                            if len(res)==0:
                                score = max(score, 3)
                                continue
                        else:
                            score = max(score, 3)
                            continue



    if bug_type == "NULL_DEREFERENCE":
        NULL_DEREFERENCE_pattern = '`(.*?)`'
        variable_name = re.findall(NULL_DEREFERENCE_pattern, qualifier)[0]
        if key_path in gitdiff:
            start_line = int(key_line_num[0])
            end_line = int(key_line_num[1])
            added_lines = gitdiff[key_path]["added"]
            for line in added_lines:
                line_num = line[0]
                line_content = line[1]
                if line_num >= start_line and line_num <= end_line:
                    score = max(score, 1)
                    if "NULL" in line_content:
                        score = max(score, 2)
                        continue
                    if variable_name in line_content and "=" in line_content:
                        left_line_content = line_content.split("=")[0]
                        pattern = f'[^a-zA-Z0-9]({variable_name})[^a-zA-Z0-9]'
                        res = re.findall(pattern, left_line_content)
                        if len(res)>0:
                            score = max(score, 2)
                            continue


    if bug_type == "UNINITIALIZED_VALUE":
        UNINITIALIZED_VALUE_pattern = 'from (.*?)[\[| ]'
        variable_name = re.findall(UNINITIALIZED_VALUE_pattern, qualifier)[0]
        if key_path in gitdiff:
            added_lines = gitdiff[key_path]["added"]
            for line in added_lines:
                line_num = line[0]
                line_content = line[1]
                if variable_name in line_content:
                    score = max(score, 1)
                    if "=" in line_content:
                        left_line_content = line_content.split("=")[0]
                        pattern = f'[^a-zA-Z0-9]({variable_name})[^a-zA-Z0-9]'
                        res = re.findall(pattern, left_line_content)
                        if len(res)>0:
                            score = max(score, 2)
                            continue

    if bug_type == "RESOURCE_LEAK":
        keyword_list = ["source", "close", "free","clean","destroy","clear","remove","release"]
        if key_path in gitdiff:
            added_lines = gitdiff[key_path]["added"]
            end_line = int(key_line_num[-2])
            for line in added_lines:
                line_num = line[0]
                line_content = line[1]
                if line_num >= end_line:
                    score = max(score, 1)
                    for word in keyword_list:
                        if word in line_content:
                            score = max(score, 2)



    return score

l = len(gitdiff_all)
for i in range(l):
    gitdiff_json = json.loads(gitdiff_all[i])
    for commit in gitdiff_json:
        gitdiff = gitdiff_json[commit]
        fixed_json = fixed_json_all[commit]

        for fix in fixed_json:
            bug_type = fix['bug_type']
            if bug_type not in ['DEAD_STORE','NULL_DEREFERENCE',"UNINITIALIZED_VALUE","RESOURCE_LEAK"]:
                continue
            score = get_match_score(fix, gitdiff)
            logjson(commit, str(score), gitdiff_score_path_file)
