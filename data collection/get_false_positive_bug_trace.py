import os
import shutil
import pydriller
import json
import networkx as nx
from util import check_compilation, readjson, classify_repository, logjson, check_commit
import configparser

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
false_positive_path = f'{local_project_path}/log/false_positive.txt'
bug_trace_log_file_path = f'{local_project_path}/log/false_positive_bug_trace.txt'

infer_path = config['PATH']['infer_path']

def build_warning_commit_dict(infer_out_path):
    warning_commit_dict = {}
    reports = os.listdir(infer_out_path)
    for commit in reports:
         if "json"in commit.split(".")[-1]:
            report = f'{infer_out_path}/{commit}'
            with open(report) as f:
                content = f.read().replace("\n","")
                if content != "[]":
                    report = json.loads(content)
                    for i in report:
                        warning_commit_dict[i['hash']] = commit.replace("_report.json","")

    return warning_commit_dict



already_repo = []
commit_repo = readjson(bug_trace_log_file_path)
for commit in commit_repo:
    repo = commit.split("/")[0]
    already_repo.append(repo)


repo_url_dict = {}
with open(repo_url_path)as f:
    cont = f.readlines()
    for line in cont:
        line = line.split("	")
        repo_url_dict[line[0]] = (line[2], line[3])

false_positive_all = readjson(false_positive_path)

bug_trace_code = []

repo_warning_dict = {}
for false_positive in false_positive_all:
    repo_id  = false_positive.split("/")[0]
    warning_hash = false_positive.split("/")[1]
    if repo_id not in repo_warning_dict:
        repo_warning_dict[repo_id] = [warning_hash]
    else:
        repo_warning_dict[repo_id].append(warning_hash)

for repo_id in repo_warning_dict:
    if repo_id not in repo_url_dict:
        continue
    repo_name = repo_url_dict[repo_id][1].replace("\n","")
    repo_url = repo_url_dict[repo_id][0]
    print(repo_id, repo_name)
    if repo_id in already_repo:
        continue
    infer_out_dirpath = f'{infer_out_dir}/{repo_id}'
    trg_repo = f'{tmp_repository_path}/{repo_name}'
    if not os.path.exists(trg_repo):
        cd_cmd = f'cd {tmp_repository_path}'
        git_cmd = f'git clone {repo_url}'
        os.system(f'{cd_cmd} && {git_cmd}')

    gr = pydriller.Git(trg_repo)
    warning_commit_dict = build_warning_commit_dict(infer_out_dirpath)

    for warning_hash in repo_warning_dict[repo_id]:
        try:
            commit = warning_commit_dict[warning_hash]
            gr.checkout(commit)
            bug_trace_code.clear()
            commit_title = f'{repo_id}/{warning_hash}'
            if commit_title not in false_positive_all:
                continue
            false_positive = false_positive_all[commit_title]
            bug_trace = false_positive['bug_trace']
            for trace in bug_trace:
                filename = trace['filename']
                filename = filename.replace("dataserver145/genomics/Miaomiao/py_repo/", "")
                filename = filename.replace("/data1/zhipengxue/", "")
                if filename.split("/")[0] != "repository":
                    continue
                filename = f'tmp_{filename}'
                line_num = trace['line_number']
                with open(filename)as f:
                    contents = f.readlines()
                code  = contents[line_num-1]
                code = code.replace("\n", "")
                code = code.replace("\t", "")
                bug_trace_code.append(code)
        except:
            continue
        logjson(commit_title, bug_trace_code, bug_trace_log_file_path)

    shutil.rmtree(trg_repo)
    if os.path.exists('{infer_out_dir}/tmp'):
        shutil.rmtree(f'{infer_out_dir}/tmp')
