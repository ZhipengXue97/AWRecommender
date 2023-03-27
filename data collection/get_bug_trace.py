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
fixed_pair_log_file_path = f'{local_project_path}/log/fixed_pair_all.txt'
fixed_json_log_file_path = f'{local_project_path}/log/fixed_json.txt'
bug_trace_log_file_path = f'{local_project_path}/log/bug_trace.txt'

infer_path = config['PATH']['infer_path']

already_repo = []
commit_repo = readjson(bug_trace_log_file_path)
for commit in commit_repo:
    repo = commit.split("/")[0]
    already_repo.append(repo)

fixed_json_all = readjson(fixed_json_log_file_path)

repo_url_dict = {}
with open(repo_url_path)as f:
    cont = f.readlines()
    for line in cont:
        line = line.split("	")
        repo_url_dict[line[0]] = (line[2], line[3])

fixed_repo = readjson(fixed_pair_log_file_path)

bug_trace_code = []
hash_commit = {}
for repo_id in fixed_repo:
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
    fix_pairs = fixed_repo[repo_id]

    gr = pydriller.Git(trg_repo)

    for fix_pair in fix_pairs:
        bug_trace_code.clear()
        cur = fix_pair[0]
        pre = fix_pair[1]

        gr.checkout(pre)
        commit_title = f'{repo_id}/{cur}'
        if commit_title not in fixed_json_all:
            continue
        fixed_json = fixed_json_all[commit_title]
        bug_trace = fixed_json[0]['bug_trace']
        for trace in bug_trace:
            filename = trace['filename']
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
        logjson(commit_title, bug_trace_code, bug_trace_log_file_path)

    shutil.rmtree(trg_repo)
    if os.path.exists('{infer_out_dir}/tmp'):
        shutil.rmtree(f'{infer_out_dir}/tmp')
