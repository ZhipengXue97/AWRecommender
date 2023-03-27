import os
import shutil
import pydriller
import re
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
fixed_json_log_file_path = f'{local_project_path}/log/fixed_json.txt'

infer_path = config['PATH']['infer_path']

def check_fix(cur, pre, infer_out_dirpath):
    pre_json = f'{infer_out_dirpath}/{pre}_report.json'
    cur_json = f'{infer_out_dirpath}/{cur}_report.json'
    infer_diff_path = f'{infer_out_dirpath}/tmp'
    cd_cmd = f'cd {infer_path}'
    infer_cmd = f'./infer reportdiff -o {infer_diff_path} --report-current {cur_json} --report-previous {pre_json}'
    os.system(f'{cd_cmd} && {infer_cmd}')
    fixed_json_path = f'{infer_diff_path}/differential/fixed.json'
    introduced_json_path = f'{infer_diff_path}/differential/introduced.json'
    exit_code,  fixed_json= check_commit(fixed_json_path, introduced_json_path)
    return fixed_json

already_repo = []
commit_repo = readjson(fixed_json_log_file_path)
for commit in commit_repo:
    repo = commit.split("/")[0]
    already_repo.append(repo)


repo_url_dict = {}
with open(repo_url_path)as f:
    cont = f.readlines()
    for line in cont:
        line = line.split("	")
        repo_url_dict[line[0]] = (line[2], line[3])

fixed_repo = readjson(fixed_pair_log_file_path)

hash_commit = {}
for repo_id in fixed_repo:
    hash_commit.clear()
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
    for commit in pydriller.Repository(trg_repo).traverse_commits():
        hash_commit[commit.hash] = commit

    for fix_pair in fix_pairs:
        cur = fix_pair[0]
        pre = fix_pair[1]
        fixed_json = check_fix(cur, pre, infer_out_dirpath)
        if len(fixed_json)>1:
            print(1)
        commit_title = f'{repo_id}/{cur}'
        print(commit_title)

        logjson(commit_title , fixed_json, fixed_json_log_file_path)

    shutil.rmtree(trg_repo)
    if os.path.exists(f'{infer_out_dir}/tmp'):
        shutil.rmtree(f'{infer_out_dir}/tmp')




                


        


