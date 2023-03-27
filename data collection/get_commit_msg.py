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
commit_message_log_file_path = f'{local_project_path}/log/commit_msg.txt'

infer_path = config['PATH']['infer_path']

already_repo = []
commit_repo = readjson(commit_message_log_file_path)
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
        commit_url = repo_url.replace(".git","/commit")
        commit_url = f'{commit_url}/{cur}'
        commit_message = hash_commit[cur].msg
        commit_message = get_full_msg(repo_url, cur, commit_message)

        commit_title = f'{repo_id}/{cur}'
        print(commit_title)

        logjson(commit_title , commit_message, commit_message_log_file_path)

    shutil.rmtree(trg_repo)
    if os.path.exists('{infer_out_dir}/tmp'):
        shutil.rmtree(f'{infer_out_dir}/tmp')




                


        


