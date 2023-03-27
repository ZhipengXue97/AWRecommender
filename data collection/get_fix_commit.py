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
tmp_repository_path = f'{data_root_path}/repository'
infer_out_dir = f'{data_root_path}/infer_out'

local_project_path = config['PATH']['local_project_path']
mode_compilation_log_file_path = f'{local_project_path}/log/mode_compilation.txt'
fixed_pair_log_file_path = f'{local_project_path}/log/fixed_pair.txt'

infer_path = config['PATH']['infer_path']

repo_dir = readjson(mode_compilation_log_file_path)


if os.path.exists(fixed_pair_log_file_path):
    repo_already_check = readjson(fixed_pair_log_file_path)
else:
    repo_already_check = []

def rm_tmp(path, RemindFile):
    dirsList = os.listdir(path)
    for f in dirsList:
        if f != RemindFile:
            filepath = os.path.join(path,f)
            if os.path.isdir(filepath):
                shutil.rmtree(filepath, True)
            else:
                os.remove(filepath)

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
    return exit_code


def run_infer(repo_path, commit, infer_out_dirpath):
    infer_out_path = f'{infer_out_dirpath}/{commit}'
    infer_out_json = f'{infer_out_dirpath}/{commit}_report.json'
    if os.path.exists(infer_out_json):
        return "Already_built"
    gr = pydriller.Git(repo_path)
    gr.checkout(commit)
    mode = classify_repository(repo_path)
    # mode = "custom"
    exit_code = check_compilation(repo_path,mode)
    if exit_code == 1:
        cd_cmd = f'cd {infer_path}'
        if mode == "cmake":
            infer_cmd = f'./infer run -o {infer_out_path} --compilation-database {repo_path}/build/compile_commands.json --keep-going'
        elif mode == "automake":
            infer_cmd = f'./infer run -o {infer_out_path} --compilation-database {repo_path}/compile_commands.json --keep-going'
        elif mode == "custom":
            infer_cmd = f'./infer run -o {infer_out_path} --compilation-database {repo_path}/compile_commands.json --keep-going'
        os.system(f'{cd_cmd} && {infer_cmd}')
        rm_tmp(infer_out_path, "report.json")
        shutil.move(f'{infer_out_path}/report.json', f'{infer_out_dirpath}/{commit}_report.json')
        shutil.rmtree(infer_out_path)
        if mode == "cmake":
            os.remove(f'{repo_path}/build/compile_commands.json')
        elif mode == "automake":
            os.remove(f'{repo_path}/compile_commands.json')
        elif mode == "custom":
            os.remove(f'{repo_path}/compile_commands.json')
        return "Already_built"
    else:
        return "Cannot_Build"

def get_pred(g, node):
    return list(g._pred[node])

def get_succ(g, node):
    return list(g._succ[node])


def get_linear_commit(g):
    linear_commits = []

    def get_commits(g, cur):
        if cur != None:
            commits = []
            succ = get_succ(g, cur)
            pred = get_pred(g, cur)

            if g._node[cur]['status'] == "Already_Check":
                if len(succ) < 2:
                    return

            # if cur is a fork commit and a merge commit
            if len(succ) > 1 and len(pred) > 1:
                commits.append(cur)
                g._node[cur]['status'] = "Already_Check"
                linear_commits.append(commits)
                for node in pred:
                    get_commits(g, node)
                return


            # if cur is a fork commit
            if len(succ) > 1:
                commits.append(cur)
                g._node[cur]['status'] = "Already_Check"
                if len(pred)>0:
                    cur = pred[0]
                else:
                    linear_commits.append(commits)
                    return

            # if cur is a merge commit
            if len(pred) > 1:
                g._node[cur]['status'] = "Already_Check"
                for node in pred:
                    get_commits(g, node)
                return

            # add commit if commit is not fork or merge commit
            while (len(succ) == 1 and len(pred) == 1) or (len(succ) == 0):
                commits.append(cur)
                g._node[cur]['status'] = "Already_Check"
                cur = pred[0]
                pred = get_pred(g, cur)
                succ = get_succ(g, cur)

            commits.append(cur)
            g._node[cur]['status'] = "Already_Check"
            linear_commits.append(commits)
            for node in g._pred[cur]:
                get_commits(g, node)

            

    end_node = [node for node in g._succ if len(g._succ[node]) == 0]
    for node in end_node:
        get_commits(g, node)
    for commits in linear_commits[::-1]:
        if len(commits) < 2:
            linear_commits.remove(commits)

    return linear_commits


def binary_search(cur, pre, commits, g, infer_out_dirpath):

    while cur < pre:
        cur_commit = commits[cur]
        pre_commit = commits[pre]

        if g._node[cur_commit]['status'] == "No_Check":
            exit_status = run_infer(trg_repo, cur_commit, infer_out_dirpath)
            g._node[cur_commit]['status'] = exit_status
            logjson(cur_commit, exit_status, f'{infer_out_dirpath}/commit_status.txt')
        if g._node[pre_commit]['status'] == "No_Check":
            exit_status = run_infer(trg_repo, pre_commit, infer_out_dirpath)
            g._node[pre_commit]['status'] = exit_status
            logjson(pre_commit, exit_status, f'{infer_out_dirpath}/commit_status.txt')

        if g._node[cur_commit]['status'] == "Cannot_Build" and g._node[pre_commit]['status'] == "Cannot_Build":
            break

        if g._node[cur_commit]['status'] == "Cannot_Build":
            cur = cur+1
            continue
        if g._node[pre_commit]['status'] == "Cannot_Build":
            pre = pre-1
            continue

        if g._node[cur_commit]['status'] == "Already_built" and g._node[pre_commit]['status'] == "Already_built":
            fix_exists = check_fix(cur_commit, pre_commit, infer_out_dirpath)
            if fix_exists == 1:
                if cur == pre-1:
                    fiexed_commit.append((cur_commit, pre_commit))
                else:
                    mid = int((cur+pre)/2)
                    binary_search(cur, mid, commits, g, infer_out_dirpath)
                    binary_search(mid, pre, commits, g, infer_out_dirpath)
            break


for repo in repo_dir:
    if repo_dir[repo] == 'failed':
        continue
    if repo in repo_already_check:
        continue
    fiexed_commit = []
    print(repo, repo_dir[repo])
    repo_name = repo.split('/')[-1]
    repo_id = repo.split('/')[-2]
    trg_repo = f'{tmp_repository_path}/{repo_name}'
    infer_out_dirpath = f'{infer_out_dir}/{repo_id}'
    if not os.path.exists(infer_out_dirpath):
        os.mkdir(infer_out_dirpath)

    g = nx.DiGraph()
    for commit in pydriller.Repository(trg_repo).traverse_commits():
        g.add_node(commit.hash, commit = commit, status = 'No_Check')
    for node in g:
        commit = g._node[node]['commit']
        if len(commit.parents)>0:
            for parent in commit.parents:
                g.add_edge(parent, node)

    linear_commits = get_linear_commit(g)


    for node in g._node:
        g._node[node]['status'] = "No_Check"
    if os.path.exists(f'{infer_out_dirpath}/commit_status.txt'):
        previous_commit_status = readjson(f'{infer_out_dirpath}/commit_status.txt')
        for hash in previous_commit_status:
            g._node[hash]['status'] = previous_commit_status[hash]


    for commits in linear_commits:
        cur = 0
        pre = len(commits)-1
        binary_search(cur, pre, commits, g, infer_out_dirpath)

    
    
    logjson(repo, fiexed_commit, fixed_pair_log_file_path)
    shutil.rmtree(trg_repo)
    if os.path.exists(f'{infer_out_dirpath}/tmp'):
        shutil.rmtree(f'{infer_out_dirpath}/tmp')


    

    