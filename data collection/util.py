import os
import json
import re
import difflib
from bs4 import BeautifulSoup
import requests
   

def logjson(repo, content, path):
    json_str = json.dumps({repo:content})
    with open(path,"a+")as f:
        f.write(json_str+"\n")

def readjson(path):
    repo_dir = {}
    with open(path)as f:
        lines = f.readlines()
    for line in lines:
        tmp_dir = json.loads(line)
        repo_dir.update(tmp_dir)
    return repo_dir



def classify_repository(repo_name):
    if "autogen.sh" in os.listdir(repo_name) or "bootstrap" in os.listdir(repo_name):
        return 'custom'
    if "CMakeLists.txt" in os.listdir(repo_name):
        return 'cmake'
    elif "Makefile.am" in os.listdir(repo_name) or "configure.ac" in os.listdir(repo_name) or "configure" in os.listdir(repo_name) or "Makefile" in os.listdir(repo_name):
        return 'automake'
    else:
        return 'nobuild'

def check_compilation(trg_repo, mode):
    if mode == 'custom':
        if "autogen.sh" in os.listdir(trg_repo):
            compilation_cmd1 = "./autogen.sh"
        if "autogen.sh" in os.listdir(trg_repo):
            compilation_cmd1 = "bootstrap"
        cd_cmd = f'cd {trg_repo}'
        config_cmd = "./configure"
        compilation_cmd = "bear make"
        os.system(f'{cd_cmd} && {compilation_cmd1}')
        os.system(f'{cd_cmd} && {config_cmd}')
        os.system(f'{cd_cmd} && {compilation_cmd}')
        if os.path.exists(f'{trg_repo}/compile_commands.json'):
            return 1
    
    elif mode == "cmake":
        build_dir = f'{trg_repo}/build'
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        cd_cmd = f'cd {build_dir}'
        compilation_cmd = "cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .."
        os.system(f'{cd_cmd} && {compilation_cmd}')
        os.system(f'{cd_cmd} && bear make')
        if os.path.exists(f'{build_dir}/compile_commands.json'):
            return 1

    elif mode == 'automake':
        cd_cmd = f'cd {trg_repo}'
        config_cmd = "./configure"
        compilation_cmd = "bear make"
        os.system(f'{cd_cmd} && aclocal')
        os.system(f'{cd_cmd} && autoheader')
        os.system(f'{cd_cmd} && autoconf')
        os.system(f'{cd_cmd} && libtoolize')
        os.system(f'{cd_cmd} && automake --add-missing')
        os.system(f'{cd_cmd} && {config_cmd}')
        exit_code = os.system(f'{cd_cmd} && {compilation_cmd}')
        if exit_code == 0:
            return 1

    return 0

def check_commit(fixed_json_path, introduced_json_path):
    def remove(s):
        s = re.sub(u"\\`.*?\\`", "", s)
        s = re.sub(r'[0-9]+', '', s)
        return s

    def get_info(bug):
        bug_type = bug['bug_type']
        bug_type = bug_type.replace('_',' ')
        bug_qualifier = bug['qualifier']
        bug_qualifier = remove(bug_qualifier)
        bug_procedure = bug['procedure']
        bug_procedure = bug_procedure.replace('_',' ')
        bug_info = f'{bug_type} {bug_qualifier} {bug_procedure}'
        return bug_type, bug_info




    with open(fixed_json_path) as f:
        fixed_json = json.load(f)
    with open(introduced_json_path) as f:
        introduced_json = json.load(f)
    

    for fix_bug in fixed_json[::-1]:
        fix_bug_type, fix_bug_info = get_info(fix_bug)
        for intro_bug in introduced_json[::-1]:
            intro_bug_type, intro_bug_info = get_info(intro_bug)
            if fix_bug_type != intro_bug_type:
                continue
            else:
                simi_ratio = difflib.SequenceMatcher(None, fix_bug_info, intro_bug_info).quick_ratio()
                if simi_ratio>0.9:
                    fixed_json.remove(fix_bug)
                    introduced_json.remove(intro_bug)
                    break
                
    return len(fixed_json), fixed_json

def get_full_msg(url, commit, msg):
    if url == "":
        return msg
    res = re.findall(r"#\d+",msg)
    url_msg = ""
    if len(res)>0:
        url = url.replace(".git","/commit")
        url = f'{url}/{commit}'
        content = requests.get(url)
        soup = BeautifulSoup(content.text, 'html.parser')
        tag = soup.find_all('div')
        url_link = []
        for div in tag:
            if 'class'in div.attrs:
                if 'commit-title' in div['class'] or "commit-desc" in div['class']:
                    url = div.find_all('a')
                    for href in url:
                        href = href.get('href')
                        url_link.append(href)
        for link in url_link:
            if link.split("/")[-2] == 'pull' or link.split("/")[-2] == 'issues':
                content = requests.get(link)
                soup = BeautifulSoup(content.text, 'html.parser')
                tag = soup.find_all('td')
                for td in tag:
                    if 'class' in td.attrs:
                        if 'comment-body' in td['class']:
                            url_msg+=td.get_text()

        return msg + url_msg
    else:
        return msg
