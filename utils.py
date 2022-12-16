# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:44:54 2022

@author: tsoukj
"""

import os
import shutil
import platform
import time
import subprocess
import pickle
import json
import requests
import pandas as pd
import numpy as np
from git import Git, Repo, exc

def clone_project(cwd, git_url, git_commit):
    print('~~~~~~ Cloning repo ~~~~~~')

    if git_commit == 'latest':
        # Get latest commit hash
        git_commit = Git().ls_remote(git_url, 'HEAD')[0:40]
            
    # Set repo name from GitHub URL
    if '.git' not in git_url:
        repo_name = (('%s_%s_%s') % (git_url.split('/')[-2], (git_url.split('/')[-1]), git_commit)).lower().strip()
    else:
        repo_name = (('%s_%s_%s') % (git_url.split('/')[-2], (git_url.split('/')[-1]).split('.')[-2], git_commit)).lower().strip()
    
    # Create repo clone directory
    clone_dir = r'%s/projects_cloned/%s' % (cwd, repo_name)
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    
    # Clone repo 
    start_time = time.time()
    try:
        # If repo is already cloned then use existing one
        cloned_repo = Repo(clone_dir)
        print('- Repo %s already exists in %s (%s sec)' % (repo_name, cloned_repo.working_tree_dir, round(time.time() - start_time, 2)))
    except exc.InvalidGitRepositoryError:
        # .. else clone a new one
        cloned_repo = Repo.clone_from(git_url, clone_dir, no_checkout=True)
        cloned_repo.git.checkout(git_commit)
        print('- Successfully cloned repo %s in %s (%s sec)' % (repo_name, cloned_repo.working_tree_dir, round(time.time() - start_time, 2)))
    
    return repo_name, cloned_repo.working_tree_dir, cloned_repo.head.object.hexsha

def run_ck(cwd, path, repo_name):
    print('~~~~~~ Running CK ~~~~~~')
    
    # Create ck results directory
    ck_dir = r'%s/tool_results/ck/%s' % (cwd, repo_name)
    if not os.path.exists(ck_dir):
        os.makedirs(ck_dir)
    
    # Execute ck jar
    start_time = time.time()
    COMMAND = r'java -jar %s/lib/ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar %s false 0 false %s/' % (cwd, path, ck_dir)
    p = subprocess.Popen(COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    exit_code = p.returncode
    stdout = stdout
    stderr = stderr
    
    if exit_code != 0:
        print('- Error executing command [%s]\n- Exit code [%s]\n- stderr: [%s]\n- stdout: [%s]' % (COMMAND, exit_code, stderr, stdout))
    else:
        print('- Successfully fetched CK metrics (%s sec)' % round(time.time() - start_time, 2))

def run_cpd(cwd, path, repo_name):
    print('~~~~~~ Running CPD ~~~~~~')
    
    # Create cpd results directory
    cpd_dir = r'%s/tool_results/cpd/%s' % (cwd, repo_name)
    if not os.path.exists(cpd_dir):
        os.makedirs(cpd_dir)
    
    # Execute cpd tool
    start_time = time.time()
    if platform.system() == 'Windows':
        COMMAND = r'%s/lib/pmd/bin/cpd.bat --minimum-tokens 100 --files %s --skip-lexical-errors --format csv > %s/%s_duplication_measures.csv' % (cwd, path, cpd_dir, repo_name)
    elif platform.system() == 'Linux':
        COMMAND = r'%s/lib/pmd/bin/run.sh cpd --minimum-tokens 100 --files %s --skip-lexical-errors --format csv > %s/%s_duplication_measures.csv' % (cwd, path, cpd_dir, repo_name)
    p = subprocess.Popen(COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    exit_code = p.returncode
    stdout = stdout
    stderr = stderr
    
    if exit_code != 0 and exit_code != 4:
        print('- Error executing command [%s]\n- Exit code [%s]\n- stderr: [%s]\n- stdout: [%s]' % (COMMAND, exit_code, stderr, stdout))
    else:
        print('- Successfully fetched CPD metrics (%s sec)' % round(time.time() - start_time, 2))

def run_cloc(cwd, path, repo_name):
    print('~~~~~~ Running cloc ~~~~~~')
    
    # Create cpd results directory
    cloc_dir = r'%s/tool_results/cloc/%s' % (cwd, repo_name)
    if not os.path.exists(cloc_dir):
        os.makedirs(cloc_dir)
    
    # Execute cloc tool
    start_time = time.time()
    if platform.system() == 'Windows':
        COMMAND = r'%s/lib/cloc-1.88.exe %s --by-file --force-lang="Java",java --include-ext=java --csv --out="%s/%s_comments_measures.csv"' % (cwd, path, cloc_dir, repo_name)
    elif platform.system() == 'Linux':
        COMMAND = r'cloc %s --by-file --force-lang="Java",java --include-ext=java --csv --out="%s/%s_comments_measures.csv"' % (path, cloc_dir, repo_name)
    p = subprocess.Popen(COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    exit_code = p.returncode
    stdout = stdout
    stderr = stderr
    
    if exit_code != 0:
        print('- Error executing command [%s]\n- Exit code [%s]\n- stderr: [%s]\n- stdout: [%s]' % (COMMAND, exit_code, stderr, stdout))
    else:
        print('- Successfully fetched CLOC metrics (%s sec)' % round(time.time() - start_time, 2))

def run_sam(cwd, path, repo_name, git_url, commit_sha):
    print('~~~~~~ Running SAM ~~~~~~')
    
    # Create SAM results directory
    sam_dir = r'%s/tool_results/sam/%s' % (cwd, repo_name)
    if not os.path.exists(sam_dir):
        os.makedirs(sam_dir)
    
    # Execute SAM tool
    start_time = time.time()
    url_measures = 'http://160.40.52.130:8088/DependabilityToolbox/SecurityAssessment?project=%s&commit_sha=%s&lang=java&inspection=yes&user_name=data_collector' % (git_url, commit_sha)
    get_measures = requests.get(url_measures)
    
    if get_measures.status_code != 200:
        print('- Error fetching results from SAM tool.')
    else:
        with open(r'%s/%s_sam_raw.json' % (sam_dir, repo_name), 'w') as outfile:
            outfile.write(get_measures.text)
        print('- Successfully fetched SAM metrics (%s sec)' % round(time.time() - start_time, 2))

def merge_results(cwd, path, repo_name, commit_sha):
    print('~~~~~~ Merging metrics ~~~~~~')
    start_time = time.time()
    metrics_df = pd.DataFrame()
    
    # Read ck csv results
    ck_dir = r'%s/tool_results/ck/%s' % (cwd, repo_name)
    temp_dataset = pd.read_csv(r'%s/class.csv' % ck_dir, sep=",")
    # Consider only entries marked as 'class'
    temp_dataset = temp_dataset.loc[temp_dataset['type'] == 'class']
    # Remove useless prefix from class path and replace '\' with '/'
    if platform.system() == 'Windows':
        temp_dataset['file'] = temp_dataset['file'].apply(lambda x: x.split('%s\\' % path)[-1].replace('\\', '/'))
    elif platform.system() == 'Linux':
        temp_dataset['file'] = temp_dataset['file'].apply(lambda x: x.split('%s/' % path)[-1])
    # Get class name
    temp_dataset['class'] = temp_dataset['class'].apply(lambda x: f"{x.split('.')[-1]}.java")
    # Get commit sha
    temp_dataset['commit_sha'] = commit_sha
    # Merge ck metrics with metrics_df
    metrics_df = temp_dataset[['file','class','commit_sha','cbo','wmc','dit','rfc','lcom','lcom*','publicMethodsQty','totalMethodsQty','maxNestedBlocksQty','variablesQty','fanin','fanout','noc']].rename(columns = {'file':'class_path','class':'class_name','lcom*':'lcom3','publicMethodsQty':'npm','totalMethodsQty':'total_methods','maxNestedBlocksQty':'max_nested_blocks','variablesQty':'total_variables'})
    # Export dataframe to csv
    # metrics_df.to_csv(r'%s/%s_ck_measures.csv' % (ck_dir, repo_name), sep=',', na_rep='', index=False)
    print('- Successfully merged CK metrics')
    
    # Read CPD csv results 
    cpd_dir = r'%s/tool_results/cpd/%s' % (cwd, repo_name)
    temp_dataset = pd.read_table(r'%s/%s_duplication_measures.csv' % (cpd_dir, repo_name))
    if not temp_dataset.empty:
        temp_dataset = temp_dataset.iloc[:,0].str.split(',', expand=True, regex=False)
        temp_dataset_2 = pd.DataFrame()
        # create tupples with duplicated line intervals
        for index, row in temp_dataset.iterrows():
            dup_lines = row[0] # get lines
            dup_classes = row[3:].reset_index(drop=True) # get occurences
            for i, v in dup_classes.items():
                # start from even columns that are not None
                if i % 2 == 0 and dup_classes[i] != None:
                    temp_dataset_2 = pd.concat([temp_dataset_2, pd.DataFrame([{'class_path': dup_classes[i+1], 'line_tuple': [int(dup_classes[i]), int(dup_classes[i])+int(dup_lines)]}])], ignore_index=True)
        # group dataframe by class and merge each classe line tuples into a list         
        temp_dataset_2 = pd.DataFrame(temp_dataset_2.groupby('class_path').apply(lambda x: list(np.unique(x)))).reset_index()
        # merge overlapping tuple intervals
        for i, temp_tuple in temp_dataset_2[0].items():
            temp_tuple.sort(key=lambda interval: interval[0])
            merged = [temp_tuple[0]]
            for current in temp_tuple:
                previous = merged[-1]
                if current[0] <= previous[1]:
                    previous[1] = max(previous[1], current[1])
                else:
                    merged.append(current)
            temp_dataset_2[0][i] = merged
        # calculate total duplicate lines per class from tuple intervals
        sum_tuple_list = []
        for i, temp_tuple in temp_dataset_2[0].items():
            sum_tuple = 0
            for tuple_list in temp_tuple:
                sum_tuple = sum_tuple + (tuple_list[1] - tuple_list[0] + 1)
            sum_tuple_list.append(sum_tuple)
        temp_dataset_2[0] = sum_tuple_list
        temp_dataset_2.rename(columns = {0:'duplicated_lines'}, inplace=True)
        # Remove useless prefix from class path and replace '\' with '/'
        if platform.system() == 'Windows':            
            temp_dataset_2['class_path'] = temp_dataset_2['class_path'].apply(lambda x: x.split('%s\\' % path)[-1].replace('\\', '/'))
        elif platform.system() == 'Linux':
            temp_dataset_2['class_path'] = temp_dataset_2['class_path'].apply(lambda x: x.split('%s/' % path)[-1])
        # Merge CPD metrics dataframe with metrics_df
        metrics_df = pd.merge(metrics_df, temp_dataset_2[['class_path','duplicated_lines']], on=['class_path'], how='left')
        # Fill NaN values of duplicated_lines with zeros
        metrics_df['duplicated_lines'].fillna(0, inplace=True)
    else:
        metrics_df['duplicated_lines'] = 0
    # Export dataframe to csv
    # metrics_df.to_csv(r'%s/%s_cpd_measures.csv' % (cpd_dir, repo_name), sep=',', na_rep='', index=False)
    print('- Successfully merged CPD metrics')
    
    # Read cloc csv results
    cloc_dir = r'%s/tool_results/cloc/%s' % (cwd, repo_name)
    temp_dataset = pd.read_csv(r'%s/%s_comments_measures.csv' % (cloc_dir, repo_name), sep=",")
    # Remove last row with redundant data
    temp_dataset = temp_dataset[:-1]
    # Remove useless prefix from class path and replace '\' with '/'
    if platform.system() == 'Windows':            
        temp_dataset['filename'] = temp_dataset['filename'].apply(lambda x: x.split('%s\\' % path.lower())[-1].replace('\\', '/'))
    elif platform.system() == 'Linux':
        temp_dataset['filename'] = temp_dataset['filename'].apply(lambda x: x.split('%s/' % path)[-1])
    # Rename colum and calculate total lines and 
    temp_dataset.rename(columns = {'comment':'comment_lines', 'code':'ncloc'}, inplace=True)
    temp_dataset['total_lines'] = temp_dataset['blank'] + temp_dataset['ncloc'] + temp_dataset['comment_lines']
    # Merge cloc metrics dataframe with the updated dataset
    if platform.system() == 'Windows':
        metrics_df = metrics_df.merge(temp_dataset[['filename','comment_lines','ncloc','total_lines']], left_on=metrics_df['class_path'].str.lower(), right_on=['filename'], how='left').drop(columns = ['filename'])
    elif platform.system() == 'Linux':
        metrics_df = metrics_df.merge(temp_dataset[['filename','comment_lines','ncloc','total_lines']], left_on=metrics_df['class_path'], right_on=['filename'], how='left').drop(columns = ['filename'])
    # Fill NaN values of comment_lines with zeros
    metrics_df['comment_lines'].fillna(0, inplace=True)
    # Export dataframe to csv
    # metrics_df.to_csv(r'%s/%s_cloc_measures.csv' % (cloc_dir, repo_name), sep=',', na_rep='', index=False)
    print('- Successfully merged CLOC metrics')
    
    # Read SAM results
    sam_dir = r'%s/tool_results/sam/%s' % (cwd, repo_name)
    with open(r'%s/%s_sam_raw.json' % (sam_dir, repo_name), 'r') as f:
        sam_data = json.load(f)
    for index, row in metrics_df.iterrows():
        counter_total = 0
        for property in ['Resource_Handling','Assignment','Exception_Handling','Misused_Functionality','Synchronization','Null_Pointer','Logging']:
            counter = 0
            property_issues = [x['issues'] for x in sam_data['issues'] if x['propertyName'] == property][0]
            for issue in property_issues:
                if row['class_path'] == issue['classPath']:
                    counter += 1
                    counter_total += 1
            metrics_df.loc[index, '%s_issues' % property.lower()] = counter
        metrics_df.loc[index, 'total_issues'] = counter_total
    # Export dataframe to csv
    # metrics_df.to_csv(r'%s/%s_sam_measures.csv' % (sam_dir, repo_name), sep=',', na_rep='', index=False)
    print('- Successfully merged SAM metrics')
    
    
    print('- All tools metrics successfully merged (%s sec)' % round(time.time() - start_time, 2))
    
    return metrics_df

def run_classifier(cwd, metrics_df):
    print('~~~~~~ Running classifier ~~~~~~')
    
    pd.options.mode.chained_assignment = None  # default='warn'
    start_time = time.time()
    
    features = ['cbo','wmc','dit','rfc','lcom','total_methods','max_nested_blocks',
                'total_variables','ncloc','duplicated_lines','comment_lines']
    
    metrics_df = metrics_df[features]
    
    # Replace 'ncloc' label with 'ncloc_cloc'
    metrics_df['ncloc_cloc'] = metrics_df['ncloc']
    
    # Divide by lines of code
    metrics_df['duplicated_lines_cpd_density'] = metrics_df['duplicated_lines'] / metrics_df['ncloc_cloc']
    metrics_df['comment_lines_cloc_density'] = metrics_df['comment_lines'] / (metrics_df['ncloc_cloc'] + metrics_df['comment_lines'])
    
    # Multiply by 100 to transform range between [0,100]
    metrics_df['duplicated_lines_cpd_density'] = metrics_df['duplicated_lines_cpd_density'] * 100
    metrics_df['comment_lines_cloc_density'] = metrics_df['comment_lines_cloc_density'] * 100
    
    # Fill any nan lines with zeros
    metrics_df.fillna(0, inplace=True)
    
    # Remove replaced features
    metrics_df.drop('ncloc', axis=1, inplace=True)
    metrics_df.drop('duplicated_lines', axis=1, inplace=True)
    metrics_df.drop('comment_lines', axis=1, inplace=True)
    
    # Load the model and scaler from disk
    loaded_model = pickle.load(open(r'%s/models/finalized_model_no_rm_no_pd.sav' % cwd, 'rb'))
    loaded_scaler = pickle.load(open(r'%s/models/finalized_scaler_no_rm_no_pd.sav' % cwd, 'rb'))
    
    # Transform to scaled data
    metrics_df = loaded_scaler.transform(metrics_df)

    # Predict the high-TD probability of classes
    y_pred_proba = loaded_model.predict_proba(metrics_df)[:, 1]
    
    # Get the high-TD classes
    y_pred = []
    for i in range(len(y_pred_proba)):
        if y_pred_proba[i] > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    print('- Successfully run classifier (%s sec)' % round(time.time() - start_time, 2))
    
    return y_pred, y_pred_proba

def aggregate_project_level(cwd, git_url, repo_name, commit_sha, metrics_df):
    print('~~~~~~ Aggregating metrics at project level ~~~~~~')
    start_time = time.time()
    
    ncloc_sum = metrics_df['ncloc'].sum()
    cbo_weighted = (metrics_df['cbo'] * metrics_df['ncloc']).sum() / ncloc_sum
    wmc_weighted = (metrics_df['wmc'] * metrics_df['ncloc']).sum() / ncloc_sum
    dit_weighted = (metrics_df['dit'] * metrics_df['ncloc']).sum() / ncloc_sum
    rfc_weighted = (metrics_df['rfc'] * metrics_df['ncloc']).sum() / ncloc_sum
    lcom_weighted = (metrics_df['lcom'] * metrics_df['ncloc']).sum() / ncloc_sum
    npm_weighted = (metrics_df['npm'] * metrics_df['ncloc']).sum() / ncloc_sum
    fanin_weighted = (metrics_df['fanin'] * metrics_df['ncloc']).sum() / ncloc_sum
    fanout_weighted = (metrics_df['fanout'] * metrics_df['ncloc']).sum() / ncloc_sum
    noc_weighted = (metrics_df['noc'] * metrics_df['ncloc']).sum() / ncloc_sum
    duplicated_lines_sum = metrics_df['duplicated_lines'].sum()
    comment_lines_sum = metrics_df['comment_lines'].sum()
    total_lines_sum = metrics_df['total_lines'].sum()
    high_td_sum = metrics_df['high_td'].sum()
    total_issues_sum = metrics_df['total_issues'].sum()
    
    # Read SAM results
    sam_dir = r'%s/tool_results/sam/%s' % (cwd, repo_name)
    with open(r'%s/%s_sam_raw.json' % (sam_dir, repo_name), 'r') as f:
        sam_data = json.load(f)
    
    agg_metrics_df = pd.DataFrame({
        'project_name': repo_name,
        'git_url': git_url,
        'commit_sha': commit_sha,
        'cbo': [cbo_weighted],
        'wmc': [wmc_weighted],
        'dit': [dit_weighted],
        'rfc': [rfc_weighted],
        'lcom': [lcom_weighted],
        'npm': [npm_weighted],
        'fanin': [fanin_weighted],
        'fanout': [fanout_weighted],
        'noc': [noc_weighted],
        'ncloc': [ncloc_sum],
        'duplicated_lines': [duplicated_lines_sum],
        'duplicated_lines_density': [duplicated_lines_sum / ncloc_sum],
        'comment_lines': [comment_lines_sum],
        'comment_lines_density': [comment_lines_sum / total_lines_sum],
        'total_lines': [total_lines_sum],
        'high_td_classes': [high_td_sum],
        'high_td_classes_density': [high_td_sum / len(metrics_df)],
        'sam_resource_handling_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Resource_Handling'],
        'sam_resource_handling_density': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Resource_Handling'],
        'sam_assignment_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Assignment'],
        'sam_assignment_density': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Assignment'],
        'sam_exception_handling_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Exception_Handling'],
        'sam_exception_handling_density': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Exception_Handling'],
        'sam_misused_functionality_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Misused_Functionality'],
        'sam_misused_functionality_density': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Misused_Functionality'],
        'sam_synchronization_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Synchronization'],
        'sam_synchronization_density': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Synchronization'],
        'sam_null_pointer_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Null_Pointer'],
        'sam_null_pointer_density': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Null_Pointer'],
        'sam_logging_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Logging'],
        'sam_logging_density': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Logging'],
        'sam_cohesion_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Cohesion'],
        'sam_cohesion_norm': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Cohesion'],
        'sam_coupling_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Coupling'],
        'sam_coupling_norm': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Coupling'],
        'sam_complexity_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Complexity'],
        'sam_complexity_norm': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Complexity'],
        'sam_encapsulation_eval': [x['eval'] for x in sam_data['properties']['properties'] if x['name'] == 'Encapsulation'],
        'sam_encapsulation_norm': [x['measure']['normValue'] for x in sam_data['properties']['properties'] if x['name'] == 'Encapsulation'],
        'sam_confidentiality_eval': [x['eval'] for x in sam_data['characteristics']['characteristics'] if x['name'] == 'Confidentiality'],
        'sam_integrity_eval': [x['eval'] for x in sam_data['characteristics']['characteristics'] if x['name'] == 'Integrity'],
        'sam_availability_eval': [x['eval'] for x in sam_data['characteristics']['characteristics'] if x['name'] == 'Availability'],
        'sam_total_issues': [total_issues_sum],
        'security_index': sam_data['security_index']['eval']
    })
    
    print('- All project-level metrics successfully aggregated (%s sec)' % round(time.time() - start_time, 2))
    
    return agg_metrics_df  

def export_results(cwd, repo_name, metrics_df, agg_metrics_df):       
    # Create results directory
    data_dir = r'%s/results/%s' % (cwd, repo_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Export dataframe to csv
    metrics_df.to_csv(r'%s/%s_class_level.csv' % (data_dir, repo_name), sep=',', na_rep='', index=False)
    agg_metrics_df.to_csv(r'%s/%s_project_level.csv' % (data_dir, repo_name), sep=',', na_rep='', index=False)
        
    # Export dataframe to json
    metrics_df.to_json(r'%s/%s_class_level.json' % (data_dir, repo_name), orient='records')
    agg_metrics_df.to_json(r'%s/%s_project_level.json' % (data_dir, repo_name), orient='records')
    
    # Export dataframe to html
    metrics_df.to_html(r'%s/%s_class_level.html' % (data_dir, repo_name), justify='left')
    agg_metrics_df.to_html(r'%s/%s_project_level.html' % (data_dir, repo_name), justify='left')
    
    print('- Detailed results in csv, json and html format can be found in %s folder' % data_dir)

def remove_temp_files(cwd, repo_name):
    clone_dir = r'%s/projects_cloned/%s' % (cwd, repo_name)
    ck_dir = r'%s/tool_results/ck/%s' % (cwd, repo_name)
    cpd_dir = r'%s/tool_results/cpd/%s' % (cwd, repo_name)
    cloc_dir = r'%s/tool_results/cloc/%s' % (cwd, repo_name)
    
    shutil.rmtree(clone_dir)
    shutil.rmtree(ck_dir)
    shutil.rmtree(cpd_dir)
    shutil.rmtree(cloc_dir)
