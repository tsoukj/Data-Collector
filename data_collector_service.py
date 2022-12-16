# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:40:37 2022

@author: tsoukj
"""

import argparse
import sys
import os
import time
from utils import clone_project, run_ck, run_cpd, run_cloc, export_results, merge_results, run_classifier, aggregate_project_level, run_sam

#===============================================================================
# td_classifier ()
#===============================================================================
def data_collector(git_url, git_commit):
    """
    API Call to DataCollector analyser
    Arguments:
        git_url: retrieved from create_arg_parser() as a string
        git_commit: retrieved from create_arg_parser() as a string
    Returns:
        A JSON containing the classification results stored in "/results" folder
        and the intermediate static analysis results stored in "/tool_results" 
        folder.
    """
    
    start_time = time.time()
    
    # Set current working directory
    cwd = os.environ.get('CWD')
    
    # Get path of cloned project and its repo name (identifier)
    repo_name, path, commit_sha = clone_project(cwd, git_url, git_commit)
    
    # Check if provided project directory exists
    if os.path.isdir(path) or os.path.isfile(path):
        # Run CK
        run_ck(cwd, path, repo_name)
        # Run CPD
        run_cpd(cwd, path, repo_name)
        # Run CLOC
        run_cloc(cwd, path, repo_name)
        # Run SAM
        run_sam(cwd, path, repo_name, git_url, commit_sha)
        # Merge results
        metrics_df = merge_results(cwd, path, repo_name, commit_sha)
        
        # Drop NAN values
        metrics_df.dropna(inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)
        
        # Run classifier
        y_pred, y_pred_proba = run_classifier(cwd, metrics_df)
        metrics_df['high_td'] = y_pred
        metrics_df['high_td_proba'] = y_pred_proba
        
        # Aggregate class-level metrics at project level
        agg_metrics_df = aggregate_project_level(cwd, git_url, repo_name, commit_sha, metrics_df)
        
        # Export results in csv, json and html
        export_results(cwd, repo_name, metrics_df, agg_metrics_df)
        
        # Remove temp files like source code, analysis intermediate results, etc.
        # remove_temp_files(cwd, repo_name)
        
        print('- Successfully finished process in %s sec. Found %s high-TD classes (out of %s)' % (round(time.time() - start_time, 2), metrics_df['high_td'].sum(), metrics_df.shape[0]))
    else:
        print('- The project directory "%s" does not exist. Please provide a valid directory.' % path)

#===============================================================================
# run_server ()
#===============================================================================
def run_analyser(git_url, git_commit, cwd):
    """
    Executes the command to start the analyser
    Arguments:
        git_url: retrieved from create_arg_parser() as a string
    """

    print('GitHub URL:         %s' % (git_url))
    print('Commit SHA:         %s' % (git_commit))
    print('working_directory:  %s' % (cwd))

    # Store settings in environment variables
    os.environ['CWD'] = str(cwd)

    data_collector(git_url, git_commit)

#===============================================================================
# create_arg_parser ()
#===============================================================================
def create_arg_parser():
    """
    Creates the parser to retrieve arguments from the command line
    Returns:
        A Parser object
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('u', metavar='GIT_URL', help='The GitHub URL of the project to be analysed pointing to a public project.', type=str)
    parser.add_argument('-cs', metavar='COMMIT_SHA', help='The hexadecimal sha of a specific commit to be analysed. If ommited then the latest commit is considered by default.', type=str, default='latest')
    
    return parser

#===============================================================================
# main ()
#===============================================================================
def main():
    """
    The main() function of the script acting as the entry point
    """
    parser = create_arg_parser()

    # If script run without arguments, print syntax
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()
    git_url = args.u
    git_commit = args.cs
        
    # Set current working directory
    cwd = os.getcwd()

    # Run analyser with user-given arguments
    run_analyser(git_url, git_commit, cwd)

if __name__ == '__main__':
    main()
