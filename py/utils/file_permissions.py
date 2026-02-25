#!/usr/bin/python

from pathlib import Path
import json
import glob
import shutil
import os
import pwd
import grp
import pandas as pd
import numpy as np

from dotenv import load_dotenv

# change owner
def chown(filepath, user="root", group="root"):

    """ Function to change ownership
    
    Args:
        user: username
        group: group name
        
    Returns:
    
    """
    uid = pwd.getpwnam(user).pw_uid
    gid = grp.getgrnam(group).gr_gid
    shutil.chown(filepath, user = uid, group = gid)
    
# Change owner recursivo
def chown_recursive(path, user="root", group=None, recursive=False):
    
    """Function to Change user/group ownership of file

    Args:
        path: path of file or directory.
        user: new owner username
        group: new owner group
        recursive: set files/dirs recursively

    Returns:

    """
    if group is None:
        group = user
    
    uid = pwd.getpwnam(user).pw_uid
    gid = grp.getgrnam(group).gr_gid
    
    try:
        if not recursive or os.path.isfile(path):
            shutil.chown(path, user = uid, group = gid)
        else:
            for root, dirs, files in os.walk(path):
                shutil.chown(root, uid, gid)
                for item in dirs:
                    shutil.chown(os.path.join(root, item), user = uid, group = gid)
                for item in files:
                    shutil.chown(os.path.join(root, item), user = uid, group = gid)
    except OSError as e:
        print(f"Error saving file: {e}")

def openFile(filename):

    """ Function to  open a file
        
    Args:
        filename: name/path to file
        
    Returns:
        dict: config file

    """
    with open(filename) as f:
        config = json.load(f)

    return config

def get_current_path():

    """ Function to get current path
        
    Args:
        
    Returns:
        str: current path
    
    """
    
    return str(Path(__file__).parent.resolve())

def get_absolute_path(relative_path):

    """ Function to get absolute (from root) path
        
    Args:
        relative_path: some path
        
    Returns:
        str: current path
    
    """
    return Path(relative_path).absolute().as_posix()

def verifyDataFrame(df):
    columns_all_zeros = df.columns[(df == 0).all()].tolist()
    rows_all_zeros = df.iloc[:,1:].index[(df == 0).all(axis=1)].tolist()
    
    print(f"Rows with all zeros: {rows_all_zeros}, size: {len(rows_all_zeros)}")
    print(f"Columns with all zeros: {columns_all_zeros}, size: {len(columns_all_zeros)}")

def verifyFile(files_list):

    """ Function to verify if File exists
        
    Args:
        files_list: object/path
        
    Returns:
        bool: True, file exists, otherwise false.
    
    """
    return Path(files_list).is_file()

def verifyType(file_name):

    """ Function to verify type of file
        
    Args:
        file_name: path/file name
        
    Returns:
        str: dir if directory, otherwise file
    
    """
    if Path(file_name).is_dir():
        return "dir"
    elif Path(file_name).is_file():
        return "file"
    else:
        return None

def verifyDir(dir_path):

    """ Function to verify if dir exists
        
    Args:
        dir_path: path/dir name
        
    Returns:
    
    """
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, mode=0o777, exist_ok=True)

def load_parent_env():
    
    repo_root = Path(get_current_path()).parent.parent
    _ENV_PATH = repo_root / '.env'
    
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH)
    else:
        print(f"Warning: .env file not found at {_ENV_PATH}")