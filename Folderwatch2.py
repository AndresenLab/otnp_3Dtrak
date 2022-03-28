# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:17:09 2018

@author: jm2080
"""

def folderwatch(directory, count):
    """
    Creates a dictionary containing the contents of a chosen directory.
    If the directory is empty it checks every 'count' number seconds until something is added.

    Example:
        folderwatch(C:/Users/jm2080/Google Drive/TBAnalysed, 30)
    """
    import os
    import time

    vidfiles = []
    while not vidfiles:
        print("Monitoring Folder")
        time.sleep(count)
        for dirpath, dirnames, files in os.walk(directory):
            if not files:
                continue
            for name in files:
                if name.lower().endswith('.avi'):
                    vidfiles.append(os.path.join(dirpath, name))
    return vidfiles
