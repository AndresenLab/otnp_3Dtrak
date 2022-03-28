# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:18:10 2018

@author: jm2080
"""

import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename

def filedlg(idir, topname):
    """
    Offer dialog box for user to select a directory.
    
    Example:
        filedlg("E:/Google Drive", "Choose root directory")
    """
    return askdirectory(parent=root, initialdir=idir, title=topname)
root = tk.Tk()
root.lift()
root.attributes("-topmost", True)
root.withdraw()

def opnfile(idir, topname):
    """
    Offer dialog box for user to select a file.
    
    Example:
        opnfile("E:/Google Drive", "Choose file")
    """
    return askopenfilename(initialdir=idir,
                           filetypes=(("NPY File", "*.npy"),
                                      ("Comma Separated Values File", "*.csv"),
                                      ("Text File", "*.txt"),
                                      ("All Files", "*.*")), title=topname)
root = tk.Tk()
root.lift()
root.attributes("-topmost", True)
root.withdraw()
