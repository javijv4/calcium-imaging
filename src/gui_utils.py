#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Shared GUI helpers for the calcium-imaging scripts."""

import os
import subprocess
import sys


def select_folder(initialdir=None):
    """Open folder picker; starts in ``initialdir`` or the process current working directory.

    Runs the native folder dialog in a separate process. On machines with
    Office installed, the native Windows folder dialog can pull Office's
    bundled MSVCP140.dll into whichever process opens it, which then
    conflicts with the MSVCP140.dll ITK/elastix needs later and crashes the
    process with no Python traceback. Isolating the dialog in a subprocess
    keeps that conflict out of the process that runs the registration.
    """
    start = os.getcwd() if initialdir is None else initialdir
    script = (
        "from tkinter import Tk, filedialog\n"
        "root = Tk()\n"
        "root.withdraw()\n"
        "path = filedialog.askdirectory(title='Select a Folder', initialdir=r'''{start}''')\n"
        "root.destroy()\n"
        "print(path, end='')\n"
    ).format(start=start)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
