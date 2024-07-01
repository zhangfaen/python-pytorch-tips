# coding: utf-8
# This is a collection of util functions for pdbpp debugging.
# How to use this:
# 1. !pip install pdbpp
# 2. vim ~/.pdbrc.py
# 3. put below code into .pdbrc.py
#    import sys
#    import pdb

#    # update below line to indicate where your pdbpp until functions file is.
#    sys.path.append('/home/zhangfaen/dev/python-pytorch-tips/') 
#    from pdb_pdbpp import grep
#    pdb.grep = grep
# 4. in pdb debugging session, you can use like this way pdb.grep('normalize', dir(a_pytorch_tensor_object))


import re
from pprint import pprint 
from typing import List


def grep(pattern: str, target: List[str] | str):
    """an util function to search a pattern in a string list of string"""
    if isinstance(target, str):
        return re.search(pattern, target)
    else:
        return [item for item in target if re.search(pattern, item)]



