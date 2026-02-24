import os, sys
from monty.serialization import loadfn

try:
    config_vars = loadfn(os.path.join(os.path.expanduser('~'), 'myml4hea.yaml'))
except:
    sys.exit('No myml4hea.yaml file was found. Please configure the '
             ' myml4hea.yaml and put it in your home directory.')

