import sys, os

this_dir = os.path.dirname(os.path.abspath(__file__))
PATH_APP = os.path.abspath(os.path.join(this_dir, '..'))
PATH_BINDS = os.path.join(PATH_APP, 'bindings')
PATH_SCRIPTS = os.path.join(PATH_APP, 'scripts')
sys.path.append(PATH_BINDS)