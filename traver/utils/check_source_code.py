import os, sys
from subprocess import run

source_code_root = sys.argv[1]

project_list = os.listdir(source_code_root)
for project in project_list:
    project_path = os.path.join(source_code_root, project)
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                if 'tmp_' + file in files:
                    print(os.path.join(root, file))
                    print(os.path.join(root, 'tmp_' + file))
                    print('---------------------')
                    run(['mv', os.path.join(root, 'tmp_' + file), os.path.join(root, file)])