import json
import os
import sys
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', help='Path to the data.jsonl file')
    parser.add_argument('--source_code_root', help='Path to the source code directory')
    return parser.parse_args()


def main():
    args = parse_args()

    test_file_paths = []
    with open(args.data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            project_name = data['completion_path'].split('/')[0]
            project_path = os.path.join(args.source_code_root, project_name)
            for test in data['tests']:
                test_path = test.split('::')[0]
                if test_path not in test_file_paths:
                    test_file_paths.append((project_path, test_path))

    test_file_paths = set(test_file_paths)

    for item in test_file_paths:
        project_path, test_path = item
        test_path = os.path.join(project_path, test_path)
        code = ""
        with open(test_path, 'r') as f:
            code = f.read()
            _import = f'import sys\nsys.path.append("{project_path}")\n'
            code = _import + code
        with open(test_path, 'w') as f:
            f.write(code)
        print(f'Updated {test_path}')

if __name__ == '__main__':
    main()