import logging
import os
from tqdm import tqdm
import dill as pickle

from pyan_zyf_v2.analyzer import CallGraphVisitor
from pyan_zyf_v2.call_analyzer import CallAnalyzer, FolderMaker
from pyan_zyf_v2.anutils import get_module_name



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='func_call.log',
                    filemode='w')

def find_py_files(folder):
    py_files = []
    for root, dirs, files in os.walk(folder):
        if True in [item.startswith('.') for item in root.split(os.sep)]:
            continue
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

folder_path = "path/to/folder"
py_files = find_py_files(folder_path)


def process(target_object,func_object_root, func_path, analyzer_result, target_root):

    with open(func_path, 'r') as f:
        func_content = f.read()
    
    with open(analyzer_result, 'rb') as analyzer:
        v: CallGraphVisitor = pickle.loads(analyzer.read())
    
    virtual_path = func_path.replace(func_object_root, target_object)
    
    v.add_process_one(virtual_path, content=func_content)
    v.postprocess()

    namespace = get_module_name(virtual_path, root=None)

    graph = CallAnalyzer.from_visitor(v, target_root, prefix=namespace,logger=logger)
    folder_maker = FolderMaker(target_root)
    folder_maker.process(graph, v, target_object)
