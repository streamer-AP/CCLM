import os
import shutil

_ignore_files_ = ["analysis", "outputs", "configs", "__pycache__","data",".git",".vscode", "local_eval","output_old","weights"]

def copy_files(src_path, dest_path):
    files = os.listdir(src_path)
    for f in files:
        if len(f) >= 4:
            if f[-4:] in [".out"]:
                continue

        if f not in _ignore_files_:
            next_src_path = os.path.join(src_path, f)
            next_dest_path = os.path.join(dest_path, f)
            if os.path.isfile(next_src_path):
                # print("src", next_src_path)
                # print("tgt", next_dest_path)
                shutil.copy(next_src_path, next_dest_path)
                # print(next_src_path, "  ", next_dest_path)
            if os.path.isdir(next_src_path):
                if not os.path.exists(next_dest_path):
                    os.makedirs(next_dest_path)
                copy_files(next_src_path, next_dest_path)