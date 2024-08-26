import shutil
import os
root="outputs"

dir_list=os.listdir(root)
for dir in dir_list:
    dir_path=os.path.join(root,dir)
    check_path=os.path.join(dir_path,"checkpoints/best.pth")
    if not os.path.exists(check_path):
        shutil.rmtree(dir_path)
        print(f"remove {dir_path}")
    else:
        print(f"keep {dir_path}")