import os

root_dir="/home/xinyan/workspace/counting/hrcrowd/outputs/nwpu_202310231028_hrnetv5/inference"
test_or_val="test"
file_list=f"data/NWPU_origin/{test_or_val}.txt"
# file_list="data/NWPU_origin/val.txt"
file_list=open(file_list,"r").readlines()
file_list=[file.strip().split(" ")[0] for file in file_list]
output_path=f"local_eval/nwpu_{test_or_val}_guass.txt"
output_file=open(output_path,"w")
for file in file_list:
    pt_file_path=os.path.join(root_dir,file+".txt")
    with open(pt_file_path,"r") as f:
        lines=f.readlines()
        pts=[]
        for line in lines:
            pt=line.strip().split(",")
            pt=[float(p) for p in pt]
            pts.append(pt)
        pts=sorted(pts,key=lambda x:x[0])
        output_file.write(f"{file} {len(pts)} ")
        for pt in pts:
            output_file.write(f"{round(pt[0])} {round(pt[1])} ")
        output_file.write("\n")
    