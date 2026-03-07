import numpy as np
import torch
# import linearsum_assignment
from scipy.optimize import linear_sum_assignment
import cv2
from tqdm import tqdm
import os

def read_txt(path):
    results=[]
    with open(path,"r") as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip().split(" ")
            file=line[0]
            num=int(line[1])
            pts=[]
            for i in range(num):
                pts.append([int(line[2*i+2]),int(line[2*i+3])])
            pts=np.array(pts)
            results.append([file,pts])
    return results
        
def match_pts(pts1,pts2):
    # pts1: (N,2)
    # pts2: (M,2)
    # return: (N,2)
    # N<=M
    cost_matrix=np.zeros((pts1.shape[0],pts2.shape[0]))
    for i in range(pts1.shape[0]):
        for j in range(pts2.shape[0]):
            cost_matrix[i,j]=np.linalg.norm(pts1[i]-pts2[j])
    row_ind,col_ind=linear_sum_assignment(cost_matrix)
    cdist=cost_matrix[row_ind,col_ind]
    unmatched_pts2=[]
    for i in range(pts2.shape[0]):
        if i not in col_ind:
            unmatched_pts2.append(pts2[i])
    pts2=pts2[col_ind]
    
    return pts2,cdist,unmatched_pts2

high_resolution_file_path="local_eval/nwpu_test_pred_v2.txt"
low_resolution_file_path="local_eval/nwpu_test_pred_v3.txt"
high_resolution_results=read_txt(high_resolution_file_path)
low_resolution_results=read_txt(low_resolution_file_path)
data_root="data/NWPU_origin/images"
mergered_results=[]
for i in tqdm(range(len(high_resolution_results))):
    pts1=high_resolution_results[i][1]
    pts2=low_resolution_results[i][1]
    mergered_pts=[]
    img=cv2.imread(os.path.join(data_root,high_resolution_results[i][0]+".jpg"))
    if pts1.shape[0]<pts2.shape[0]:
        pts2,cdist,unmatched_pts=match_pts(pts1,pts2)
    else:
        pts1,cdist,unmatched_pts=match_pts(pts2,pts1)
    
    for pt1,pt2 in zip(pts1,pts2):
        d=np.linalg.norm(pt1-pt2)
        if d>33:
            mergered_pts.append(pt1)
            mergered_pts.append(pt2)
            cv2.circle(img,(pt1[0],pt1[1]),10,(0,0,255),2)
            cv2.circle(img,(pt2[0],pt2[1]),10,(0,255,0),2)
            cv2.line(img,(pt1[0],pt1[1]),(pt2[0],pt2[1]),(255,0,0),2)
        else:
            mergered_pts.append(pt1)
            cv2.circle(img,(pt1[0],pt1[1]),10,(255,0,255),-1)

    for pt in unmatched_pts:
        cv2.rectangle(img,(pt[0]-10,pt[1]-10),(pt[0]+10,pt[1]+10),(0,255,255),2)
    cv2.imwrite(f"local_eval/merge/{high_resolution_results[i][0]}.jpg",img)
    mergered_pts.extend(unmatched_pts)
    mergered_results.append([high_resolution_results[i][0],np.array(mergered_pts)])

output_path="local_eval/nwpu_test_pred_v4.txt"
output_file=open(output_path,"w")
for result in mergered_results:
    output_file.write(f"{result[0]} {len(result[1])} ")
    for pt in result[1]:
        output_file.write(f"{round(pt[0])} {round(pt[1])} ")
    output_file.write("\n")
output_file.close()

    

