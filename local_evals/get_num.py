f_o=open("local_eval/nwpu_test_guass_count.txt","w")
with open("local_eval/nwpu_test_guass.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip().split(" ")
        f_o.write(f"{line[0]} {line[1]}\n")
f_o.close()
    