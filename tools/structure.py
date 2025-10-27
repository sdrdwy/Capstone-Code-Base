#txt2jsonl
FILE_IN_PATH = "皮肤病中医诊疗学fix.txt"
FILE_OUT_PATH = "诊疗学struct.txt"
file_in = open(FILE_IN_PATH,"r")
file_out = open(FILE_OUT_PATH,"w")
content = file_in.readlines()
content_out = []
print(f"origin length: {len(content)}")
i = 0
while(i<len(content)):
    if "---" in content[i]:
        i += 1 #忽略页码标
    else:
        content_out.append(content[i])
        i += 1

def get_section(content,idx,name):
    s = ""
    for i in range(idx,len(content)):
        if name in content[i]:
            idx = i
            break
        else:
            s += content[i]
    if name == "病名释义":
        s.replace(content[idx-1],"") if content[idx-1] != "\n" else s.replace(content[idx-2],"")
    return s.replace("\n","")+"\n",idx

def get_disease(content, idx):
    disease={}
    disease["name"] = content[idx-2] if content[idx-1] == "\n" else content[idx-1]

    disease["name"] = disease["name"].replace("〈","(") #fix typo
    disease["name"] = disease["name"].replace("《","(") #fix typo

    disease["name"] = disease["name"].replace("》",")") #fix typo
    disease["name"] = disease["name"].replace("〉",")") #fix typo

    disease["name"] = disease["name"].replace("))",")") #fix typo
    disease["name"] = disease["name"].replace("((","(") #fix typo

    disease["name"] = disease["name"].replace("（","(") #fix typo
    disease["name"] = disease["name"].replace("）",")") #fix typo

    # p1,p2 = disease["name"].find("("), disease["name"].find(")")
    # if p1 != -1:
    #     disease["name"]=[disease["name"][:p1],disease["name"][p1+1:p2]]
    # else:
    #     disease["name"] = [disease["name"].replace("\n","")]
    disease["name_exp"],idx = get_section(content,idx,"病因病机")
    disease["cause"],idx = get_section(content,idx,"诊鉴要点")
    disease["key_point"],idx = get_section(content,idx,"辨证施治")
    disease["solution"],idx = get_section(content,idx,"调摄护理")
    disease["after"],idx = get_section(content,idx,"病名释义")
    return disease,idx
import json
diseases = []
for i in range(500,len(content_out)):
    if "病名释义" in content_out[i]:
        cont,i = get_disease(content_out,i)
        diseases.append(cont)
print(f"disease found: {len(diseases)}")
with open("disease.jsonl","w") as f:
    for d in diseases:
        t = json.dumps(d,ensure_ascii=False,indent=4)
        f.write(t+"\n")

print(f"processed length: {len(content_out)}")
file_out.writelines(content_out)
file_in.close()
file_out.close()