import os
import argparse

parser = argparse.ArgumentParser(description="script to create annotations file")
parser.add_argument('--videos_path', type=str, default=None)
args = parser.parse_args()


path_annotations = os.path.join("\\".join(args.videos_path.split("\\")[:-1]),"annotations")
print(path_annotations)
if not os.path.exists(path_annotations):
    print(path_annotations)
    os.mkdir(path_annotations)
list_class = os.listdir(args.videos_path)
print(list_class)
for i in range(1,4):
    for class_name in list_class:
        if "create" in str(class_name):
            continue 
        path_class_anno = os.path.join(path_annotations,class_name) + "_test_split{}.txt".format(i)
        print(path_class_anno)
        with open(path_class_anno,"w") as f:
            list_video = os.listdir(os.path.join(args.videos_path,class_name))
            print(class_name)
            len_lv = len(list_video)
            len_train = int(0.8*len_lv)
            for j in range(len_train):
                context_train_ano = str(list_video[j]+" "+"1\n")
                f.write(context_train_ano)
            for j in range(len_train,len_lv):
                context_test_ano = str(list_video[j]+" "+"2\n")
                f.write(context_test_ano)

