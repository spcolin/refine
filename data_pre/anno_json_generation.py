"""
generate the json file used for dataloader.
The annotation json file is of following format:
[{'rgb_path':'the path to the rgb image','depth_path':'the path to the depth map img'},{},{},{},.......]
"""
import os,json

def json_generate(data_path):
    """
    generate annotation json file
    :param data_path: the path to NYU depth V2 raw dataset.The structure of this path must follows:
            data_path:
                scene1:
                    rgb:
                        1.png
                        2.png
                        ...
                    depth:
                        1.png
                        2.png
                        ...
                scene2:
                scene3:
    :return:generated list of jsons
    """
    json_list=[]
    scene_list=os.listdir(data_path)
    for i in scene_list:
        scene_path=data_path+i
        rgb_path=scene_path+"/rgb"
        depth_path=scene_path+"/depth"

        rgb_list=os.listdir(rgb_path)
        depth_list=os.listdir(depth_path)

        # rgb_list.sort()
        # depth_list.sort()
        
        for j in range(len(rgb_list)):
            if rgb_list[j] in depth_list:
                json_list.append({"rgb_path":rgb_path+"/"+rgb_list[j],"depth_path":depth_path+"/"+rgb_list[j]})
            
    
    return json_list


path="E:/p_nyu2/"
a=json_generate(path)

print(len(a))

saved_path="E:/anno/train_annotations.json"
f=open(saved_path,'w')

json.dump(a,f)

f.close()





