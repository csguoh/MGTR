import os
import json

def mutualgazestatics(annotation_file="./data/timeEval/four_peo_uco.json"):
    cnt = 0
    laeo_base = [json.loads(l.strip()) for l in open(annotation_file, 'r').readlines()]
    for item in laeo_base:
        cnt += len(item['laeo'])
    print('The num of laeo instance is {}'.format(cnt))


def mutualgazeNumClassifier(mode='ava'):
    if mode=='ava':
        annotation_file = "./data/ava_testScence.json"
    else:
        annotation_file = "./data/uco_testScence.json"
    two_person = []
    three_person = []
    four_person = []
    laeo_base = [json.loads(l.strip()) for l in open(annotation_file, 'r').readlines()]
    for item in laeo_base:
        num_person = len(item['gt_bboxes'])
        if num_person == 2:
            two_person.append(item)
        elif num_person==3:
            three_person.append(item)
        elif num_person >= 4:
            four_person.append(item)
    save_path = './data/timeEval'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_one = os.path.join(save_path,f'two_peo_{mode}.json')
    save_path_two=os.path.join(save_path,f'three_peo_{mode}.json')
    save_path_three = os.path.join(save_path,f'four_peo_{mode}.json')
    with open(save_path_one, 'w') as json_file:
        for each_anno in two_person:
            json_file.write(json.dumps(each_anno) + '\n')
    with open(save_path_two, 'w') as json_file:
        for each_anno in three_person:
            json_file.write(json.dumps(each_anno) + '\n')
    with open(save_path_three, 'w') as json_file:
        for each_anno in four_person:
            json_file.write(json.dumps(each_anno) + '\n')



if __name__ == '__main__':
    mutualgazestatics()
   # mutualgazeNumClassifier(mode='uco')

