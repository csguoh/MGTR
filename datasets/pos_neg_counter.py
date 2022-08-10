import json

def PosNegCounter(anno_file):
    total_pos = 0
    total_neg = 0
    annotations = [json.loads(l.strip()) for l in open(anno_file, 'r').readlines()]
    for item in annotations:
        laeo = item['laeo']
        for laeo_pair in laeo:
            if laeo_pair['interaction'] == 1:
                total_pos += 1
            else:
                total_neg += 1
    print(total_pos,total_neg)
    print(total_neg/total_pos)
    print(total_pos/(total_pos+total_neg))



if __name__ == '__main__':
    PosNegCounter('./data/ava_trainScence.json')