file_path = './img_feat.json'
import json
import copy
with open(file_path, 'wt') as outfile:
    data = {}
    data['img_feat'] = copy.copy(img_feat[0]).to('cpu').numpy().tolist()
    json.dump(data,outfile)
outfile.close()
