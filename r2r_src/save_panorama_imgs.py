import sys; sys.path.append('tasks/R2R/')  # NoQA
import os
import json
import skimage.io
# from util.geometry import gen_panorama_img
# from geometry import gen_panorama_img
from utils import gen_panorama_img

def load_viewpointids():
    viewpointIds = []
    with open('connectivity/scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open('connectivity/%s_connectivity.json' % scan) as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds


viewpointIds = load_viewpointids()
save_dir = 'tasks/R2R/data/panorama_imgs/'
os.makedirs(save_dir, exist_ok=True)

for n, (scanId, viewpointId) in enumerate(viewpointIds):
    if n % 100 == 0:
        print('processing %d / %d' % (n, len(viewpointIds)))
    im = gen_panorama_img(scanId, viewpointId)
    save_path = os.path.join(save_dir, '%s_%s.png' % (scanId, viewpointId))
    skimage.io.imsave(save_path, im)
