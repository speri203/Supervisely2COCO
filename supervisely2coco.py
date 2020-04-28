## 
# Author: Caio Marcellos
# Modifier: Sai Peri
# Email: caiocuritiba@gmail.com
##
import os
import numpy as np
import json
import glob
from datetime import datetime
from pathlib import Path 
import argparse
import sys
from skimage import draw

"""
Converting from suvervisely to COCO Format (only detection (bbox) tested in this version)

Example of Usage from commandline:
`py supervisely2coco.py meta.json './ds/ann/'  formatted2.json `
"""


def convert_supervisely_to_coco(meta_path,
        ann_base_dir = './ds/ann/', save_as=None,
        only_img_name=False
    ):
    """
    - ann_base_dir: directory for annotation files
        - Annotation files are expected to be <image-filename>.json
    - save_as: if defined (not None) is a path to save the COCO generated json format
    - bbox outputted as BoxMode.XYWH_ABS

    TODO: 
    - tags: e.g train, val
    """ 

    ann_fnames, ann_jsons = get_all_ann_file(ann_base_dir)
    map_category = get_categories_from_meta(meta_path)

    catg_repr = [{
            "id": v,
            "name": k,
            "supercategory": "type"
    } for k,v in map_category.items()]

    out_cnv_imgs = [
        convert_single_image(id_img, ann_fnames[id_img], ann_jsons[id_img], 
            map_category, ann_base_dir, only_img_name)
        for id_img in range(len(ann_fnames))
    ]
    images_repr = [o[0] for o in out_cnv_imgs]
    ann_repr = [o[1] for o in out_cnv_imgs]

    # Flatten annotation (len(images) to len(all-annotations))
    ann_repr_flatten = [inner for lst in ann_repr for inner in lst]

    # Adjust Annotations ID:
    for i, ann in enumerate(ann_repr_flatten):
        ann['id'] = i

    coco_fmt = {
        "info": {
            "year": datetime.now().strftime('%Y'),
            "version": "1",
            "description": "",
            "contributor": "converted from supervisely2coco - caiofcm",
            "url": "",
            "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        },
        "images": images_repr,
        "annotations": ann_repr_flatten,
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": catg_repr
    }

    if save_as:
        with open(save_as, 'w') as fp:
            print("Dumping JSON file")
            json.dump(coco_fmt, fp, cls=NpEncoder)
    return coco_fmt

class NpEncoder(json.JSONEncoder):
    def default(self, obj): #pylint: disable=method-hidden
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
    

def convert_single_image(idimg, fname_img, json_suprv, map_category, imgs_base_dir, only_img_name=False, start_annotation_id=0):
    # output in mode BoxMode.XYWH_ABS
    
    image_base = {
            "id": idimg,
            "width": json_suprv['size']['width'],
            "height": json_suprv['size']['height'],
            "file_name": fname_img if not only_img_name else Path(fname_img).name,
            "license": 1,
            "date_captured": ""
        }

    objects = [obj for obj in json_suprv['objects'] if obj['classTitle'] != 'bg']

    obj_exteriors = [
        np.array(obj['points']['exterior'])
        for obj in objects
    ] 

    bboxes = [
        [
            extr.min(axis=0)[0],
            extr.min(axis=0)[1],
            extr.max(axis=0)[0] - extr.min(axis=0)[0],
            extr.max(axis=0)[1] - extr.min(axis=0)[1],
        ]
        for extr in obj_exteriors
    ]

    ann = [
        {
            "id": start_annotation_id + i,
            "image_id": idimg,
            "segmentation": [extr],
            "area": bbox[2]*bbox[3],
            "bbox": bbox,
            "category_id": map_category[obj['classTitle']],
            "iscrowd": 0
        }
        for i, (obj, bbox, extr) in enumerate(zip(objects, bboxes, obj_exteriors))
    ]
    #print(ann)

    return image_base, ann

def get_all_ann_file(base_dir):
    all_ann_files = glob.glob(os.path.join(base_dir, "*.json"))
    all_fname_img = [fname[:-5] for fname in all_ann_files]
    all_json_ann = []
    for json_path in all_ann_files:
        with open(json_path) as fs:
            json_suprv = json.load(fs)
        all_json_ann += [json_suprv]
    return all_fname_img, all_json_ann

def get_categories_from_meta(meta_json_path):
    with open(meta_json_path) as fs:
        json_meta = json.load(fs)
    
    classes = [clss['title'] for clss in json_meta['classes'] if clss['title'] != 'bg']
    mapCategories = {c: i for i, c in enumerate(classes)}
    return mapCategories



###Test
def case_dev():
    coco_fmt = convert_supervisely_to_coco('./meta.json', save_as='formatted_coco.json', only_img_name=True)
    pass

def main():
    parser = argparse.ArgumentParser(description="""
    Supervisely2Coco:
    Converting from suvervisely to COCO Format (only detection (bbox) tested in this version)
    Example of Usage from commandline:
        `py supervisely2coco.py meta.json './ds/ann/'  formatted2.json `
    """)
    parser.add_argument(
        "-v",
        "--version",
        help="display version information",
        action="version",
        version="Supervisely2Coco {}, Python {}".format('0.0.1', sys.version),
    )
    parser.add_argument("meta", type=str, help="Meta JSON File")
    parser.add_argument("ann_base_dir", type=str, help="Annotations base directory (usually downloaded in './ds/ann/' )")
    parser.add_argument("output", type=str, help="Output Coco JSON File")
    parser.add_argument('-n', '--only-image-name', action='store_true', 
                        help="Save only the image name (not the full path)")    
    args = parser.parse_args()

    meta = args.meta
    ann_base_dir = args.ann_base_dir
    save_as = args.output
    flag_only_name = args.only_image_name
    print('Converting from meta={}; annotations in [{}] to output={}'.format(meta, ann_base_dir, save_as))
    coco_fmt = convert_supervisely_to_coco(meta, ann_base_dir=ann_base_dir, save_as=save_as, only_img_name=flag_only_name)

    print('Done.')


if __name__ == "__main__":

    main()

    pass
