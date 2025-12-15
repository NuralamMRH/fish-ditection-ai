import os
import sys
import shutil
import json

def find_weights(defaults):
    for p in defaults:
        if os.path.exists(p):
            return p
    return None

def export_ultralytics(weights, imgsz, out):
    from ultralytics import YOLO
    m = YOLO(weights)
    f = m.export(format='tflite', imgsz=imgsz, nms=True)
    if out:
        dst = out if out.endswith('.tflite') else os.path.join(out, os.path.basename(f))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(f, dst)
        return dst
    return f

def torchscript_to_tflite(model_ts, imgsz, out):
    import torch
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    m = torch.jit.load(model_ts).eval()
    x = torch.randn(1, 3, imgsz, imgsz)
    onnx_path = 'model.onnx'
    torch.onnx.export(m, x, onnx_path, input_names=['images'], output_names=['predictions'], dynamic_axes={'images': {0: 'batch'}, 'predictions': {0: 'batch'}}, opset_version=12)
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved = 'tf_model'
    tf_rep.export_graph(saved)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    out_path = out if out and out.endswith('.tflite') else (out and os.path.join(out, 'model.tflite')) or 'model.tflite'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    open(out_path, 'wb').write(tflite_model)
    return out_path

def export_classification_database(database_pt, out_dir):
    import torch
    data = torch.load(database_pt)
    base = data[0].cpu().numpy()
    internal_ids = data[1]
    image_ids = data[2]
    annotation_ids = data[3]
    drawn_fish_ids = data[4]
    keys = data[5]
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, 'classification_database.npz')
    import numpy as np
    np.savez(npz_path, data_base=base, internal_ids=internal_ids, image_ids=image_ids, annotation_ids=annotation_ids, drawn_fish_ids=drawn_fish_ids)
    labels_path = os.path.join(out_dir, 'classification_labels.json')
    labels = {}
    for k, v in keys.items():
        labels[int(k)] = {'label': v.get('label'), 'species_id': v.get('species_id')}
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    return npz_path, labels_path

def export_detector_bundle(weights, imgsz, out_dir):
    tfl = export_ultralytics(weights, imgsz, None)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, 'ditector_best.tflite')
    shutil.copy2(tfl, dst)
    labels_path = None
    info_path = os.path.join('detector_v12', 'model_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        names = info.get('class_names')
        if names:
            labels_path = os.path.join(out_dir, 'detector_labels.json')
            with open(labels_path, 'w') as f:
                json.dump(names, f)
    return dst, labels_path

def export_classifier_bundle(model_ts, database_pt, imgsz, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dst = None
    try:
        tfl_path = torchscript_to_tflite(model_ts, imgsz, None)
        dst = os.path.join(out_dir, 'classification_best.tflite')
        shutil.copy2(tfl_path, dst)
    except Exception as e:
        pass
    db_npz, labels_json = export_classification_database(database_pt, out_dir)
    return dst, db_npz, labels_json

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default=None)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--method', choices=['ultralytics', 'onnx'], default='ultralytics')
    p.add_argument('--bundle', action='store_true', default=True)
    a = p.parse_args()

    out_dir = a.out or 'export_for_react_native'
    if a.method == 'ultralytics' and a.bundle:
        if not a.weights:
            a.weights = find_weights([
                'detector_v12/best.pt',
                'detector_v12/train/weights/best.pt',
                'yolov8n.pt'
            ])
        if not a.weights:
            print('No Ultralytics weights found')
            sys.exit(1)
        try:
            det_tfl, det_labels = export_detector_bundle(a.weights, a.imgsz, out_dir)
            print(det_tfl)
            if det_labels:
                print(det_labels)
        except Exception as e:
            print(f'Export failed: {e}')
            sys.exit(1)
        try:
            cls_ts = 'classification_rectangle_v7-1/model.ts'
            cls_db = 'classification_rectangle_v7-1/database.pt'
            if os.path.exists(cls_ts) and os.path.exists(cls_db):
                cls_tfl, db_npz, labels_json = export_classifier_bundle(cls_ts, cls_db, 224, out_dir)
                if cls_tfl:
                    print(cls_tfl)
                else:
                    print('classification_best.tflite skipped')
                print(db_npz)
                print(labels_json)
            else:
                print('Classification assets not found')
        except Exception as e:
            print(f'Classification export failed: {e}')
    else:
        if a.method == 'ultralytics':
            if not a.weights:
                a.weights = find_weights([
                    'detector_v12/best.pt',
                    'detector_v12/train/weights/best.pt',
                    'yolov8n.pt'
                ])
            if not a.weights:
                print('No Ultralytics weights found')
                sys.exit(1)
            try:
                out_path = export_ultralytics(a.weights, a.imgsz, out_dir)
                print(out_path)
            except Exception as e:
                print(f'Export failed: {e}')
                sys.exit(1)
        else:
            model_ts = a.weights or 'detector_v10_m3/model.ts'
            if not os.path.exists(model_ts):
                print('TorchScript model not found')
                sys.exit(1)
            try:
                out_path = torchscript_to_tflite(model_ts, a.imgsz, os.path.join(out_dir, 'model.tflite'))
                print(out_path)
            except Exception as e:
                print(f'Conversion failed: {e}')
                sys.exit(1)

if __name__ == '__main__':
    main()
