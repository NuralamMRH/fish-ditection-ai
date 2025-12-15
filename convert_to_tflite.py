import os
import sys
import shutil

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

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default=None)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--method', choices=['ultralytics', 'onnx'], default='ultralytics')
    a = p.parse_args()

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
            out_path = export_ultralytics(a.weights, a.imgsz, a.out)
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
            out_path = torchscript_to_tflite(model_ts, a.imgsz, a.out)
            print(out_path)
        except Exception as e:
            print(f'Conversion failed: {e}')
            sys.exit(1)

if __name__ == '__main__':
    main()

