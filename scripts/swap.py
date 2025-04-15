import os
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer

app = None
swapper = None
enhancer = None

def setup_models(execution_provider="cpu"):
    global app, swapper, enhancer

    if app is None:
        app = FaceAnalysis(name="buffalo_l", providers=[execution_provider])
        app.prepare(ctx_id=0, det_size=(640, 640))

    if swapper is None:
        model_path = "models/inswapper_128.onnx"
        swapper = get_model(model_path, providers=[execution_provider])

    if enhancer is None:
        enhancer = GFPGANer(
            model_path="gfpgan/weights/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cuda" if execution_provider == "cuda" else "cpu"
        )

def run_swap(target_path, source_path, output_path, execution_provider="cpu", frame_processors=["face_swapper"]):
    setup_models(execution_provider)

    target_img = cv2.imread(target_path)
    source_img = cv2.imread(source_path)

    if target_img is None or source_img is None:
        print("❌ No se pudieron cargar las imágenes.")
        return False

    target_faces = app.get(target_img)
    source_faces = app.get(source_img)

    if len(target_faces) == 0 or len(source_faces) == 0:
        print("⚠️ No se detectaron rostros.")
        return False

    source_face = source_faces[0]

    for face in target_faces:
        target_img = swapper.get(target_img, face, source_face, paste_back=True)

    if "face_enhancer" in frame_processors:
        h, w = target_img.shape[:2]
        _, _, enhanced_img = enhancer.enhance(
            target_img,
            has_aligned=False,
            only_center_face=False
        )
        target_img = cv2.resize(enhanced_img, (w, h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(output_path, target_img)
    return True

