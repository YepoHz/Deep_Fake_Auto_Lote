
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2

app = None
swapper = None

def setup_models(execution_provider="cpu"):
    global app, swapper
    if app is None:
        app = FaceAnalysis(name="buffalo_l", providers=[execution_provider])
        app.prepare(ctx_id=0, det_size=(640, 640))
    if swapper is None:
        model_path = "models/inswapper_128.onnx"
        swapper = get_model(model_path, providers=[execution_provider])

def run_swap(target_path, source_path, output_path, execution_provider="cpu", frame_processors=["face_swapper"]):
    setup_models(execution_provider)

    # Leer imágenes
    target_img = cv2.imread(target_path)
    source_img = cv2.imread(source_path)

    # Verificar carga
    if target_img is None or source_img is None:
        print("❌ Error al leer las imágenes.")
        return False

    # Detectar rostros
    target_faces = app.get(target_img)
    source_faces = app.get(source_img)

    if len(target_faces) == 0 or len(source_faces) == 0:
        print("⚠️ No se detectaron rostros.")
        return False

    # Tomar el primer rostro detectado de la fuente
    source_face = source_faces[0]

    # Procesar cada rostro en la imagen objetivo
    for face in target_faces:
        target_img = swapper.get(target_img, face, source_face, paste_back=True)

    # Guardar imagen final
    cv2.imwrite(output_path, target_img)
    return True
