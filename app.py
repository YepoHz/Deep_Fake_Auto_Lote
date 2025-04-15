import os
import shutil
import zipfile
from PIL import Image
import gradio as gr
import uuid
import subprocess

# Crear carpetas necesarias
INPUT_DIR = "batch_images"
OUTPUT_DIR = "batch_results"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def face_swap_batch(rb_img, lote_imgs, usar_zip):
    # Limpiar carpetas anteriores
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Guardar rostro base
    rb_path = os.path.join(INPUT_DIR, "RB.jpg")
    rb_img.save(rb_path)

    # Guardar imágenes por lote
    img_paths = []
    for i, file in enumerate(lote_imgs):
        name = f"img_{i}.jpg"
        path = os.path.join(INPUT_DIR, name)
        img = Image.open(file)
        img.save(path)
        img_paths.append(path)

    # Procesar imágenes con Roop
    for path in img_paths:
        filename = os.path.basename(path)
        output_img = os.path.join(OUTPUT_DIR, f"swapped_{filename}")
        subprocess.run([
            "python", "run.py",
            "-s", rb_path,
            "-t", path,
            "-o", output_img,
            "--execution-provider", "cpu",
            "--frame-processor", "face_swapper", "face_enhancer"
        ])

    # Recolectar resultados
    resultados = []
    for path in img_paths:
        filename = os.path.basename(path)
        generado_path = os.path.join(OUTPUT_DIR, f"swapped_{filename}")
        if os.path.exists(generado_path):
            resultados.append(generado_path)

    # ZIP opcional
    if usar_zip or len(resultados) > 1:
        zip_path = f"resultados_{uuid.uuid4().hex[:6]}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in resultados:
                zipf.write(file, os.path.basename(file))
        return zip_path
    else:
        return resultados[0]

# Interfaz Gradio
gr.Interface(
    fn=face_swap_batch,
    inputs=[
        gr.Image(label="🧠 Rostro base", type="pil"),
        gr.File(label="🖼️ Imágenes por lote", file_types=["image"], file_count="multiple"),
        gr.Checkbox(label="📦 Descargar como ZIP", value=False)
    ],
    outputs=gr.File(label="⬇ Resultado procesado"),
    title="Deep Fake Lote - Yepo Hz",
    description="Sube una imagen de rostro base y varias imágenes objetivo. El sistema aplicará el cambio de rostro. Puedes elegir si descargar como ZIP o imagen directa."
).launch()
