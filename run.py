import argparse
import os
from scripts.swap import run_swap

def parse_args():
    parser = argparse.ArgumentParser(description="Face Swap con Roop (por imagen)")
    parser.add_argument("-s", "--source", required=True, help="Ruta al rostro base (RB)")
    parser.add_argument("-t", "--target", required=True, help="Ruta a la imagen objetivo")
    parser.add_argument("-o", "--output", required=True, help="Ruta para guardar el resultado")
    parser.add_argument("--execution-provider", default="cpu", help="cpu o cuda")
    parser.add_argument("--frame-processor", nargs="+", default=["face_swapper"], help="Procesadores: face_swapper, face_enhancer")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    result = run_swap(
        target_path=args.target,
        source_path=args.source,
        output_path=args.output,
        execution_provider=args.execution_provider,
        frame_processors=args.frame_processor
    )

    if result:
        print(f"✅ Proceso exitoso: {args.target} → {args.output}")
    else:
        print(f"❌ Falló el proceso con {args.target}")

if __name__ == "__main__":
    main()
