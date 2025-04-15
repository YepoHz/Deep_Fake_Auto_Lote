from scripts.swap import run_swap
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True)
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--execution-provider", default="CPUExecutionProvider")
    parser.add_argument("--frame-processor", nargs="+", default=["face_swapper"])
    args = parser.parse_args()

    run_swap(
        args.target,
        args.source,
        args.output,
        args.execution_provider,
        args.frame_processor
    )

if __name__ == "__main__":
    main()
