import argparse
import os

from .inference import predict


def main():
    parser = argparse.ArgumentParser(description="Deepfake audio detector (local inference)")
    parser.add_argument("audio_path", help="Calea către fișierul audio")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(args.audio_path)

    with open(args.audio_path, "rb") as f:
        result = predict(f.read())

    print("=" * 40)
    print(f"FILE:      {os.path.basename(args.audio_path)}")
    print(f"PREDICTION: {result['prediction']}")
    print(f"CONFIDENCE: {result['confidence']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
