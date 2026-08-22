import os
import sys
import traceback


def _fail(msg: str, exit_code: int = 1) -> None:
    print("\n" + msg.strip() + "\n", file=sys.stderr)
    raise SystemExit(exit_code)


def _check_python() -> None:
    if sys.version_info < (3, 10):
        _fail(
            f"Python 3.10+ is required. You are using: {sys.version.split()[0]}\n"
            "Install a newer Python, then re-run."
        )


def _check_imports() -> None:
    missing = []
    for mod in ("cv2", "mediapipe", "numpy"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)

    if missing:
        _fail(
            "Missing required packages: " + ", ".join(missing) + "\n\n"
            "Fix:\n"
            "  python -m pip install -r requirements.txt\n"
        )

    import mediapipe as mp  # type: ignore

    if not hasattr(mp, "solutions"):
        _fail(
            "Your installed 'mediapipe' does not include `mediapipe.solutions` (FaceMesh).\n\n"
            "Quick fix:\n"
            "  python -m pip uninstall -y mediapipe\n"
            "  python -m pip install -r requirements.txt\n"
        )


def _run(camera_index: int) -> int:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        from emotion_app.main import open_camera  # type: ignore

        cap = open_camera(preferred_index=camera_index)
        cap.release()

        sys.argv = [sys.argv[0], "--camera", str(camera_index)]
        from emotion_app.main import main as app_main  # type: ignore

        app_main()
        return 0
    except SystemExit as e:
        return int(getattr(e, "code", 1) or 0)
    except Exception:
        print("Unexpected error while starting the app:\n", file=sys.stderr)
        traceback.print_exc()
        return 1


def main() -> None:
    _check_python()
    _check_imports()

    for idx in (0, 1, 2, 3):
        code = _run(idx)
        if code == 0:
            raise SystemExit(0)

    _fail(
        "Could not open any webcam (tried camera indices 0..3).\n\n"
        "Things to try:\n"
        "  - Close apps that might be using the camera (Zoom/Teams/Camera app)\n"
        "  - Check Windows Settings → Privacy & Security → Camera permissions\n"
        "  - If you have an external webcam, plug it in and retry\n"
    )


if __name__ == "__main__":
    main()