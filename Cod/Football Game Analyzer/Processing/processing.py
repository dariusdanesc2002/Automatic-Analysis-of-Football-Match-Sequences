from ultralytics import YOLO
from ReadVideo import ReadVideo
import os

def load_model(weights_path: str, use_cuda: bool = True):
    """Încarcă modelul YOLO cu sau fără CUDA."""
    device = "cuda" if use_cuda else "cpu"
    model = YOLO(weights_path)
    model.to(device)
    return model


def load_video(uploaded_file) -> str:
    """Primește un UploadedFile de la Streamlit și-l salvează pe disc."""
    tmp_path = f"temp_videos/{uploaded_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    return tmp_path


# def apply_on_video(video_path: str, model, use_cuda: bool = True) -> str:
#     """
#     Apelează ReadVideo.process_video_with_tracker cu modelul și returnează
#     calea către fișierul rezultat.
#     """
#     rv = ReadVideo(
#         model=model,
#         use_cuda=use_cuda,
#         # poți adăuga aici parametri suplimentari (praguri, output dir etc.)
#     )
#     output_path = video_path.replace(".mp4", "_out.mp4")
#     rv.process_video_with_tracker(
#         video_path=video_path,
#         output_path=output_path
#     )
#     return output_path


DEFAULT_OUTPUT = r"C:\Users\dariu\OneDrive\Desktop\Licenta\Rezultate Videoclipuri\nu_conteaza.mp4"

def apply_on_video(video_path: str, model, output_path: str = DEFAULT_OUTPUT) -> str:
    """
    Aplică procesarea și salvează direct rezultatul la `output_path`.
    """
    # Creează folderul dacă nu există

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Instanţiem ReadVideo cu modelul tău
    rv = ReadVideo(model=model)

    # Direct folosim output_path-ul fix
    rv.process_video_with_tracker(
        video_path=video_path,
        output_path=output_path
    )

    return output_path