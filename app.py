import os
from pathlib import Path
import wave
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

def select_input_folder():
    folder = filedialog.askdirectory()
    if folder:
        input_path_var.set(folder)
        # Generar salida predeterminada
        default_output = folder + "_wavs"
        output_path_var.set(default_output)

def select_output_folder():
    folder = filedialog.askdirectory()
    if folder:
        output_path_var.set(folder)

def count_raw_files(folder):
    raw_files = list(Path(folder).rglob("*.raw"))
    return len(raw_files)

def convert_raw_to_wav(raw_path, wav_path, n_channels=1, sampwidth=2, framerate=44100):
    # Asegurarse de que la carpeta de salida exista
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    # Leer datos RAW
    with open(raw_path, "rb") as raw_file:
        raw_data = raw_file.read()
    # Escribir WAV
    with wave.open(str(wav_path), "wb") as wav_file:  # str() evita el error PosixPath
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(raw_data)

def convert_all():
    input_folder = Path(input_path_var.get())
    output_folder = Path(output_path_var.get())
    
    if not input_folder.exists():
        messagebox.showerror("Error", "Carpeta de entrada no válida")
        return
    
    try:
        n_channels = int(n_channels_var.get())
        sampwidth = int(sampwidth_var.get())
        framerate = int(framerate_var.get())
    except ValueError:
        messagebox.showerror("Error", "Parámetros de audio inválidos")
        return

    # Contar archivos
    raw_files = list(input_folder.rglob("*.raw"))
    total_files = len(raw_files)
    
    if total_files == 0:
        messagebox.showinfo("Info", "No se encontraron archivos .raw")
        return

    progress_bar["maximum"] = total_files
    progress_bar["value"] = 0

    for i, raw_file in enumerate(raw_files, start=1):
        relative_path = raw_file.relative_to(input_folder)
        wav_file = output_folder / relative_path
        wav_file = wav_file.with_suffix(".wav")
        convert_raw_to_wav(raw_file, wav_file, n_channels, sampwidth, framerate)
        progress_bar["value"] = i
        root.update_idletasks()
    
    messagebox.showinfo("¡Listo!", f"Se convirtieron {total_files} archivos a WAV")

# --- INTERFAZ ---
root = tb.Window(themename="cosmo")
root.title("RAW a WAV")

input_path_var = tk.StringVar()
output_path_var = tk.StringVar()

n_channels_var = tk.StringVar(value="1")
sampwidth_var = tk.StringVar(value="2")
framerate_var = tk.StringVar(value="44100")

# Carpeta de entrada
tb.Label(root, text="Carpeta de entrada:").pack(pady=(10, 0))
input_frame = tb.Frame(root)
input_frame.pack(fill=X, padx=10)
tb.Entry(input_frame, textvariable=input_path_var, width=50).pack(side=LEFT, padx=(0,5))
tb.Button(input_frame, text="Seleccionar", command=select_input_folder).pack(side=LEFT)

# Carpeta de salida
tb.Label(root, text="Carpeta de salida:").pack(pady=(10,0))
output_frame = tb.Frame(root)
output_frame.pack(fill=X, padx=10)
tb.Entry(output_frame, textvariable=output_path_var, width=50).pack(side=LEFT, padx=(0,5))
tb.Button(output_frame, text="Seleccionar", command=select_output_folder).pack(side=LEFT)

# Parámetros de audio
tb.Label(root, text="Parámetros de audio:").pack(pady=(10,0))
params_frame = tb.Frame(root)
params_frame.pack(fill=X, padx=10)
tb.Label(params_frame, text="Canales:").grid(row=0, column=0, padx=5, pady=5)
tb.Entry(params_frame, textvariable=n_channels_var, width=5).grid(row=0, column=1, padx=5)
tb.Label(params_frame, text="Bytes por muestra:").grid(row=0, column=2, padx=5)
tb.Entry(params_frame, textvariable=sampwidth_var, width=5).grid(row=0, column=3, padx=5)
tb.Label(params_frame, text="Frecuencia (Hz):").grid(row=0, column=4, padx=5)
tb.Entry(params_frame, textvariable=framerate_var, width=10).grid(row=0, column=5, padx=5)

# Botón de conversión
tb.Button(root, text="Convertir todos los RAW a WAV", command=convert_all, bootstyle=SUCCESS).pack(pady=10)

# Barra de progreso
progress_bar = tb.Progressbar(root, orient=HORIZONTAL, length=400, mode="determinate")
progress_bar.pack(pady=(0,20))

root.mainloop()
