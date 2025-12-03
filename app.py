import os
from pathlib import Path
import wave
import csv
import json
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False

# New imports for preview + spectrogram
import numpy as np
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Prefer librosa for mel spectrograms; fallback to matplotlib spectrogram
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

# Prefer pygame for play/pause control
try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# Try to enable drag & drop support if tkinterdnd2 is installed
try:
    from tkinterdnd2 import DND_FILES
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

# --- SETTINGS ---
def get_settings_path():
    appdata = os.getenv("APPDATA") or str(Path.home())
    cfg_dir = Path(appdata) / "raw2wav"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "settings.json"

DEFAULT_SETTINGS = {
    "input_folder": "",
    "output_folder": "",
    "n_channels": 1,
    "sampwidth": 2,
    "framerate": 500000,
    "window_geometry": "1000x600"
}

def load_settings():
    cfg = get_settings_path()
    if cfg.exists():
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
                # merge with defaults
                for k, v in DEFAULT_SETTINGS.items():
                    data.setdefault(k, v)
                return data
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    cfg = get_settings_path()
    try:
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print("Failed to save settings:", e)

# --- AUDIO BACKEND INIT ---
def init_audio_backend():
    global AUDIO_BACKEND
    AUDIO_BACKEND = None
    if PYGAME_AVAILABLE:
        try:
            pygame.init()
            pygame.mixer.init()
            AUDIO_BACKEND = "pygame"
        except Exception:
            AUDIO_BACKEND = None
    if AUDIO_BACKEND is None and WINSOUND_AVAILABLE:
        AUDIO_BACKEND = "winsound"
    # else no playback available
    return AUDIO_BACKEND

# --- FUNCTIONS ---
def select_input_folder():
    folder = filedialog.askdirectory()
    if folder:
        input_path_var.set(folder)
        # default output next to input if empty or was default
        if not output_path_var.get() or output_path_var.get().endswith("_wavs"):
            default_output = folder + "_wavs"
            output_path_var.set(default_output)
        save_current_settings()

def select_output_folder():
    folder = filedialog.askdirectory()
    if folder:
        output_path_var.set(folder)
        save_current_settings()

def _normalize_sampwidth(sampwidth):
    """Accept either bytes (1-4) or bits (e.g. 8,16,24,32) and return bytes (1-4)."""
    try:
        sw = int(sampwidth)
    except Exception:
        raise ValueError("Sample width must be an integer (bytes or bits).")
    if sw <= 0:
        raise ValueError("Sample width must be positive.")
    # if value looks like bits (>=8) convert to bytes
    if sw >= 8:
        if sw % 8 != 0:
            raise ValueError("If sample width is given in bits it must be a multiple of 8.")
        sw = sw // 8
    if sw not in (1, 2, 3, 4):
        raise ValueError("Sample width (bytes) must be 1, 2, 3 or 4.")
    return sw

def convert_raw_to_wav(raw_path, wav_path, n_channels=1, sampwidth=2, framerate=44100):
    # normalize/validate sampwidth here to avoid wave.Error during file close
    sampwidth_bytes = _normalize_sampwidth(sampwidth)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "rb") as raw_file:
        raw_data = raw_file.read()
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth_bytes)
        wav_file.setframerate(framerate)
        wav_file.writeframes(raw_data)

def save_conversion_log(log_file, files_info):
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Raw File", "WAV File", "Channels", "Sample Width", "Sample Rate"])
        for row in files_info:
            writer.writerow(row)

def convert_all():
    input_folder = Path(input_path_var.get())
    output_folder = Path(output_path_var.get())
    
    if not input_folder.exists():
        messagebox.showerror("Error", "Input folder does not exist.")
        return
    
    try:
        n_channels = int(n_channels_var.get())
        sampwidth = int(sampwidth_var.get())
        framerate = int(framerate_var.get())
    except ValueError:
        messagebox.showerror("Error", "Audio parameters must be valid integers.")
        return

    raw_files = list(input_folder.rglob("*.raw"))
    total_files = len(raw_files)
    
    if total_files == 0:
        messagebox.showinfo("Info", "No .raw files found in input folder.")
        return

    progress_bar["maximum"] = total_files
    progress_bar["value"] = 0

    files_info = []

    for i, raw_file in enumerate(raw_files, start=1):
        relative_path = raw_file.relative_to(input_folder)
        wav_file = output_folder / relative_path
        wav_file = wav_file.with_suffix(".wav")
        try:
            convert_raw_to_wav(raw_file, wav_file, n_channels, sampwidth, framerate)
        except Exception as e:
            messagebox.showerror("Conversion error", f"Failed to convert {raw_file}:\n{e}")
            # continue with next file
            continue
        files_info.append([str(raw_file), str(wav_file), n_channels, sampwidth, framerate])
        progress_bar["value"] = i
        root.update_idletasks()
    
    # Save log
    log_path = output_folder / "conversion_log.csv"
    save_conversion_log(log_path, files_info)

    # Save last used parameters
    save_current_settings()
    
    messagebox.showinfo("Done!", f"Converted {total_files} files.\nConversion log saved to {log_path}")

# --- AUDIO PREVIEW & SPECTROGRAM ---
_preview_temp_wav = None
_preview_playing = False
_preview_paused = False
_preview_wav_path = None
_spec_canvas = None
_spec_fig = None

def _read_wav_to_numpy(path):
    # return mono numpy array and sample rate
    if LIBROSA_AVAILABLE:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y, sr
    else:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        dtype = None
        if sw == 1:
            dtype = np.uint8  # unsigned
        elif sw == 2:
            dtype = np.int16
        elif sw == 4:
            dtype = np.int32
        else:
            # best-effort
            dtype = np.int16
        audio = np.frombuffer(frames, dtype=dtype)
        if nch > 1:
            audio = audio.reshape(-1, nch)
            audio = audio.mean(axis=1)  # mix to mono
        # if unsigned 8-bit, convert to signed centered at 0
        if sw == 1:
            audio = audio.astype(np.int16) - 128
        # normalize to float32 -1..1
        max_val = np.iinfo(audio.dtype).max if np.issubdtype(audio.dtype, np.integer) else 1.0
        audio = audio.astype(np.float32) / max_val
        return audio, sr

def _compute_log_mel(audio, sr):
    """
    Return (S_db, freqs, times)
    - S_db: 2D array (freq_bins x time_frames)
    - freqs: 1D array of frequencies in Hz corresponding to the Y axis rows
    - times: 1D array of times in seconds corresponding to the X axis columns
    """
    if LIBROSA_AVAILABLE:
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        # mel bin center frequencies in Hz
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2.0)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
        return S_db, freqs, times
    else:
        # fallback to standard spectrogram (Pxx: freq_bins x times)
        from matplotlib import mlab
        NFFT = 2048
        noverlap = NFFT // 2
        Pxx, freqs, bins = mlab.specgram(audio, NFFT=NFFT, Fs=sr, noverlap=noverlap)
        # log scale
        Pxx_db = np.log10(Pxx + 1e-10)
        times = bins  # mlab.specgram returns time bins in seconds
        return Pxx_db, freqs, times

def _draw_spectrogram(S_db, freqs, times, sr):
    global _spec_canvas, _spec_fig
    if not MATPLOTLIB_AVAILABLE:
        status_var.set("matplotlib not available: cannot show spectrogram.")
        return
    if _spec_fig is None:
        _spec_fig = Figure(figsize=(6,3), dpi=100)
    else:
        _spec_fig.clf()
    ax = _spec_fig.add_subplot(111)

    # Use pcolormesh so non-uniform mel-frequency bin positions are respected
    # S_db shape: (n_freq_bins, n_time_frames)
    # times: 1D array of time centers (s), freqs: 1D array of freq centers (Hz)
    # pcolormesh with shading='auto' will handle matching shapes.
    pcm = ax.pcolormesh(times, freqs, S_db, shading='auto', cmap='magma')
    ax.set_xlabel('Time (s)')

    # Choose Hz or kHz display depending on max frequency
    max_freq = freqs[-1] if len(freqs) > 0 else 0.0
    if max_freq >= 1000.0:
        ax.set_ylabel('Frequency (kHz)')
        # show only the min and max values on y-axis
        ax.set_yticks(np.linspace(0, max_freq, num=2))
    else:
        ax.set_ylabel('Frequency (Hz)')
        # show only the min and max values on y-axis
        ax.set_yticks(np.linspace(0, max_freq, num=2))
        # ax.set_yticklabels([f"{int(f)}" for f in np.linspace(0, max_freq)])

    ax.set_title('Log-Mel Spectrogram' if LIBROSA_AVAILABLE else 'Log Spectrogram (fallback)')
    _spec_fig.colorbar(pcm, ax=ax, format='%+2.0f dB')

    ax.set_xlim(times[0] if len(times) > 0 else 0.0, times[-1] if len(times) > 0 else 0.0)

    if _spec_canvas is None:
        _spec_canvas = FigureCanvasTkAgg(_spec_fig, master=spec_frame)
        _spec_canvas.get_tk_widget().pack(fill="both", expand=True)
    else:
        _spec_canvas.figure = _spec_fig
        _spec_canvas.draw()

def preview_raw():
    global _preview_temp_wav, _preview_playing, _preview_paused, _preview_wav_path
    # ask user for raw file
    initial_dir = input_path_var.get() or "."
    raw_file = filedialog.askopenfilename(initialdir=initial_dir, title="Select RAW file to preview",
                                          filetypes=[("RAW files", "*.raw"), ("All files", "*.*")])
    if not raw_file:
        return
    try:
        n_channels = int(n_channels_var.get())
        sampwidth = int(sampwidth_var.get())
        framerate = int(framerate_var.get())
    except ValueError:
        messagebox.showerror("Error", "Audio parameters must be valid integers.")
        return

    tmp_dir = Path(tempfile.gettempdir())
    tmp_wav = tmp_dir / (Path(raw_file).stem + "_preview.wav")
    try:
        convert_raw_to_wav(Path(raw_file), tmp_wav, n_channels, sampwidth, framerate)
    except Exception as e:
        messagebox.showerror("Conversion error", f"Could not create preview WAV:\n{e}")
        return
    _preview_temp_wav = tmp_wav
    _preview_wav_path = str(tmp_wav)
    try:
        preview_name_var.set(tmp_wav.name)   # <--- update preview filename textbox
    except Exception:
        pass

    # Stop any existing playback
    try:
        preview_stop()
    except Exception:
        pass

    # compute spectrogram and draw with proper axes
    try:
        audio, sr = _read_wav_to_numpy(tmp_wav)
        S_db, freqs, times = _compute_log_mel(audio, sr)
        _draw_spectrogram(S_db, freqs, times, sr)
    except Exception as e:
        status_var.set(f"Spectrogram error: {e}")

    status_var.set(f"Ready: {tmp_wav.name} (backend: {AUDIO_BACKEND or 'none'})")

def preview_play_pause():
    global _preview_playing, _preview_paused, _preview_wav_path
    if AUDIO_BACKEND == "pygame":
        if not _preview_playing:
            if not _preview_wav_path:
                messagebox.showinfo("Info", "No preview file. Use 'Preview RAW' to select a file.")
                return
            try:
                pygame.mixer.music.load(_preview_wav_path)
                pygame.mixer.music.play()
                _preview_playing = True
                _preview_paused = False
                status_var.set("Playing")
                play_pause_btn.config(text="Pause")
            except Exception as e:
                messagebox.showerror("Playback error", f"Could not play: {e}")
        else:
            if not _preview_paused:
                pygame.mixer.music.pause()
                _preview_paused = True
                status_var.set("Paused")
                play_pause_btn.config(text="Resume")
            else:
                pygame.mixer.music.unpause()
                _preview_paused = False
                status_var.set("Playing")
                play_pause_btn.config(text="Pause")
    elif AUDIO_BACKEND == "winsound":
        # winsound does not support pause/unpause; implement simple play/stop toggle
        if not _preview_playing:
            if not _preview_wav_path:
                messagebox.showinfo("Info", "No preview file. Use 'Preview RAW' to select a file.")
                return
            try:
                winsound.PlaySound(_preview_wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                _preview_playing = True
                play_pause_btn.config(text="Stop")
                # status_var.set("Playing (no pause support)")
            except Exception as e:
                messagebox.showerror("Playback error", f"Could not play: {e}")
        else:
            # stop
            winsound.PlaySound(None, winsound.SND_PURGE)
            _preview_playing = False
            play_pause_btn.config(text="Play")
            status_var.set("")
    else:
        messagebox.showwarning("Playback unavailable", "No playback backend available (install pygame or use Windows winsound).")

def preview_stop():
    global _preview_playing, _preview_paused, _preview_wav_path
    if AUDIO_BACKEND == "pygame":
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
    elif AUDIO_BACKEND == "winsound":
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
    _preview_playing = False
    _preview_paused = False
    status_var.set("")
    try:
        preview_name_var.set("")   # <--- clear preview filename when stopped
    except Exception:
        pass
    if 'play_pause_btn' in globals():
        play_pause_btn.config(text="Play")

# --- DRAG & DROP HANDLER ---
def on_drop(event):
    # event.data may look like '{C:/path1} {C:/path2}' or 'C:/path'
    data = event.data
    # split into tokens
    try:
        parts = root.splitlist(data)
    except Exception:
        parts = [data]
    if not parts:
        return
    first = parts[0].strip('{}')
    p = Path(first)
    if p.is_dir():
        input_path_var.set(str(p))
        if not output_path_var.get() or output_path_var.get().endswith("_wavs"):
            output_path_var.set(str(p) + "_wavs")
        save_current_settings()
    else:
        # if a file was dropped, use its parent as input folder
        parent = p.parent
        input_path_var.set(str(parent))
        if not output_path_var.get() or output_path_var.get().endswith("_wavs"):
            output_path_var.set(str(parent) + "_wavs")
        save_current_settings()

def save_current_settings():
    settings = {
        "input_folder": input_path_var.get(),
        "output_folder": output_path_var.get(),
        "n_channels": int(n_channels_var.get()) if n_channels_var.get().isdigit() else DEFAULT_SETTINGS["n_channels"],
        "sampwidth": int(sampwidth_var.get()) if sampwidth_var.get().isdigit() else DEFAULT_SETTINGS["sampwidth"],
        "framerate": int(framerate_var.get()) if framerate_var.get().isdigit() else DEFAULT_SETTINGS["framerate"],
        "window_geometry": root.geometry()
    }
    save_settings(settings)

# --- GUI ---
# Load settings
settings = load_settings()

root = tb.Window(themename="cosmo")
root.title("RAW 2 WAV")
root.geometry(settings.get("window_geometry", "1000x600"))  # restore last size/position if present
root.resizable(True, True)

# Centering all widgets in a frame
main_frame = tb.Frame(root, padding=10)
main_frame.pack(fill="both", expand=True)

# Left: controls, Right: spectrogram
controls_frame = tb.Frame(main_frame)
controls_frame.pack(side="left", fill="both", padx=(10,5), pady=10)

# Add top/bottom flexible spacers so the controls content is centered vertically
controls_top_spacer = tb.Frame(controls_frame)
controls_top_spacer.pack(fill="both", expand=True)

controls_content = tb.Frame(controls_frame)
controls_content.pack(fill="both")

controls_bottom_spacer = tb.Frame(controls_frame)
controls_bottom_spacer.pack(fill="both", expand=True)

spec_frame = tb.Frame(main_frame)
spec_frame.pack(side="right", fill="both", expand=True, padx=(5,10), pady=10)

input_path_var = tk.StringVar(value=settings.get("input_folder", ""))
output_path_var = tk.StringVar(value=settings.get("output_folder", ""))
n_channels_var = tk.StringVar(value=str(settings.get("n_channels", 1)))
sampwidth_var = tk.StringVar(value=str(settings.get("sampwidth", 2)))
framerate_var = tk.StringVar(value=str(settings.get("framerate", 44100)))
status_var = tk.StringVar(value="")
preview_name_var = tk.StringVar(value="")   # <--- new: holds selected preview filename

# Input folder
tb.Label(controls_content, text="Input Folder:", font=("Calibri", 11, "bold")).pack(pady=(5,2))
input_frame = tb.Frame(controls_content)
input_frame.pack(pady=(0,10))
input_entry = tb.Entry(input_frame, textvariable=input_path_var, width=40)
input_entry.pack(side=LEFT, padx=(0,5))
tb.Button(input_frame, text="Select", command=select_input_folder).pack(side=LEFT)

# Output folder
tb.Label(controls_content, text="Output Folder:", font=("Calibri", 11, "bold")).pack(pady=(5,2))
output_frame = tb.Frame(controls_content)
output_frame.pack(pady=(0,10))
tb.Entry(output_frame, textvariable=output_path_var, width=40).pack(side=LEFT, padx=(0,5))
tb.Button(output_frame, text="Select", command=select_output_folder).pack(side=LEFT)

# Audio parameters
tb.Label(controls_content, text="Audio Parameters:", font=("Calibri", 11, "bold")).pack(pady=(5,2))
params_frame = tb.Frame(controls_content)
params_frame.pack(pady=(0,10))
tb.Label(params_frame, text="Channels:").grid(row=0, column=0, padx=5, pady=5)
ch_entry = tb.Entry(params_frame, textvariable=n_channels_var, width=5)
ch_entry.grid(row=0, column=1, padx=5)
tb.Label(params_frame, text="Sample Width (bytes):").grid(row=0, column=2, padx=5)
sw_entry = tb.Entry(params_frame, textvariable=sampwidth_var, width=5)
sw_entry.grid(row=0, column=3, padx=5)
tb.Label(params_frame, text="Sample Rate (Hz):").grid(row=0, column=4, padx=5)
fr_entry = tb.Entry(params_frame, textvariable=framerate_var, width=10)
fr_entry.grid(row=0, column=5, padx=5)

# Buttons: Convert
buttons_frame = tb.Frame(controls_content)
buttons_frame.pack(pady=10, fill="x")
tb.Button(buttons_frame, text="Convert RAW to WAV", command=convert_all, bootstyle=SUCCESS).pack(fill="x", pady=3)

# Playback availability hint / backend init
backend_hint = AUDIO_BACKEND = init_audio_backend()

# Progress bar
progress_bar = tb.Progressbar(controls_content, orient=HORIZONTAL, length=300, mode="determinate")
progress_bar.pack(pady=(10,10))

# Status label
status_label = tb.Label(controls_content, textvariable=status_var, bootstyle="secondary")
status_label.pack(pady=(2,0))

# Spectrogram initial message if matplotlib missing
if not MATPLOTLIB_AVAILABLE:
    tb.Label(spec_frame, text="matplotlib not available.\nInstall matplotlib for spectrogram preview.", foreground="red").pack(expand=True)
else:
    # placeholder empty figure
    _spec_fig = Figure(figsize=(6,3), dpi=100)
    _spec_canvas = FigureCanvasTkAgg(_spec_fig, master=spec_frame)
    _spec_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Preview controls on the right side (select file, preview filename textbox, play/pause)
    preview_frame = tb.Frame(spec_frame)
    preview_frame.pack(fill="x", pady=(6,4))
    tb.Button(preview_frame, text="Select RAW file for preview", command=preview_raw, bootstyle=INFO).pack(fill="x", pady=(0,6))
    tb.Entry(preview_frame, textvariable=preview_name_var, state="readonly").pack(fill="x", pady=(0,6))

    play_controls = tb.Frame(spec_frame)
    play_controls.pack(pady=5)
    play_pause_btn = tb.Button(play_controls, text="Play", command=preview_play_pause, bootstyle=PRIMARY)
    play_pause_btn.pack(side=LEFT, padx=5)

# Save settings when user changes entries (simple binding)
def on_param_change(*args):
    # Debounced quick save
    save_current_settings()

input_path_var.trace_add("write", lambda *a: save_current_settings())
output_path_var.trace_add("write", lambda *a: save_current_settings())
n_channels_var.trace_add("write", lambda *a: save_current_settings())
sampwidth_var.trace_add("write", lambda *a: save_current_settings())
framerate_var.trace_add("write", lambda *a: save_current_settings())

def on_close():
    save_current_settings()
    try:
        preview_stop()
    except Exception:
        pass
    try:
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
            pygame.quit()
    except Exception:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
