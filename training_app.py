import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset" / "thai_handwriting_56_60"
DEFAULT_OUTPUT = ROOT / "models" / "thai_handwriting_56_60_torch"
TRAIN_SCRIPT = ROOT / "scripts" / "train_model_torch.py"
EXTRACT_SCRIPT = ROOT / "scripts" / "extract_dataset.py"


class TrainingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thai Handwriting Model Trainer")
        self.geometry("980x720")
        self.minsize(860, 620)

        self.process = None
        self.log_queue = queue.Queue()

        self.json_var = tk.StringVar(value=str(Path.home() / "Downloads" / "thai_handwriting_56_60_dataset.json"))
        self.convert_output_var = tk.StringVar(value=str(DEFAULT_DATASET))
        self.dataset_var = tk.StringVar(value=str(DEFAULT_DATASET))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.epochs_var = tk.StringVar(value="25")
        self.batch_var = tk.StringVar(value="32")
        self.image_size_var = tk.StringVar(value="96")
        self.status_var = tk.StringVar(value="Ready")
        self.gpu_var = tk.StringVar(value="Checking GPU...")

        self._build_ui()
        self._check_gpu()
        self._load_existing_metrics()
        self.after(120, self._drain_log_queue)

    def _build_ui(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        title = ttk.Label(root, text="Thai Handwriting Model Trainer", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            root,
            text="Train a PyTorch CNN for labels 56, 57, 58, 59, 60 and save metrics/model files.",
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        convert_tab = ttk.Frame(notebook, padding=12)
        train_tab = ttk.Frame(notebook, padding=12)
        notebook.add(convert_tab, text="Convert Dataset")
        notebook.add(train_tab, text="Train Model")

        convert_settings = ttk.LabelFrame(convert_tab, text="JSON to Image Dataset", padding=12)
        convert_settings.pack(fill="x")
        convert_settings.columnconfigure(1, weight=1)

        self._path_row(convert_settings, 0, "Dataset JSON", self.json_var, self._choose_json)
        self._path_row(convert_settings, 1, "Output dataset folder", self.convert_output_var, self._choose_convert_output)

        convert_actions = ttk.Frame(convert_tab)
        convert_actions.pack(fill="x", pady=14)
        ttk.Button(convert_actions, text="Convert JSON", command=self._convert_dataset).pack(side="left")
        ttk.Button(convert_actions, text="Use Output for Training", command=self._use_converted_dataset).pack(side="left", padx=8)
        ttk.Button(convert_actions, text="Open Dataset Folder", command=self._open_dataset_folder).pack(side="left", padx=8)

        convert_info = ttk.LabelFrame(convert_tab, text="Convert Log", padding=10)
        convert_info.pack(fill="both", expand=True)
        self.convert_log_text = tk.Text(convert_info, height=12, wrap="word")
        self.convert_log_text.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(train_tab, text="Training Settings", padding=12)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)

        self._path_row(settings, 0, "Dataset folder", self.dataset_var, self._choose_dataset)
        self._path_row(settings, 1, "Output folder", self.output_var, self._choose_output)

        ttk.Label(settings, text="Epochs").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(settings, textvariable=self.epochs_var, width=12).grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(settings, text="Batch size").grid(row=2, column=1, sticky="w", padx=(140, 0), pady=6)
        ttk.Entry(settings, textvariable=self.batch_var, width=12).grid(row=2, column=1, sticky="w", padx=(220, 0), pady=6)

        ttk.Label(settings, text="Image size").grid(row=2, column=1, sticky="w", padx=(360, 0), pady=6)
        ttk.Entry(settings, textvariable=self.image_size_var, width=12).grid(row=2, column=1, sticky="w", padx=(440, 0), pady=6)

        gpu_label = ttk.Label(settings, textvariable=self.gpu_var)
        gpu_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        actions = ttk.Frame(train_tab)
        actions.pack(fill="x", pady=14)

        self.start_button = ttk.Button(actions, text="Start Training", command=self._start_training)
        self.start_button.pack(side="left")

        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop_training, state="disabled")
        self.stop_button.pack(side="left", padx=8)

        ttk.Button(actions, text="Refresh Metrics", command=self._load_existing_metrics).pack(side="left", padx=8)
        ttk.Button(actions, text="Open Output Folder", command=self._open_output_folder).pack(side="left", padx=8)

        status = ttk.Label(actions, textvariable=self.status_var)
        status.pack(side="right")

        panes = ttk.PanedWindow(train_tab, orient="vertical")
        panes.pack(fill="both", expand=True)

        report_frame = ttk.LabelFrame(panes, text="Metrics", padding=10)
        panes.add(report_frame, weight=1)

        self.metrics_text = tk.Text(report_frame, height=10, wrap="word")
        self.metrics_text.pack(fill="both", expand=True)
        self.metrics_text.configure(state="disabled")

        log_frame = ttk.LabelFrame(panes, text="Training Log", padding=10)
        panes.add(log_frame, weight=2)

        log_wrap = ttk.Frame(log_frame)
        log_wrap.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_wrap, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_wrap, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", pady=6)

    def _choose_dataset(self):
        path = filedialog.askdirectory(initialdir=self.dataset_var.get() or str(ROOT))
        if path:
            self.dataset_var.set(path)

    def _choose_json(self):
        path = filedialog.askopenfilename(
            initialdir=str(Path.home() / "Downloads"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.json_var.set(path)

    def _choose_convert_output(self):
        path = filedialog.askdirectory(initialdir=self.convert_output_var.get() or str(ROOT))
        if path:
            self.convert_output_var.set(path)

    def _choose_output(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get() or str(ROOT))
        if path:
            self.output_var.set(path)

    def _convert_dataset(self):
        json_path = Path(self.json_var.get())
        output_path = Path(self.convert_output_var.get())

        if not json_path.exists():
            messagebox.showerror("JSON not found", f"Dataset JSON does not exist:\n{json_path}")
            return

        output_path.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            str(json_path),
            "--out",
            str(output_path),
        ]

        self.convert_log_text.delete("1.0", "end")
        self.convert_log_text.insert("end", "Running command:\n" + " ".join(command) + "\n\n")

        try:
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as error:
            self.convert_log_text.insert("end", f"Convert failed: {error}\n")
            messagebox.showerror("Convert failed", str(error))
            return

        if result.stdout:
            self.convert_log_text.insert("end", result.stdout)
        if result.stderr:
            self.convert_log_text.insert("end", "\nErrors:\n" + result.stderr)

        if result.returncode == 0:
            self.dataset_var.set(str(output_path))
            self.status_var.set("Dataset converted")
            messagebox.showinfo("Convert complete", f"Dataset extracted to:\n{output_path}")
        else:
            messagebox.showerror("Convert failed", f"Extractor exited with code {result.returncode}")

    def _use_converted_dataset(self):
        self.dataset_var.set(self.convert_output_var.get())
        messagebox.showinfo("Dataset selected", "Converted dataset folder is now selected for training.")

    def _check_gpu(self):
        def worker():
            command = [
                sys.executable,
                "-c",
                "import torch; "
                "print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')",
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines and lines[0] == "True":
                    self.log_queue.put(("gpu", f"GPU ready: {lines[1]}"))
                else:
                    self.log_queue.put(("gpu", "GPU not available. Training will use CPU."))
            except Exception as error:
                self.log_queue.put(("gpu", f"GPU check failed: {error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _validate_settings(self):
        dataset = Path(self.dataset_var.get())
        output = Path(self.output_var.get())

        if not dataset.exists():
            messagebox.showerror("Dataset not found", f"Dataset folder does not exist:\n{dataset}")
            return None

        for label in ["56", "57", "58", "59", "60"]:
            class_dir = dataset / label
            if not class_dir.exists() or not list(class_dir.glob("*.png")):
                messagebox.showerror("Missing class folder", f"Missing PNG images in:\n{class_dir}")
                return None

        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            image_size = int(self.image_size_var.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "Epochs, batch size, and image size must be numbers.")
            return None

        if epochs < 1 or batch_size < 1 or image_size < 16:
            messagebox.showerror("Invalid settings", "Use epochs >= 1, batch size >= 1, image size >= 16.")
            return None

        return dataset, output, epochs, batch_size, image_size

    def _start_training(self):
        settings = self._validate_settings()
        if settings is None:
            return

        dataset, output, epochs, batch_size, image_size = settings
        output.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-u",
            str(TRAIN_SCRIPT),
            "--data",
            str(dataset),
            "--out",
            str(output),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(image_size),
        ]

        self._clear_log()
        self._append_log("Running command:\n" + " ".join(command) + "\n\n")
        self.status_var.set("Training...")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self):
        try:
            for line in self.process.stdout:
                self.log_queue.put(("log", line))

            return_code = self.process.wait()
            self.log_queue.put(("done", return_code))
        except Exception as error:
            self.log_queue.put(("error", str(error)))

    def _stop_training(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status_var.set("Stopping...")
            self._append_log("\nStop requested.\n")

    def _drain_log_queue(self):
        while True:
            try:
                kind, value = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(value)
            elif kind == "gpu":
                self.gpu_var.set(value)
            elif kind == "done":
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.process = None
                if value == 0:
                    self.status_var.set("Training complete")
                    self._append_log("\nTraining complete.\n")
                    self._load_existing_metrics()
                else:
                    self.status_var.set(f"Training stopped or failed: exit {value}")
                    self._append_log(f"\nProcess ended with exit code {value}.\n")
            elif kind == "error":
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status_var.set("Error")
                self._append_log(f"\nError: {value}\n")

        self.after(120, self._drain_log_queue)

    def _load_existing_metrics(self):
        output = Path(self.output_var.get())
        metrics_path = output / "metrics.json"
        report_path = output / "classification_report.txt"
        summary_path = output / "training_summary.txt"

        lines = []
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                lines.append(f"Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
                lines.append(f"Macro precision: {metrics.get('macro_precision', 0):.4f}")
                lines.append(f"Macro recall: {metrics.get('macro_recall', 0):.4f}")
                lines.append(f"Macro F1-score: {metrics.get('macro_f1', 0):.4f}")
                lines.append(f"Weighted precision: {metrics.get('weighted_precision', 0):.4f}")
                lines.append(f"Weighted recall: {metrics.get('weighted_recall', 0):.4f}")
                lines.append(f"Weighted F1-score: {metrics.get('weighted_f1', 0):.4f}")
                lines.append(f"Train samples: {metrics.get('train_count', '-')}")
                lines.append(f"Validation samples: {metrics.get('validation_count', '-')}")
                lines.append(f"Test samples: {metrics.get('test_count', '-')}")
                matrix = metrics.get("confusion_matrix")
                if matrix:
                    lines.append("")
                    lines.append("Confusion matrix:")
                    lines.extend(" ".join(f"{value:>3}" for value in row) for row in matrix)
                lines.append("")
                lines.append(f"Model folder: {output}")
            except Exception as error:
                lines.append(f"Could not read metrics.json: {error}")
        else:
            lines.append("No metrics.json found yet. Train a model first.")

        if summary_path.exists():
            lines.append("")
            lines.append(f"Summary file: {summary_path}")

        if report_path.exists():
            lines.append("")
            lines.append("Classification report:")
            lines.append(report_path.read_text(encoding="utf-8"))

        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "\n".join(lines))
        self.metrics_text.configure(state="disabled")

    def _open_output_folder(self):
        output = Path(self.output_var.get())
        output.mkdir(parents=True, exist_ok=True)
        os.startfile(output)

    def _open_dataset_folder(self):
        output = Path(self.convert_output_var.get())
        output.mkdir(parents=True, exist_ok=True)
        os.startfile(output)

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _append_log(self, text):
        self.log_text.insert("end", text)
        self.log_text.see("end")


if __name__ == "__main__":
    app = TrainingApp()
    app.mainloop()
