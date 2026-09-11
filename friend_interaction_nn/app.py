from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
from identity_validation import IDENTITY_QUESTION  # noqa: E402
IDENTITY_ANSWER_LABELS = {
    "Same person": "same_person",
    "Different people": "different_people",
    "Multiple-person dataset": "multiple_person_dataset",
    "Not sure": "not_sure",
}


class RelationshipApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CDR/IPDR Friend-or-Interaction Neural Network")
        self.geometry("980x760")
        self.minsize(820, 650)
        self.active_process: subprocess.Popen[str] | None = None
        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#4b5563")

    def _default_file(self, name: str) -> str:
        candidate = PARENT_DIR / name
        return str(candidate) if candidate.exists() else ""

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Relationship Neural Network", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Train from your positive and negative base-truth pairs, then score new pairs. "
                "All processing stays on this computer."
            ),
            style="Hint.TLabel",
            wraplength=900,
        ).pack(anchor="w", pady=(3, 12))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=False)
        self.train_tab = ttk.Frame(notebook, padding=12)
        self.predict_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.train_tab, text="1. Train model")
        notebook.add(self.predict_tab, text="2. Predict pairs")
        self._build_train_tab()
        self._build_predict_tab()

        ttk.Label(outer, text="Run log").pack(anchor="w", pady=(12, 4))
        self.log = ScrolledText(outer, height=15, font=("Consolas", 9), state="disabled")
        self.log.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status, style="Hint.TLabel").pack(anchor="w", pady=(5, 0))

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        folder: bool = False,
        save: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)

        def browse() -> None:
            if folder:
                selected = filedialog.askdirectory(initialdir=variable.get() or str(PROJECT_DIR))
            elif save:
                selected = filedialog.asksaveasfilename(
                    initialdir=str(Path(variable.get()).parent) if variable.get() else str(PROJECT_DIR),
                    initialfile=Path(variable.get()).name if variable.get() else "predictions.csv",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                )
            else:
                selected = filedialog.askopenfilename(
                    initialdir=str(Path(variable.get()).parent) if variable.get() else str(PARENT_DIR),
                    filetypes=[("Supported files", "*.csv *.pt"), ("All files", "*.*")],
                )
            if selected:
                variable.set(selected)

        ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _build_train_tab(self) -> None:
        tab = self.train_tab
        tab.columnconfigure(1, weight=1)
        self.train_cdr = tk.StringVar(value=self._default_file("call_logs_cdr_schema_clean.csv"))
        self.train_ipdr = tk.StringVar(value=self._default_file("internet_sessions_ipdr_schema_clean.csv"))
        self.train_labels = tk.StringVar()
        self.train_subscribers = tk.StringVar(value=self._default_file("subscribers.csv"))
        self.train_output = tk.StringVar(value=str(PROJECT_DIR / "trained_model"))
        self.train_epochs = tk.StringVar(value="250")
        self.train_device = tk.StringVar(value="auto")
        self.train_identity_answer = tk.StringVar(value="")
        self.train_identity_confirmation = tk.BooleanVar(value=False)
        self._path_row(tab, 0, "CDR CSV", self.train_cdr)
        self._path_row(tab, 1, "IPDR CSV", self.train_ipdr)
        self._path_row(tab, 2, "Base-truth CSV", self.train_labels)
        self._path_row(tab, 3, "Subscribers map (optional)", self.train_subscribers)
        self._path_row(tab, 4, "Output folder", self.train_output, folder=True)
        ttk.Label(tab, text=IDENTITY_QUESTION, wraplength=300).grid(row=5, column=0, sticky="w", pady=4)
        ttk.Combobox(
            tab,
            textvariable=self.train_identity_answer,
            values=("", *IDENTITY_ANSWER_LABELS.keys()),
            state="readonly",
            width=28,
        ).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Checkbutton(
            tab,
            text="I reviewed the evidence (not an override)",
            variable=self.train_identity_confirmation,
        ).grid(row=5, column=2, sticky="w", padx=(8, 0), pady=4)
        ttk.Label(
            tab,
            text="This determines how calling and internet activity are combined. Identifier agreement supports matching; it does not prove personal identity.",
            style="Hint.TLabel",
            wraplength=760,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 6))
        ttk.Label(tab, text="Maximum epochs").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=1, to=5000, textvariable=self.train_epochs, width=10).grid(
            row=7, column=1, sticky="w", pady=4
        )
        ttk.Label(tab, text="Device").grid(row=8, column=0, sticky="w", pady=4)
        ttk.Combobox(
            tab,
            textvariable=self.train_device,
            values=("auto", "cpu", "cuda"),
            state="readonly",
            width=10,
        ).grid(row=8, column=1, sticky="w", pady=4)
        ttk.Label(
            tab,
            text=(
                "Base truth must contain person_a, person_b, and label. Include both positive (1) "
                "and negative (0) examples; at least 10 are required and hundreds are recommended."
            ),
            style="Hint.TLabel",
            wraplength=760,
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 8))
        self.train_button = ttk.Button(tab, text="Train neural network", command=self._start_training)
        self.train_button.grid(row=10, column=0, columnspan=3, sticky="w")

    def _build_predict_tab(self) -> None:
        tab = self.predict_tab
        tab.columnconfigure(1, weight=1)
        self.predict_model = tk.StringVar(value=str(PROJECT_DIR / "trained_model" / "model.pt"))
        self.predict_cdr = tk.StringVar(value=self._default_file("call_logs_cdr_schema_clean.csv"))
        self.predict_ipdr = tk.StringVar(value=self._default_file("internet_sessions_ipdr_schema_clean.csv"))
        self.predict_subscribers = tk.StringVar(value=self._default_file("subscribers.csv"))
        self.predict_a = tk.StringVar()
        self.predict_b = tk.StringVar()
        self.predict_output = tk.StringVar(value=str(PROJECT_DIR / "trained_model" / "prediction.csv"))
        self.predict_identity_answer = tk.StringVar(value="")
        self.predict_identity_confirmation = tk.BooleanVar(value=False)
        self._path_row(tab, 0, "Trained model", self.predict_model)
        self._path_row(tab, 1, "CDR CSV", self.predict_cdr)
        self._path_row(tab, 2, "IPDR CSV", self.predict_ipdr)
        self._path_row(tab, 3, "Subscribers map (optional)", self.predict_subscribers)
        ttk.Label(tab, text="Person A identifier").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.predict_a).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(tab, text="Person B identifier").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.predict_b).grid(row=5, column=1, sticky="ew", pady=4)
        self._path_row(tab, 6, "Prediction CSV", self.predict_output, save=True)
        ttk.Label(tab, text=IDENTITY_QUESTION, wraplength=300).grid(row=7, column=0, sticky="w", pady=4)
        ttk.Combobox(
            tab,
            textvariable=self.predict_identity_answer,
            values=("", *IDENTITY_ANSWER_LABELS.keys()),
            state="readonly",
            width=28,
        ).grid(row=7, column=1, sticky="w", pady=4)
        ttk.Checkbutton(
            tab,
            text="I reviewed the evidence (not an override)",
            variable=self.predict_identity_confirmation,
        ).grid(row=7, column=2, sticky="w", padx=(8, 0), pady=4)
        ttk.Label(
            tab,
            text="This determines how calling and internet activity are combined. Identifier agreement supports matching; it does not prove personal identity.",
            style="Hint.TLabel",
            wraplength=760,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(2, 6))
        ttk.Label(
            tab,
            text="Use the same identifier type and data window used for training whenever possible.",
            style="Hint.TLabel",
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 8))
        self.predict_button = ttk.Button(tab, text="Score this pair", command=self._start_prediction)
        self.predict_button.grid(row=10, column=0, columnspan=3, sticky="w")

    def _require_paths(self, values: list[tuple[str, str]]) -> bool:
        missing = [name for name, value in values if not value.strip()]
        if missing:
            messagebox.showerror("Missing input", "Choose or enter: " + ", ".join(missing))
            return False
        return True

    def _start_training(self) -> None:
        required = [
            ("CDR CSV", self.train_cdr.get()),
            ("IPDR CSV", self.train_ipdr.get()),
            ("base-truth CSV", self.train_labels.get()),
            ("output folder", self.train_output.get()),
        ]
        if not self._require_paths(required):
            return
        identity_answer = IDENTITY_ANSWER_LABELS.get(self.train_identity_answer.get())
        if identity_answer is None:
            messagebox.showerror(
                "Identity review required",
                "Choose Same person, Different people, Multiple-person dataset, or Not sure before training.",
            )
            return
        if not self.train_identity_confirmation.get():
            messagebox.showerror("Identity review required", "Review the detected evidence and check the acknowledgement box before training.")
            return
        command = [
            sys.executable,
            str(PROJECT_DIR / "train_model.py"),
            "--cdr",
            self.train_cdr.get(),
            "--ipdr",
            self.train_ipdr.get(),
            "--base-truth",
            self.train_labels.get(),
            "--output-dir",
            self.train_output.get(),
            "--epochs",
            self.train_epochs.get(),
            "--device",
            self.train_device.get(),
            "--identity-answer",
            identity_answer,
        ]
        if self.train_identity_confirmation.get():
            command.append("--identity-confirmation")
        if self.train_subscribers.get().strip():
            command.extend(["--subscribers", self.train_subscribers.get()])
        self._run_command(command, self.train_button, "Training")

    def _start_prediction(self) -> None:
        required = [
            ("trained model", self.predict_model.get()),
            ("CDR CSV", self.predict_cdr.get()),
            ("IPDR CSV", self.predict_ipdr.get()),
            ("Person A", self.predict_a.get()),
            ("Person B", self.predict_b.get()),
            ("prediction CSV", self.predict_output.get()),
        ]
        if not self._require_paths(required):
            return
        identity_answer = IDENTITY_ANSWER_LABELS.get(self.predict_identity_answer.get())
        if identity_answer is None:
            messagebox.showerror(
                "Identity review required",
                "Choose Same person, Different people, Multiple-person dataset, or Not sure before prediction.",
            )
            return
        if not self.predict_identity_confirmation.get():
            messagebox.showerror("Identity review required", "Review the detected evidence and check the acknowledgement box before prediction.")
            return
        command = [
            sys.executable,
            str(PROJECT_DIR / "predict_pairs.py"),
            "--model",
            self.predict_model.get(),
            "--cdr",
            self.predict_cdr.get(),
            "--ipdr",
            self.predict_ipdr.get(),
            "--person-a",
            self.predict_a.get(),
            "--person-b",
            self.predict_b.get(),
            "--output",
            self.predict_output.get(),
            "--identity-answer",
            identity_answer,
        ]
        if self.predict_identity_confirmation.get():
            command.append("--identity-confirmation")
        if self.predict_subscribers.get().strip():
            command.extend(["--subscribers", self.predict_subscribers.get()])
        self._run_command(command, self.predict_button, "Predicting")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run_command(self, command: list[str], button: ttk.Button, activity: str) -> None:
        if self.active_process is not None:
            messagebox.showinfo("Already running", "Wait for the current operation to finish.")
            return
        button.configure(state="disabled")
        self.status.set(activity + "...")
        self._append_log("\n$ " + subprocess.list2cmdline(command) + "\n")

        def worker() -> None:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                self.active_process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creation_flags,
                )
                assert self.active_process.stdout is not None
                for line in self.active_process.stdout:
                    self.after(0, self._append_log, line)
                return_code = self.active_process.wait()
                final_status = "Finished successfully" if return_code == 0 else f"Failed (exit {return_code})"
            except Exception as error:  # GUI boundary: show unexpected launch errors.
                self.after(0, self._append_log, f"ERROR: {error}\n")
                final_status = "Failed"
            finally:
                self.active_process = None
                self.after(0, button.configure, {"state": "normal"})
                self.after(0, self.status.set, final_status)

        threading.Thread(target=worker, daemon=True).start()


def main() -> int:
    app = RelationshipApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
