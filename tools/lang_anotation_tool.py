#!/usr/bin/env python3
"""Language annotation GUI tool for NavVLA datasets."""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from PIL import Image, ImageTk

DEFAULT_DUMMY_LANGUAGE = "No language instruction"
DISPLAY_IMAGE_SIZE: tuple[int, int] = (224, 224)


class LangAnnotationTool:
    """GUI tool for annotating language instructions on navigation episodes."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.root = tk.Tk()
        self.root.title("Lang Annotation Tool")

        self.episode_dirs: list[Path] = []
        self.current_episode_idx: int = 0
        self.frame_count: int = 0
        self.prompts: list[str] = []
        self.start_photo: ImageTk.PhotoImage | None = None
        self.end_photo: ImageTk.PhotoImage | None = None
        self._loaded_lang: str = ""

        self.start_var = tk.IntVar(value=0)
        self.end_var = tk.IntVar(value=0)
        self.lang_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self.init_data()
        self.init_gui()
        self.load_episode(0)

    def init_data(self) -> None:
        self.episode_dirs = sorted(
            p for p in self.dataset_dir.iterdir()
            if p.is_dir() and (p / "traj_data.pkl").exists()
        )
        if not self.episode_dirs:
            raise RuntimeError(f"No episodes found in {self.dataset_dir}")

    def init_gui(self) -> None:
        self.init_layout()
        self.init_bindings()

    def init_layout(self) -> None:
        root = self.root

        nav_frame = tk.Frame(root)
        nav_frame.grid(row=0, column=0, columnspan=2, pady=4)
        tk.Button(nav_frame, text="◀ Prev Episode", command=self.prev_episode).pack(side=tk.LEFT, padx=4)
        self.episode_label = tk.Label(nav_frame, text="", width=30)
        self.episode_label.pack(side=tk.LEFT, padx=8)
        tk.Button(nav_frame, text="Next Episode ▶", command=self.next_episode).pack(side=tk.LEFT, padx=4)

        self.start_index_label = tk.Label(root, text="Start Index: 0")
        self.start_index_label.grid(row=1, column=0)
        self.end_index_label = tk.Label(root, text="End Index: 0")
        self.end_index_label.grid(row=1, column=1)

        self.start_panel = tk.Label(root)
        self.start_panel.grid(row=2, column=0, padx=8)
        self.end_panel = tk.Label(root)
        self.end_panel.grid(row=2, column=1, padx=8)

        self.start_scale = tk.Scale(
            root, variable=self.start_var, orient=tk.HORIZONTAL,
            from_=0, to=0, length=280, command=self.on_start_changed,
        )
        self.start_scale.grid(row=3, column=0, padx=8)
        self.end_scale = tk.Scale(
            root, variable=self.end_var, orient=tk.HORIZONTAL,
            from_=0, to=0, length=280, command=self.on_end_changed,
        )
        self.end_scale.grid(row=3, column=1, padx=8)

        self.start_lang_label = tk.Label(root, text="", fg="blue", wraplength=280, justify="left")
        self.start_lang_label.grid(row=4, column=0, padx=8, sticky="w")
        self.end_lang_label = tk.Label(root, text="", fg="blue", wraplength=280, justify="left")
        self.end_lang_label.grid(row=4, column=1, padx=8, sticky="w")

        bottom = tk.Frame(root)
        bottom.grid(row=5, column=0, columnspan=2, pady=4, sticky="ew")

        self.range_label = tk.Label(bottom, text="")
        self.range_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=8)

        tk.Label(bottom, text="Lang:").grid(row=1, column=0, padx=4)
        self.lang_entry = tk.Entry(bottom, textvariable=self.lang_var, width=50)
        self.lang_entry.grid(row=1, column=1, padx=4)
        tk.Button(bottom, text="Save", command=self.save_annotation).grid(row=1, column=2, padx=8)

        tk.Label(bottom, textvariable=self.status_var, fg="gray").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=8,
        )

    def init_bindings(self) -> None:
        self.root.bind("<Return>", lambda _: self.save_annotation())

    def load_episode(self, idx: int) -> None:
        if not (0 <= idx < len(self.episode_dirs)):
            return
        self.current_episode_idx = idx
        ep_dir = self.episode_dirs[idx]

        self.frame_count = len(sorted(ep_dir.glob("*.jpg"), key=lambda p: int(p.stem)))

        prompt_path = ep_dir / "traj_prompt.txt"
        lines = prompt_path.read_text(encoding="utf-8").splitlines() if prompt_path.exists() else []
        pad = max(0, self.frame_count - len(lines))
        self.prompts = lines + [DEFAULT_DUMMY_LANGUAGE] * pad

        self.start_scale.configure(to=self.frame_count - 1)
        self.end_scale.configure(to=self.frame_count - 1)
        self.start_var.set(0)
        self.end_var.set(self.frame_count - 1)

        initial_lang = self.prompts[0] if self.prompts else DEFAULT_DUMMY_LANGUAGE
        self.lang_var.set(initial_lang)
        self._loaded_lang = initial_lang

        n = len(self.episode_dirs)
        self.episode_label.configure(text=f"{ep_dir.name}  ({idx + 1} / {n})")
        self.root.title(f"Lang Annotation Tool — {ep_dir.name}")
        self.status_var.set(f"{ep_dir.name} loaded ({self.frame_count} frames)")
        self.update_display()

    def update_display(self) -> None:
        self.update_image_panel("start", self.start_var.get())
        self.update_image_panel("end", self.end_var.get())
        self.update_range_label()
        s, e = self.start_var.get(), self.end_var.get()
        self.start_lang_label.configure(text=self.prompts[s] if self.prompts else "")
        self.end_lang_label.configure(text=self.prompts[e] if self.prompts else "")

    def update_image_panel(self, side: str, index: int) -> None:
        ep_dir = self.episode_dirs[self.current_episode_idx]
        img = Image.open(ep_dir / f"{index}.jpg").resize(DISPLAY_IMAGE_SIZE)
        photo = ImageTk.PhotoImage(img)
        if side == "start":
            self.start_panel.configure(image=photo)
            self.start_photo = photo
            self.start_index_label.configure(text=f"Start Index: {index}")
        else:
            self.end_panel.configure(image=photo)
            self.end_photo = photo
            self.end_index_label.configure(text=f"End Index: {index}")

    def update_range_label(self) -> None:
        s, e = self.start_var.get(), self.end_var.get()
        self.range_label.configure(text=f"Range: [{s}] ─ [{e}]  ({e - s + 1} frames)")

    def on_start_changed(self, value: str) -> None:
        s = int(value)
        if s > self.end_var.get():
            self.end_var.set(s)
        self.update_display()

    def on_end_changed(self, value: str) -> None:
        e = int(value)
        if e < self.start_var.get():
            self.start_var.set(e)
        self.update_display()

    def prev_episode(self) -> None:
        if self.current_episode_idx == 0:
            self.status_var.set("Already at the first episode")
            return
        if self.lang_var.get() != self._loaded_lang:
            if not messagebox.askyesno("Unsaved changes", "Switch episode without saving?"):
                return
        self.load_episode(self.current_episode_idx - 1)

    def next_episode(self) -> None:
        if self.current_episode_idx >= len(self.episode_dirs) - 1:
            self.status_var.set("Already at the last episode")
            return
        if self.lang_var.get() != self._loaded_lang:
            if not messagebox.askyesno("Unsaved changes", "Switch episode without saving?"):
                return
        self.load_episode(self.current_episode_idx + 1)

    def save_annotation(self) -> None:
        s, e = self.start_var.get(), self.end_var.get()
        lang = self.lang_var.get().strip() or DEFAULT_DUMMY_LANGUAGE
        for i in range(s, e + 1):
            self.prompts[i] = lang
        ep_dir = self.episode_dirs[self.current_episode_idx]
        (ep_dir / "traj_prompt.txt").write_text("\n".join(self.prompts), encoding="utf-8")
        self._loaded_lang = lang
        self.status_var.set(f"Saved: {ep_dir.name} [{s}→{e}] '{lang}'")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Language annotation tool for NavVLA datasets")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    LangAnnotationTool(args.dataset_dir).run()


if __name__ == "__main__":
    main()
