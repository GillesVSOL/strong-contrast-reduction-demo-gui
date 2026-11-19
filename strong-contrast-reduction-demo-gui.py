#!/usr/bin/env python3
"""Standalone contrast reduction explorer with a minimum gain (Python 3.10+).

Install: python -m pip install customtkinter matplotlib numpy
Run:     python layer_contrast_gui.py
Check:   python layer_contrast_gui.py --self-test

Plots independent detail-layer response curves, not an image decomposition.
No VSOLImage dependency. No percentile normalization. All magnitudes are raw.
Global sliders SET the corresponding parameter on every active layer.
Vertical sliders are grouped by parameter, with ALL followed by each layer.
Each layer has three parameters: unity_x, minimum_x, and min_gain.
Output slope is 1 through unity_x and min_gain from minimum_x onward, with
a cubic smoothstep transition between them. Integrating this nonnegative
slope prevents contrast reversal. min_gain ranges from 0 to 1; zero gives
saturation, not zero output. The plotted gain is output / input, whose
asymptotic value is min_gain.
"""

from dataclasses import dataclass, asdict
import argparse
import math
import sys

import numpy as np


@dataclass
class Layer:
    unity_x: float = 2.0
    minimum_x: float = 4.0
    min_gain: float = 0.0


def layer_gain(x, layer):
    """Return output/input gain by integrating a nonnegative output slope.

    t = clip((abs(x) - unity_x) / (minimum_x - unity_x), 0, 1)
    slope = min_gain + (1 - min_gain) * (1 - 3*t**2 + 2*t**3)
    Its integral has a linear tail with slope min_gain and no reversal.
    At zero, gain is defined by its limit, 1. Signed output is x * gain.
    """
    if (not all(math.isfinite(v) for v in (layer.unity_x, layer.minimum_x, layer.min_gain))
            or not 0 <= layer.unity_x < layer.minimum_x
            or not 0 <= layer.min_gain <= 1):
        raise ValueError("Require 0 <= start x < end x and 0 <= minimum gain <= 1, all finite.")
    magnitude = np.abs(np.asarray(x, dtype=np.float64))
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Detail values must be finite.")
    # Clip before division to avoid overflow for huge inputs and narrow spans.
    position = np.clip(magnitude, layer.unity_x, layer.minimum_x)
    t = (position - layer.unity_x) / (layer.minimum_x - layer.unity_x)
    # Integral of smoothstep's complement, factored to avoid large powers of x.
    retained = layer.unity_x + (position - layer.unity_x) * (1 - t**2 + 0.5 * t**3)
    ratio = np.divide(retained, magnitude, out=np.ones_like(magnitude),
                      where=magnitude > layer.unity_x)
    gain = layer.min_gain + (1 - layer.min_gain) * ratio
    return np.where(magnitude <= layer.unity_x, 1.0, gain)


def adjust_control(layer, key, value, maximum):
    """Set the floor or move one endpoint, nudging its partner if needed."""
    if key not in ("unity_x", "minimum_x", "min_gain"):
        raise ValueError("Unknown control.")
    if not math.isfinite(value) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Control and slider maximum must be finite; maximum > 0.")
    if key == "min_gain":
        return Layer(layer.unity_x, layer.minimum_x, min(max(value, 0.0), 1.0))
    gap = maximum * 1e-4
    unity, end = layer.unity_x, layer.minimum_x
    if key == "unity_x":
        unity = min(max(value, 0.0), maximum - gap)
        end = max(end, unity + gap)
    else:
        end = min(max(value, gap), maximum)
        unity = min(unity, end - gap)
    return Layer(unity, end, layer.min_gain)


def create_app():
    # Deferred imports allow --self-test without GUI packages or a display.
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.colors import LinearSegmentedColormap

    ctk.set_appearance_mode("dark")
    colors = {
        "background": "#18191c",
        "panel": "#25262b",
        "raised": "#303137",
        "plot": "#202125",
        "text": "#f5f2ee",
        "muted": "#c4c0ba",
        "border": "#606169",
        "track": "#505158",
        "accent": "#4da3ff",
        "hover": "#78b9ff",
        "on_accent": "#071a2d",
        "layer_last": "#d9d9d6",
        "reference": "#a8a8ad",
    }
    layer_palette = LinearSegmentedColormap.from_list(
        "blue_gray", [colors["accent"], colors["layer_last"]])
    scrollbar_colors = {
        "scrollbar_button_color": colors["border"],
        "scrollbar_button_hover_color": colors["muted"],
    }

    class Control(ctk.CTkFrame):
        """Vertical channel strip with a linear parameter scale."""
        def __init__(self, parent, label, value, maximum, command, color=None):
            super().__init__(parent, fg_color="transparent")
            self.command = command
            ctk.CTkLabel(self, text=label, width=33, text_color=color or colors["text"],
                         font=ctk.CTkFont(weight="bold")).grid(row=0, column=0)
            options = {"button_color": color or colors["accent"],
                       "progress_color": color or colors["accent"],
                       "button_hover_color": colors["hover"], "fg_color": colors["track"]}
            self.slider = ctk.CTkSlider(self, from_=0, to=maximum, command=self.changed,
                                       orientation="vertical", width=18, height=105, **options)
            self.slider.grid(row=1, column=0, pady=5)
            self.value_label = ctk.CTkLabel(self, text="", width=33, text_color=colors["text"])
            self.value_label.grid(row=2, column=0)
            self.set(value)

        def set(self, value):
            value = round(float(value), 2)
            self.slider.set(value)
            self.value_label.configure(text=f"{value:.2f}")

        def changed(self, value):
            value = round(float(value), 2)
            self.slider.set(value)
            self.value_label.configure(text=f"{value:.2f}")
            self.command(value)

    class App(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.configure(fg_color=colors["background"])
            self.title("Layer contrast · live curve explorer")
            self.geometry("1500x950")
            self.minsize(1200, 750)
            self.layers = [Layer() for _ in range(4)]
            self.global_values = asdict(Layer())
            self.xmax = 4.0
            self.pending = None
            self.layer_controls = []
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(1, weight=1)

            header = ctk.CTkFrame(self, fg_color="transparent")
            header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(16, 8))
            ctk.CTkLabel(header, text="Layer contrast", text_color=colors["text"],
                         font=ctk.CTkFont(size=25, weight="bold")).pack(side="left")
            ctk.CTkLabel(header, text="Raw detail values · no 95th-percentile normalization",
                         text_color=colors["muted"]).pack(side="left", padx=22)

            self.sidebar = ctk.CTkScrollableFrame(self, width=555, fg_color=colors["panel"],
                                                 **scrollbar_colors)
            self.sidebar.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
            self.sidebar.grid_columnconfigure(0, weight=1)
            self.section(self.sidebar, "Layers", 0)
            self.count = ctk.CTkOptionMenu(self.sidebar, values=[str(i) for i in range(1, 17)],
                                         fg_color=colors["raised"], text_color=colors["text"],
                                         button_color=colors["track"], button_hover_color=colors["border"],
                                         dropdown_fg_color=colors["raised"],
                                         dropdown_hover_color=colors["track"],
                                         dropdown_text_color=colors["text"],
                                         command=self.change_count)
            self.count.set("4")
            self.count.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 12))
            ctk.CTkLabel(self.sidebar, text="Each group: ALL sets every layer; L1, L2, … adjust individual layers.\nHigher values are at the top. Scroll sideways for more layers.",
                         justify="left", anchor="w", text_color=colors["muted"]).grid(row=2, column=0, sticky="ew", padx=8)
            self.parameter_groups = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            self.parameter_groups.grid(row=3, column=0, sticky="ew", pady=5)
            self.parameter_groups.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(self.sidebar, text="Controls set output slope; the top plot shows output / input.\nSlope decreases smoothly from 1 to minimum gain.\nMinimum gain: 0 = saturation; 1 = unchanged.\nBeyond the second endpoint, output has the selected minimum slope.\nIf endpoints cross, the other endpoint moves to keep their order.",
                         justify="left", anchor="w", text_color=colors["muted"]).grid(row=4, column=0, sticky="ew", padx=8)

            right = ctk.CTkFrame(self, fg_color=colors["panel"])
            right.grid(row=1, column=1, sticky="nsew", padx=(0, 16), pady=(0, 16))
            right.grid_columnconfigure(0, weight=1)
            right.grid_rowconfigure(1, weight=1)
            toolbar = ctk.CTkFrame(right, fg_color="transparent")
            toolbar.grid(row=0, column=0, sticky="ew", padx=12, pady=10)
            ctk.CTkLabel(toolbar, text="Plot maximum |detail|", text_color=colors["text"]).pack(side="left")
            self.range_entry = ctk.CTkEntry(toolbar, width=80, fg_color=colors["background"],
                                           border_color=colors["border"], text_color=colors["text"])
            self.range_entry.insert(0, "4.0")
            self.range_entry.pack(side="left", padx=8)
            self.range_entry.bind("<Return>", lambda event: self.apply_range())
            ctk.CTkButton(toolbar, text="Apply", width=65, command=self.apply_range,
                           fg_color=colors["accent"], hover_color=colors["hover"],
                           text_color=colors["on_accent"]).pack(side="left")
            ctk.CTkButton(toolbar, text="Save plot", width=100, command=self.save_plot,
                           fg_color=colors["raised"], hover_color=colors["track"],
                           border_width=1, border_color=colors["border"],
                           text_color=colors["text"]).pack(side="right")
            self.figure = Figure(figsize=(9, 8), dpi=100, facecolor=colors["panel"])
            self.axes = self.figure.subplots(2, 1, sharex=True)
            self.figure.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.09, hspace=0.32)
            self.canvas = FigureCanvasTkAgg(self.figure, master=right)
            self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8)
            self.status = ctk.CTkLabel(right, text="", text_color=colors["muted"], justify="left", anchor="w", wraplength=750)
            self.status.grid(row=2, column=0, sticky="ew", padx=18, pady=10)
            self.rebuild_groups()
            self.draw_plot()

        @staticmethod
        def section(parent, text, row):
            ctk.CTkLabel(parent, text=text, text_color=colors["text"],
                         font=ctk.CTkFont(size=15, weight="bold"), anchor="w").grid(
                row=row, column=0, sticky="ew", padx=8, pady=(14, 7))

        def slider_maximum(self):
            # Changing the viewing range must not change existing endpoints.
            return max(self.xmax, self.global_values["unity_x"], self.global_values["minimum_x"],
                       *(layer.minimum_x for layer in self.layers))

        def specs(self):
            return [("unity_x", "x where output slope is still 1 · decrease starts"),
                    ("minimum_x", "x where output slope reaches minimum · decrease ends"),
                    ("min_gain", "Minimum gain (output slope) · 0 to 1")]

        def color(self, index):
            return layer_palette(index / max(1, len(self.layers) - 1))

        def rebuild_groups(self):
            for widget in self.parameter_groups.winfo_children():
                widget.destroy()
            self.global_controls = {}
            self.layer_controls = [{} for _ in self.layers]
            self.control_maximum = self.slider_maximum()
            for row, (key, label) in enumerate(self.specs()):
                # Endpoint groups share an x scale; minimum gain uses [0, 1].
                maximum = 1.0 if key == "min_gain" else self.control_maximum
                group = ctk.CTkFrame(self.parameter_groups, fg_color=colors["raised"],
                                     border_width=1, border_color=colors["border"])
                group.grid(row=row, column=0, sticky="ew", padx=4, pady=5)
                group.grid_columnconfigure(1, weight=1)
                ctk.CTkLabel(group, text=label, font=ctk.CTkFont(size=14, weight="bold"),
                             text_color=colors["text"], anchor="w").grid(
                                 row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 0))
                master = Control(group, "ALL", self.global_values[key], maximum,
                                 lambda value, k=key: self.set_all(k, value))
                master.grid(row=1, column=0, sticky="n", padx=(3, 1), pady=(0, 18))
                self.global_controls[key] = master
                bank = ctk.CTkScrollableFrame(group, orientation="horizontal", height=175,
                                              width=420, fg_color="transparent", **scrollbar_colors)
                bank.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 4))
                for index, layer in enumerate(self.layers):
                    color = "#" + "".join(f"{round(v * 255):02x}" for v in self.color(index)[:3])
                    control = Control(bank, f"L{index + 1}", getattr(layer, key), maximum,
                                      lambda value, i=index, k=key: self.set_layer(i, k, value), color=color)
                    control.grid(row=0, column=index, sticky="n", padx=0)
                    self.layer_controls[index][key] = control

        def change_count(self, value):
            count = int(value)
            self.layers = self.layers[:count]
            self.layers.extend(Layer(**self.global_values) for _ in range(count - len(self.layers)))
            self.rebuild_groups()
            self.schedule_plot()

        def set_all(self, key, value):
            master = adjust_control(Layer(**self.global_values), key, value, self.control_maximum)
            self.global_values = asdict(master)
            for endpoint, control in self.global_controls.items():
                control.set(self.global_values[endpoint])
            for index, layer in enumerate(self.layers):
                self.layers[index] = adjust_control(layer, key, value, self.control_maximum)
                self.sync_layer_controls(index)
            self.schedule_plot()

        def set_layer(self, index, key, value):
            self.layers[index] = adjust_control(self.layers[index], key, value, self.control_maximum)
            self.sync_layer_controls(index)
            self.schedule_plot()

        def sync_layer_controls(self, index):
            for key, control in self.layer_controls[index].items():
                control.set(getattr(self.layers[index], key))

        def apply_range(self):
            try:
                value = float(self.range_entry.get())
                if not math.isfinite(value) or not 0 < value <= 1e6:
                    raise ValueError("Plot maximum must be greater than 0 and at most 1,000,000.")
            except ValueError as exc:
                messagebox.showerror("Invalid plot range", str(exc), parent=self)
                return
            self.xmax = value
            self.rebuild_groups()
            self.schedule_plot()

        def schedule_plot(self):
            # Keep one queued redraw while dragging; do not defer until release.
            if self.pending is None:
                self.pending = self.after(25, self.draw_plot)

        def draw_plot(self):
            if self.pending is not None:
                self.after_cancel(self.pending)
                self.pending = None
            # Sample every transition separately, even when it is very narrow.
            x = np.unique(np.concatenate((np.linspace(0, self.xmax, 1600),
                *(np.linspace(layer.unity_x, min(layer.minimum_x, self.xmax), 100)
                  for layer in self.layers if layer.unity_x <= self.xmax))))
            for ax in self.axes:
                ax.clear()
                ax.set_facecolor(colors["plot"])
                ax.tick_params(colors=colors["muted"])
                ax.xaxis.label.set_color(colors["text"])
                ax.yaxis.label.set_color(colors["text"])
                for spine in ax.spines.values():
                    spine.set_color(colors["border"])
                ax.grid(alpha=0.24, color=colors["reference"])
            for index, layer in enumerate(self.layers):
                gain = layer_gain(x, layer)
                self.axes[0].plot(x, gain, color=self.color(index), label=f"L{index + 1}", lw=1.8)
                self.axes[0].plot([layer.unity_x, layer.minimum_x],
                                  layer_gain([layer.unity_x, layer.minimum_x], layer), linestyle="none",
                                  marker="o", ms=4, color=self.color(index), clip_on=True)
                self.axes[1].plot(x, x * gain, color=self.color(index), lw=1.8)
            self.axes[0].axhline(1, color=colors["reference"], ls="--", lw=1, alpha=0.7)
            self.axes[1].plot([0, self.xmax], [0, self.xmax], color=colors["reference"], ls="--", lw=1, alpha=0.7)
            self.axes[0].set_title("Gain per layer · dots mark slope-transition endpoints",
                                   loc="left", fontsize=12, color=colors["text"])
            self.axes[0].set_ylabel("Gain · output / input")
            self.axes[0].tick_params(axis="x", labelbottom=True)
            self.axes[0].set_xlabel("Input |detail| · raw units")
            self.axes[0].set_ylim(-0.05, 1.12)
            self.axes[1].set_title("Output magnitude · dashed line = unchanged",
                                   loc="left", fontsize=12, color=colors["text"])
            self.axes[1].set_ylabel("Output |detail| × gain")
            self.axes[1].set_xlabel("Input |detail| · raw units")
            self.axes[1].set_xlim(0, self.xmax)
            self.axes[1].set_ylim(bottom=0)
            self.axes[0].legend(loc="upper right", ncol=min(4, len(self.layers)), fontsize=8,
                                facecolor=colors["panel"], edgecolor=colors["border"],
                                labelcolor=colors["text"], framealpha=1)
            self.status.configure(text=f"{len(self.layers)} independent layers · matching curves overlap. "
                                  "Output slope transitions smoothly from 1 to minimum gain.\n"
                                  "Output = input × gain never decreases. Zero minimum gain gives saturation. "
                                  "Raw magnitudes; no percentile normalization.")
            self.canvas.draw_idle()

        def save_plot(self):
            self.draw_plot()
            path = filedialog.asksaveasfilename(parent=self, defaultextension=".png",
                    initialfile="layer_contrast_curves.png", filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")])
            if path:
                try:
                    self.figure.savefig(path, dpi=180, facecolor=self.figure.get_facecolor())
                except OSError as exc:
                    messagebox.showerror("Could not save plot", str(exc), parent=self)

    return App()


def self_test():
    """Exercise the numerical behavior without a GUI or VSOLImage."""
    x = np.array([-100., -4., -3., -2., 0., 2., 3., 4., 100.])
    layer = Layer(unity_x=2, minimum_x=4)
    gain = layer_gain(x, layer)
    np.testing.assert_allclose(gain, gain[::-1])
    assert np.all(gain[np.abs(x) <= 2] == 1)
    np.testing.assert_allclose(np.abs(x) * gain, [3, 3, 2.8125, 2, 0, 2, 2.8125, 3, 3])
    assert layer_gain(3, layer) == 0.9375
    for start, end in [(0, 1), (2, 4), (0.0001, 0.000101), (1e5, 1e6)]:
        values = np.linspace(0, end * 2, 10001)
        gains = layer_gain(values, Layer(start, end))
        assert np.all((gains >= 0) & (gains <= 1))
        assert np.all(np.diff(gains) <= 1e-14)
        np.testing.assert_allclose(
            layer_gain([start, end], Layer(start, end)), [1, (start + (end - start) / 2) / end])
        output = values * gains
        assert np.all(np.diff(output) >= -end * 1e-14)
        # Output joins the identity with slope 1 and the saturation tail with slope 0.
        h = (end - start) * 1e-6
        np.testing.assert_allclose(
            ((start + h) * layer_gain(start + h, Layer(start, end)) - start) / h, 1, atol=1e-6)
        np.testing.assert_allclose(
            (end * layer_gain(end, Layer(start, end))
             - (end - h) * layer_gain(end - h, Layer(start, end))) / h, 0, atol=1e-6)
    # Fixed raw magnitude has fixed gain regardless of other values in an array.
    np.testing.assert_allclose(layer_gain([3], layer)[0], layer_gain([3, 1e6], layer)[0])
    # Dragging either slider through its partner or to a limit stays valid.
    for maximum in (1e-6, 4, 1e6):
        current = Layer(maximum / 4, maximum * 3 / 4)
        for key in ("unity_x", "minimum_x"):
            for value in (0, maximum, maximum / 2, -maximum, maximum * 2):
                current = adjust_control(current, key, value, maximum)
                assert 0 <= current.unity_x < current.minimum_x <= maximum
                endpoint_gain = layer_gain([current.unity_x, current.minimum_x], current)
                assert endpoint_gain[0] == 1
                assert 0 < endpoint_gain[1] <= 1
    for invalid in (Layer(2, 2), Layer(4, 2), Layer(-1, 2), Layer(0, math.inf)):
        try:
            layer_gain(x, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid endpoints were accepted")
    # The selected minimum is the tail slope, not the output/input ratio.
    for floor in (0.0, 0.2, 0.7, 1.0):
        layer = Layer(2, 4, floor)
        values = np.linspace(0, 8, 10001)
        gains = layer_gain(values, layer)
        np.testing.assert_allclose(
            layer_gain([0, 2, 4, 8], layer), [1, 1, floor + (1 - floor) * 0.75,
                                           floor + (1 - floor) * 0.375])
        np.testing.assert_allclose(layer_gain(3, layer), floor + (1 - floor) * 0.9375)
        assert np.all((gains >= floor) & (gains <= 1))
        assert np.all(np.diff(gains) <= 1e-14)
        np.testing.assert_allclose(layer_gain(-values, layer), gains)
        output = values * gains
        slopes = np.diff(output) / np.diff(values)
        assert np.all(slopes >= floor - 1e-10)
        assert np.all(slopes <= 1 + 1e-10)
        assert np.all(np.diff(slopes) <= 1e-9)
        np.testing.assert_allclose(slopes[values[:-1] >= 4], floor, atol=1e-10)
        signed = np.linspace(-8, 8, 20001)
        assert np.all(np.diff(signed * layer_gain(signed, layer)) >= -1e-13)
        h = 1e-5
        np.testing.assert_allclose(
            ((3 + h) * layer_gain(3 + h, layer) - (3 - h) * layer_gain(3 - h, layer)) / (2 * h),
            (1 + floor) / 2, atol=1e-9)
        for key in ("unity_x", "minimum_x"):
            assert adjust_control(layer, key, 3, 4).min_gain == floor
        for new_floor in (-1, 0, 0.4, 1, 2):
            moved = adjust_control(layer, "min_gain", new_floor, 4)
            assert (moved.unity_x, moved.minimum_x) == (2, 4)
            assert moved.min_gain == min(max(new_floor, 0), 1)
    for floor in (-0.1, 1.1, math.nan, math.inf):
        try:
            layer_gain(x, Layer(2, 4, floor))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid minimum gain was accepted")
    # Pointwise evaluation is finite and independent of sample order, even at extremes.
    for layer in (Layer(0, 1e-300), Layer(2, 4), Layer(1e308, 1.5e308)):
        values = np.array([0, 1e-300, 3, 1e308, np.finfo(float).max])
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            gains = layer_gain(values, layer)
        assert np.all(np.isfinite(gains))
        assert np.all((gains >= 0) & (gains <= 1))
        np.testing.assert_array_equal(layer_gain(values[::-1], layer), gains[::-1])
    print("PASS: integrated slope, no contrast reversal, saturation, symmetry, smooth joins, raw units, and slider ordering.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="Check gain math without opening a window")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    try:
        app = create_app()
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}.\nInstall: python -m pip install customtkinter matplotlib numpy")
    app.mainloop()


if __name__ == "__main__":
    main()
