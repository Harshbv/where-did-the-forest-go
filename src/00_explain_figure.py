"""Side-by-side true/false color figure with labels pointing at what to look for."""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from config import OUT_DIR

true = mpimg.imread(OUT_DIR / "01_true_color.png")
false = mpimg.imread(OUT_DIR / "01_false_color.png")
h, w = true.shape[:2]

# Labels as fractions of image width/height, so they survive a resolution change.
notes = [
    (0.90, 0.25, "Forest\n(dark green / bright red)"),
    (0.26, 0.33, "Cleared pasture\n(brown)"),
    (0.89, 0.81, "Town"),
    (0.72, 0.30, "River"),
    (0.80, 0.93, "Reservoir"),
    (0.40, 0.65, "'Fishbone' clearing\nalong straight roads"),
]

fig, axes = plt.subplots(1, 2, figsize=(18, 9.5))
for ax, img, title in [
    (axes[0], true, "TRUE COLOR - what your eyes would see"),
    (axes[1], false, "FALSE COLOR - near-infrared shown as red\n(forest glows red)"),
]:
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    for fx, fy, text in notes:
        ax.annotate(text, xy=(fx * w, fy * h), xytext=(fx * w, fy * h - 0.06 * h),
                    ha="center", fontsize=10, color="black",
                    bbox=dict(boxstyle="round", fc="yellow", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2))

fig.suptitle("Rondonia, Brazil - 29 July 2021 - about 55 x 55 km (Sentinel-2)",
             fontsize=16, y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(OUT_DIR / "00_explained.png", dpi=110)
print(OUT_DIR / "00_explained.png")
