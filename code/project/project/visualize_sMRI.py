import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Optional: for an interactive widget in Jupyter
try:
    from nilearn.plotting import view_img

    INTERACTIVE = True
except ImportError:
    INTERACTIVE = False

# ---------- 2. Load the image ----------
nii_path = "anat_thickness.nii.gz"  # adjust if the file is elsewhere
img = nib.load(nii_path)
data = img.get_fdata()  # float32/float64 array, shape (X, Y, Z)

# ---------- 3. Pick mid-slices ----------
cx, cy, cz = np.array(data.shape) // 2
sagittal = np.rot90(data[cx, :, :])  # sagittal slice: left–right axis
coronal = np.rot90(data[:, cy, :])  # coronal  slice: front–back axis
axial = np.rot90(data[:, :, cz])  # axial    slice: top–bottom axis

# ---------- 4. Display ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, slc, title in zip(
    axes, [sagittal, coronal, axial], ["Sagittal", "Coronal", "Axial"]
):
    im = ax.imshow(
        slc, cmap="gray", vmin=np.percentile(data, 1), vmax=np.percentile(data, 99)
    )
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Thickness")
plt.tight_layout()
plt.show()

# ---------- 5. Optional interactive view ----------
if INTERACTIVE:
    display(view_img(img, cmap="gray"))
