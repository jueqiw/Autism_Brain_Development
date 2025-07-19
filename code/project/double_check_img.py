import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # np.savez_compressed(
    #     f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/dispfield/{train_batch_data['subject_id'][0]}_"
    #     f"{digits}_{j}.npz",
    #     axial=axial,
    # )

    read_img = np.load(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/axial/dispfield/28705_dispfield_2_-2.npz"
    )
    axial = read_img["axial"]

    plt.imshow(np.sqrt(np.sum(axial ** 2, axis=0)), cmap="gray")
    plt.title("Axial Displacement Field")
    plt.axis("off")
    plt.show()
    plt.savefig("double_check_axial_dispfield.png")