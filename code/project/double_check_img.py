import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # np.savez_compressed(
    #     f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/dispfield/{train_batch_data['subject_id'][0]}_"
    #     f"{digits}_{j}.npz",
    #     axial=axial,
    # )
    title = "transformed"
    read_img = np.load(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/axial/{title}/51577_{title}_0_2.npz"
    )
    axial = read_img["axial"]

    plt.figure(figsize=(10, 10))
    plt.imshow(axial, cmap="gray")
    plt.title("Axial Jacobian Field")
    # add colorbar
    plt.colorbar()
    plt.axis("off")
    plt.show()
    plt.savefig("double_check_axial_jacobian.png")
