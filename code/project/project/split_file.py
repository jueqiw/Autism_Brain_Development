import os
import sys
import math
from pathlib import Path


if __name__ == "__main__":
    # read the file
    file_path = Path(
        "/project/ace-ig/jueqiw/code/CrossModalityLearning/code/job_scripts/Autism/img/MNI/ABIDE_I/ABIDE_199.sh"
    )
    with open(file_path, "r") as file:
        lines = file.readlines()

    first_15_lines = lines[:10]

    # split into 20 files:
    num_files = 20
    lines_per_file = math.ceil((len(lines) - 10) / num_files)
    for i in range(num_files):
        new_file_path = file_path.parent / f"ABIDE_{i + 200}.sh"

        with open(new_file_path, "w") as new_file:
            new_file.writelines(first_15_lines)

            start_index = i * lines_per_file + 10
            end_index = min(start_index + lines_per_file, len(lines))

            new_file.writelines(lines[start_index:end_index])

    #     print(f"Created {new_file_path} with lines {start_index} to {end_index - 1}")
    # os.system(
    #     f"cat /project/ace-ig/jueqiw/code/CrossModalityLearning/code/job_scripts/Autism/img/MNI/ABIDE_I/ABIDE_200.sh"
    # )

    #     print(f"Created {new_file_path} with lines {start_index} to {end_index - 1}")


# import os

# for i in range(200, 218):
#     os.system(
#         f"qsub /project/ace-ig/jueqiw/code/CrossModalityLearning/code/job_scripts/Autism/img/MNI/ABIDE_I/ABIDE_{i}.sh"
#     )
