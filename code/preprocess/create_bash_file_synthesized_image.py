from pathlib import Path
import os

if __name__ == "__main__":
    for i in range(20):
        if Path(
            f"/project/ace-genetics/jueqiw/Autism_Brain_Development/code/scripts/ABIDE_{i}.sh"
        ).exists():
            os.remove(
                Path(
                    f"/project/ace-genetics/jueqiw/Autism_Brain_Development/code/scripts/ABIDE_{i}.sh"
                )
            )
        with open(
            Path(
                f"/project/ace-genetics/jueqiw/Autism_Brain_Development/code/scripts/ABIDE_{i}.sh"
            ),
            "a",
        ) as f:
            f.write("#!/bin/bash -l\n")
            f.write("#$ -P ace-genetics          # SCC project name\n")
            f.write("#$ -l h_rt=12:00:00   # Specify the hard time limit for the job\n")
            f.write(f"#$ -N  Syn_{i}\n")
            f.write(
                "#$ -j y               # Merge the error and output streams into a single file\n"
            )
            f.write("#$ -pe omp 8\n")

            f.write("module load python3/3.8.10\n")
            f.write(
                "export PYTHONPATH=/projectnb/ace-ig/jueqiw/python/lib/python3.8.10/site-packages:$PYTHONPATH\n"
            )
            f.write("module load pytorch/1.13.1\n")
            f.write(
                f"python /project/ace-genetics/jueqiw/Autism_Brain_Development/code/project/create_batch_file_for_synthesized_image.py --id={i}\n"
            )

    for i in range(21):
        os.system(
            f"qsub /project/ace-genetics/jueqiw/Autism_Brain_Development/code/scripts/ABIDE_{i}.sh"
        )
