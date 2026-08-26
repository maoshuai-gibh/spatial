#!/home/mshuai/.conda/envs/picture/bin/python

import sys
import zarr
import numpy as np
import subprocess
import re

# arguments
if len(sys.argv) != 4:

    print(
        "Usage:\n"
        "python qc_lsm_omezarr.py "
        "<original_lsm> <omezarr> <report>"
    )

    sys.exit(1)

lsm_file=sys.argv[1]

zarr_path=sys.argv[2]

report=sys.argv[3]

# open zarr
root=zarr.open(
    zarr_path,
    mode="r"
)

# find array
def find_array(group):


    for key in group.keys():

        obj=group[key]


        if isinstance(obj,zarr.Array):

            return obj


        elif isinstance(obj,zarr.Group):


            result=find_array(obj)


            if result is not None:

                return result

    return None

image=find_array(root)

if image is None:

    raise RuntimeError(
        "Cannot find image array"
    )

# metadata
level0={}

try:

    level0=root["0"].attrs.asdict()

except:

    pass

# Bio-Formats metadata
def get_bioformats_metadata(path):

    try:

        result=subprocess.run(

            [
                "/home/mshuai/software/bioformats2raw-0.9.1/bin/showinf",
                "-nopix",
                path
            ],

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            timeout=60

        )


        return result.stdout

    except Exception as e:

        return str(e)

bf_meta=get_bioformats_metadata(
    lsm_file
)

# write report
with open(report,"w") as f:

    f.write(
        "OME-Zarr QC Report\n"
    )

    f.write(
        "==================\n\n"
    )

    # Dimension
    f.write(
        "1. Image dimension\n"
    )

    f.write(
        "------------------\n\n"
    )


    shape=image.shape


    f.write(
        f"Shape: {shape}\n\n"
    )



    if len(shape)==5:


        T,C,Z,Y,X=shape


        f.write(
            "Dimension order: T,C,Z,Y,X\n\n"
        )


        f.write(f"T={T}\n")
        f.write(f"C={C}\n")
        f.write(f"Z={Z}\n")
        f.write(f"Y={Y}\n")
        f.write(f"X={X}\n\n")


        f.write(
            "Dimension check: PASS\n"
        )

    # Channel
    f.write(
        "\n2. Channel information\n"
    )

    f.write(
        "----------------------\n\n"
    )

    try:


        channels=level0["omero"]["channels"]


        f.write(
            f"Channel number: {len(channels)}\n\n"
        )


        for i,c in enumerate(channels):

            f.write(
                f"Channel {i}: {c.get('label')}\n"
            )


    except:


        f.write(
            "Channel metadata not found\n"
        )

    # Pixel size
    f.write(
        "\n3. Pixel size\n"
    )

    f.write(
        "-------------\n\n"
    )

    try:


        scale=(

            level0
            ["multiscales"]
            [0]
            ["datasets"]
            [0]
            ["coordinateTransformations"]
            [0]
            ["scale"]

        )


        f.write(
            f"X pixel size: {scale[4]} um\n"
        )

        f.write(
            f"Y pixel size: {scale[3]} um\n"
        )

        f.write(
            f"Z step: {scale[2]} um\n"
        )

    except Exception as e:


        f.write(
            "Pixel metadata not found\n"
        )

    # Bit depth
    f.write(
        "\n4. Bit depth / intensity\n"
    )

    f.write(
        "------------------------\n\n"
    )


    f.write(
        f"OME-Zarr dtype: {image.dtype}\n"
    )


    if image.dtype==np.uint8:

        f.write(
            "OME-Zarr bit depth: 8 bit\n"
        )

    elif image.dtype==np.uint16:

        f.write(
            "OME-Zarr bit depth: 16 bit\n"
        )



    f.write("\n")


    # Bioformats original
    keys=[
        "Pixel type",
        "SizeX",
        "SizeY",
        "SizeZ",
        "SizeC"
    ]


    for k in keys:


        match=re.search(
            k+r".*",
            bf_meta
        )


        if match:

            f.write(
                match.group(0)+"\n"
            )

    f.write("\n")

    f.write(
        "5. Reading check\n"
    )

    f.write(
        "----------------\n\n"
    )


    try:

        image[0,0,0,:,:]

        f.write(
            "OME-Zarr reading: PASS\n"
        )

    except:

        f.write(
            "OME-Zarr reading: FAILED\n"
        )

print(
    "QC finished:"
)

print(report)
