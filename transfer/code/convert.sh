#!/bin/bash

# bioformats2raw
BF2RAW=/home/mshuai/software/bioformats2raw-0.9.1/bin/bioformats2raw

# python environment
PYTHON=/home/mshuai/.conda/envs/picture/bin/python

# project
PROJECT=/home/mshuai/transfer

INPUT_DIR=${PROJECT}/raw_picture

OUTPUT_DIR=${PROJECT}/OME

QC_DIR=${PROJECT}/QC


mkdir -p "${OUTPUT_DIR}"
mkdir -p "${QC_DIR}"


echo "====================================="
echo "Start transfer LSM to OME-Zarr"

# loop samples
for folder in "${INPUT_DIR}"/*

do


    if [ -d "$folder" ]; then


        sample_name=$(basename "$folder")


        sample_output="${OUTPUT_DIR}/${sample_name}"


        mkdir -p "${sample_output}"



        echo ""
        echo "====================================="
        echo "Sample:"
        echo "${sample_name}"


        find "$folder" -type f -name "*.lsm" | while read lsm_file

        do

            image_name=$(basename "$lsm_file" .lsm)

            output_zarr="${sample_output}/${image_name}.zarr"

            qc_report="${QC_DIR}/${sample_name}_${image_name}_QC.txt"

            echo ""
            echo "-------------------------------------"

            echo "Input:"
            echo "${lsm_file}"

            echo "Output:"
            echo "${output_zarr}"

            echo "-------------------------------------"

            # LSM -> OME-Zarr
            ${BF2RAW} \
            --overwrite \
            -r 1 \
            -c null \
            "${lsm_file}" \
            "${output_zarr}"

            if [ $? -eq 0 ]

            then

                echo "Conversion SUCCESS:"
                echo "${image_name}"

                # QC
                ${PYTHON} \
                ${PROJECT}/code/qc_lsm_omezarr.py \
                "${lsm_file}" \
                "${output_zarr}" \
                "${qc_report}"



                if [ $? -eq 0 ]

                then

                    echo "QC SUCCESS:"
                    echo "${qc_report}"


                else

                    echo "QC FAILED:"
                    echo "${image_name}"

                fi

            else

                echo "Conversion FAILED:"
                echo "${image_name}"


            fi

        done

    fi
done
echo "All finished"
