#!/home/mshuai/.conda/envs/picture/bin/python

from pathlib import Path
import argparse
import shutil
import re
import warnings

import numpy as np
import pandas as pd
import zarr
from basicpy import BaSiC

DEFAULT_INPUT_DIR = Path(
    "/home/mshuai/transfer/OME/test"
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/mshuai/preprocessing/zarr-flat"
)

# 2. Channel 设置
# 实验 Channel 1 -> c=0 -> DAPI -> 不做 flat-field
# 实验 Channel 2 -> c=1 -> probe
# 实验 Channel 3 -> c=2 -> probe
DAPI_CHANNEL = 0
#如果想要跳过mcherry通道，在下方注释：DAPI 完全跳过修改

# 3. BaSiC 参数
BASIC_SMOOTHNESS_FLATFIELD = 5
#修改该参数使图像平滑程度不同
BASIC_GET_DARKFIELD = False

# 4. Model 拟合采样参数
MAX_PLANES_PER_IMAGE_UNIT = 8

MAX_PLANES_PER_ZARR = 40

MAX_PLANES_PER_MODEL = 100

MIN_PLANES_PER_MODEL = 1
#虽然将该部分修改为1，只是为了保证大图能够顺利运行！！！！！！！！！！！！！！！！！正常应为5

# 5. 工具函数
def join_path(parent, child):

    parent = str(parent).strip("/")
    child = str(child).strip("/")

    if parent:

        return f"{parent}/{child}"

    return child


def safe_name(text):

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text)
    )

# 6. 读取 OME-Zarr axes
def parse_axes(multiscale, shape):

    axes_raw = multiscale.get(
        "axes",
        None
    )

    axes = []


    if axes_raw:

        for axis in axes_raw:

            if isinstance(
                axis,
                str
            ):

                axes.append(
                    axis.lower()
                )

            elif isinstance(
                axis,
                dict
            ):

                axes.append(
                    str(
                        axis.get(
                            "name",
                            ""
                        )
                    ).lower()
                )


    if len(axes) == len(shape):

        return axes


    # --------------------------------------------
    # fallback
    # --------------------------------------------

    if len(shape) == 5:

        return [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]

    elif len(shape) == 4:

        return [
            "c",
            "z",
            "y",
            "x",
        ]

    elif len(shape) == 3:

        return [
            "c",
            "y",
            "x",
        ]

    elif len(shape) == 2:

        return [
            "y",
            "x",
        ]


    raise RuntimeError(
        f"无法判断 axes: shape={shape}"
    )


# 7. 找出 OME-Zarr 所有 full-resolution image arrays
def discover_primary_arrays(root):

    results = []


    def recurse(
        group,
        group_path=""
    ):

        try:

            multiscales = group.attrs.get(
                "multiscales"
            )

        except Exception:

            multiscales = None


        if multiscales:

            for ms in multiscales:

                datasets = ms.get(
                    "datasets",
                    []
                )


                if not datasets:

                    continue


                dataset = datasets[0]

                relative_path = str(
                    dataset.get(
                        "path",
                        ""
                    )
                )


                full_path = join_path(
                    group_path,
                    relative_path
                )


                try:

                    arr = root[
                        full_path
                    ]

                except Exception:

                    continue


                # 必须是真正数组
                if not isinstance(
                    arr,
                    zarr.Array
                ):

                    continue


                axes = parse_axes(
                    ms,
                    arr.shape
                )


                # 必须包含 X/Y
                if (
                    "x" not in axes
                    or
                    "y" not in axes
                ):

                    continue


                results.append(
                    {
                        "path":
                            full_path,

                        "shape":
                            tuple(
                                arr.shape
                            ),

                        "axes":
                            axes,
                    }
                )

        # 继续搜索 subgroup
        for name, subgroup in group.groups():

            recurse(
                subgroup,
                join_path(
                    group_path,
                    name
                )
            )


    recurse(root)

    # 去重
    unique = {}

    for item in results:

        unique[
            item["path"]
        ] = item


    return list(
        unique.values()
    )

# 8. 获取 Y/X 尺寸
def get_yx_shape(
    shape,
    axes
):

    y_axis = axes.index(
        "y"
    )

    x_axis = axes.index(
        "x"
    )


    return (
        int(
            shape[
                y_axis
            ]
        ),

        int(
            shape[
                x_axis
            ]
        ),
    )

# 9. 获取 channel 数量
def get_channel_count(
    shape,
    axes
):

    if "c" not in axes:

        return 1


    c_axis = axes.index(
        "c"
    )


    return int(
        shape[
            c_axis
        ]
    )

# 10. 生成某个 channel 的所有 XY plane selector
def generate_channel_planes(
    shape,
    axes,
    channel
):

    y_axis = axes.index(
        "y"
    )

    x_axis = axes.index(
        "x"
    )


    if "c" in axes:

        c_axis = axes.index(
            "c"
        )


        if channel >= shape[c_axis]:

            return


    else:

        c_axis = None


    varying_axes = []


    for i in range(
        len(shape)
    ):

        if i in (
            y_axis,
            x_axis
        ):

            continue


        if (
            c_axis is not None
            and
            i == c_axis
        ):

            continue


        varying_axes.append(
            i
        )


    sizes = [
        int(
            shape[i]
        )
        for i in varying_axes
    ]


    if sizes:

        total = int(
            np.prod(
                sizes
            )
        )

    else:

        total = 1


    for flat_index in range(
        total
    ):

        selector = [
            0
        ] * len(shape)


        selector[
            y_axis
        ] = slice(None)

        selector[
            x_axis
        ] = slice(None)


        if c_axis is not None:

            selector[
                c_axis
            ] = channel


        description = []


        if sizes:

            coords = np.unravel_index(
                flat_index,
                sizes
            )


            for axis_index, value in zip(
                varying_axes,
                coords
            ):

                selector[
                    axis_index
                ] = int(
                    value
                )


                description.append(
                    (
                        axes[
                            axis_index
                        ]
                        +
                        "="
                        +
                        str(
                            int(
                                value
                            )
                        )
                    )
                )


        if not description:

            description = [
                "2D"
            ]


        yield (
            tuple(
                selector
            ),
            ",".join(
                description
            )
        )

# 11. 均匀抽样 plane
def select_evenly(
    items,
    maximum
):

    if len(items) <= maximum:

        return items


    indices = np.linspace(
        0,
        len(items) - 1,
        maximum,
        dtype=int
    )


    return [
        items[i]
        for i in indices
    ]

# 12. 把数据写回原 dtype
def convert_to_original_dtype(
    image,
    dtype
):

    dtype = np.dtype(
        dtype
    )


    if np.issubdtype(
        dtype,
        np.integer
    ):

        info = np.iinfo(
            dtype
        )


        image = np.clip(
            image,
            info.min,
            info.max
        )


        image = np.rint(
            image
        )


    elif np.issubdtype(
        dtype,
        np.floating
    ):

        pass


    else:

        raise RuntimeError(
            f"不支持 dtype: {dtype}"
        )


    return image.astype(
        dtype,
        copy=False
    )

# 13. 主程序
def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
    )


    parser.add_argument(
        "--force",
        action="store_true",
        help="删除旧输出并重新计算"
    )


    args = parser.parse_args()


    input_dir = args.input

    output_dir = args.output

    # 14. 检查路径
    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory 不存在:\n{input_dir}"
        )


    if args.force and output_dir.exists():
        shutil.rmtree(
            output_dir
        )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    model_dir = (
        output_dir
        /
        "_flatfield_models"
    )


    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 15. 找所有输入 zarr
    zarr_paths = sorted(
        [
            p
            for p in input_dir.glob(
                "*.zarr"
            )
            if p.is_dir()
        ]
    )


    if not zarr_paths:

        raise RuntimeError(
            (
                "没有找到 .zarr:\n"
                f"{input_dir}"
            )
        )


    print("========================================")
    print("Flat-field correction")
    print("========================================")

    print(
        f"Input : {input_dir}"
    )

    print(
        f"Output: {output_dir}"
    )

    # 16. 扫描所有 image units
    # 同时建立 model fitting 数据清单
    image_units = []

    for zarr_path in zarr_paths:

        print(
            "\n----------------------------------------"
        )

        print(
            zarr_path.name
        )

        print(
            "----------------------------------------"
        )


        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            root = zarr.open_group(
                str(
                    zarr_path
                ),
                mode="r"
            )


        arrays = discover_primary_arrays(
            root
        )


        if not arrays:

            print(
                "没有发现 OME multiscale image array"
            )

            continue


        for info in arrays:

            shape = info[
                "shape"
            ]

            axes = info[
                "axes"
            ]

            y, x = get_yx_shape(
                shape,
                axes
            )

            channel_count = (
                get_channel_count(
                    shape,
                    axes
                )
            )

            for channel in range(
                channel_count
            ):

                image_units.append(
                    {
                        "zarr_path":
                            zarr_path,

                        "zarr_name":
                            zarr_path.name,

                        "array_path":
                            info[
                                "path"
                            ],

                        "shape":
                            shape,

                        "axes":
                            axes,

                        "channel":
                            channel,

                        "y":
                            y,

                        "x":
                            x,
                    }
                )


    if not image_units:

        raise RuntimeError(
            "没有发现可处理的 image arrays"
        )


    # ========================================================
    # 17. 建立 shared model group
    #
    # 只按：
    #
    # Y尺寸 + X尺寸 + channel
    #
    # 分组。
    #
    # 因为当前 INPUT 文件夹被定义为：
    #
    # 同一 imaging batch /
    # 相同显微镜 acquisition configuration。
    #
    # DAPI channel 不进入 model。
    # ========================================================

    model_groups = {}


    for unit in image_units:

        channel = unit[
            "channel"
        ]

        # DAPI 完全跳过

        if channel == DAPI_CHANNEL:

            continue


        key = (
            unit[
                "y"
            ],

            unit[
                "x"
            ],

            channel,
        )


        model_groups.setdefault(
            key,
            []
        ).append(
            unit
        )


    print(
        (
            "\nProbe flat-field model groups: "
            f"{len(model_groups)}"
        )
    )


    # 18. 拟合 shared BaSiC model
    models = {}

    model_rows = []

    fit_sample_rows = []


    for key, units in sorted(
        model_groups.items()
    ):

        y, x, channel = key


        print(
            "\n========================================"
        )

        print(
            (
                f"MODEL: "
                f"{y}x{x} / c={channel}"
            )
        )

        print(
            "========================================"
        )


        all_samples = []

        # 每个 zarr 的采样量分别控制
        units_by_zarr = {}


        for unit in units:

            units_by_zarr.setdefault(
                unit[
                    "zarr_name"
                ],
                []
            ).append(
                unit
            )


        for zarr_name, zarr_units in (
            units_by_zarr.items()
        ):

            zarr_samples = []


            for unit in zarr_units:

                root = zarr.open_group(
                    str(
                        unit[
                            "zarr_path"
                        ]
                    ),
                    mode="r"
                )


                arr = root[
                    unit[
                        "array_path"
                    ]
                ]


                planes = list(
                    generate_channel_planes(
                        unit[
                            "shape"
                        ],
                        unit[
                            "axes"
                        ],
                        channel
                    )
                )


                planes = select_evenly(
                    planes,
                    MAX_PLANES_PER_IMAGE_UNIT
                )


                for selector, plane_desc in planes:

                    image = np.asarray(
                        arr[
                            selector
                        ],
                        dtype=np.float32
                    )


                    image = np.squeeze(
                        image
                    )


                    if image.shape != (
                        y,
                        x
                    ):

                        continue


                    if not np.any(
                        np.isfinite(
                            image
                        )
                    ):

                        continue


                    zarr_samples.append(
                        (
                            image,
                            unit[
                                "zarr_name"
                            ],
                            unit[
                                "array_path"
                            ],
                            plane_desc,
                        )
                    )

            # 每个 Zarr 最多贡献固定数量
            zarr_samples = select_evenly(
                zarr_samples,
                MAX_PLANES_PER_ZARR
            )


            all_samples.extend(
                zarr_samples
            )

        # 整个 model 最多 MAX_PLANES_PER_MODEL
        all_samples = select_evenly(
            all_samples,
            MAX_PLANES_PER_MODEL
        )


        print(
            (
                "Training planes: "
                f"{len(all_samples)}"
            )
        )


        if (
            len(
                all_samples
            )
            <
            MIN_PLANES_PER_MODEL
        ):

            print(
                (
                    "NO MODEL: "
                    "training planes < "
                    f"{MIN_PLANES_PER_MODEL}"
                )
            )


            model_rows.append(
                {
                    "Y":
                        y,

                    "X":
                        x,

                    "channel":
                        channel,

                    "training_planes":
                        len(
                            all_samples
                        ),

                    "status":
                        "SKIPPED_TOO_FEW_IMAGES",

                    "smoothness_flatfield":
                        BASIC_SMOOTHNESS_FLATFIELD,
                }
            )


            continue


        # Stack
        # shape:N × Y × X

        training_stack = np.stack(
            [
                x[0]
                for x in all_samples
            ],
            axis=0
        ).astype(
            np.float32,
            copy=False
        )

        # BaSiC
        print(
            "Fitting BaSiC..."
        )


        basic = BaSiC(
            get_darkfield=
                BASIC_GET_DARKFIELD,

            smoothness_flatfield=
                BASIC_SMOOTHNESS_FLATFIELD,
        )


        basic.fit(
            training_stack
        )


        flatfield = np.asarray(
            basic.flatfield,
            dtype=np.float32
        )


        flatfield = np.squeeze(
            flatfield
        )


        if flatfield.shape != (
            y,
            x
        ):

            raise RuntimeError(
                (
                    "BaSiC flatfield shape "
                    "不正确:\n"
                    f"{flatfield.shape}"
                    " != "
                    f"{(y, x)}"
                )
            )

        # 清理异常值
        finite_positive = (
            np.isfinite(
                flatfield
            )
            &
            (
                flatfield > 0
            )
        )


        if not np.any(
            finite_positive
        ):

            raise RuntimeError(
                "Flat-field 全部无效"
            )


        median_value = float(
            np.median(
                flatfield[
                    finite_positive
                ]
            )
        )

        # normalize median = 1
        flatfield = (
            flatfield
            /
            median_value
        )


        invalid = (
            ~np.isfinite(
                flatfield
            )
            |
            (
                flatfield <= 0
            )
        )


        flatfield[
            invalid
        ] = 1.0

        # 保存 model
        model_name = (
            f"YX{y}x{x}"
            f"__ch{channel}"
            f"__flatfield.npy"
        )


        model_path = (
            model_dir
            /
            model_name
        )


        np.save(
            model_path,
            flatfield.astype(
                np.float32
            )
        )


        models[
            key
        ] = flatfield


        p05 = float(
            np.percentile(
                flatfield,
                5
            )
        )

        p50 = float(
            np.percentile(
                flatfield,
                50
            )
        )

        p95 = float(
            np.percentile(
                flatfield,
                95
            )
        )


        print(
            (
                "Model complete:\n"
                f"  p05 = {p05:.5f}\n"
                f"  p50 = {p50:.5f}\n"
                f"  p95 = {p95:.5f}\n"
                f"  p95/p05 = "
                f"{p95 / p05:.4f}"
            )
        )


        model_rows.append(
            {
                "Y":
                    y,

                "X":
                    x,

                "channel":
                    channel,

                "training_planes":
                    len(
                        all_samples
                    ),

                "smoothness_flatfield":
                    BASIC_SMOOTHNESS_FLATFIELD,

                "darkfield":
                    BASIC_GET_DARKFIELD,

                "p05":
                    p05,

                "median":
                    p50,

                "p95":
                    p95,

                "p95_p05_ratio":
                    (
                        p95
                        /
                        p05
                    ),

                "model_file":
                    str(
                        model_path
                    ),

                "status":
                    "MODEL_OK",
            }
        )

        # 保存哪些图参与拟合

        for (
            image,
            zarr_name,
            array_path,
            plane_desc,
        ) in all_samples:

            fit_sample_rows.append(
                {
                    "Y":
                        y,

                    "X":
                        x,

                    "channel":
                        channel,

                    "zarr":
                        zarr_name,

                    "array":
                        array_path,

                    "plane":
                        plane_desc,
                }
            )

    # 19. 复制 raw Zarr 到 output
    apply_rows = []


    for source_zarr in zarr_paths:

        destination_zarr = (
            output_dir
            /
            source_zarr.name
        )


        print(
            "\n========================================"
        )

        print(
            f"PROCESS: {source_zarr.name}"
        )

        print(
            "========================================"
        )


        if destination_zarr.exists():

            shutil.rmtree(
                destination_zarr
            )


        print(
            "Copy raw Zarr..."
        )


        shutil.copytree(
            source_zarr,
            destination_zarr
        )


        source_root = zarr.open_group(
            str(
                source_zarr
            ),
            mode="r"
        )


        destination_root = zarr.open_group(
            str(
                destination_zarr
            ),
            mode="r+"
        )


        arrays = discover_primary_arrays(
            source_root
        )


        for info in arrays:

            array_path = info[
                "path"
            ]


            source_array = source_root[
                array_path
            ]


            destination_array = (
                destination_root[
                    array_path
                ]
            )


            shape = tuple(
                source_array.shape
            )

            axes = info[
                "axes"
            ]


            y, x = get_yx_shape(
                shape,
                axes
            )


            channel_count = (
                get_channel_count(
                    shape,
                    axes
                )
            )


            for channel in range(
                channel_count
            ):

                # DAPI

                if channel == DAPI_CHANNEL:
                    continue

                # Probe channel
                key = (
                    y,
                    x,
                    channel
                )


                flatfield = models.get(
                    key,
                    None
                )


                if flatfield is None:

                    print(
                        (
                            f"{array_path} "
                            f"| c={channel}: "
                            "NO MODEL -> unchanged"
                        )
                    )


                    apply_rows.append(
                        {
                            "zarr":
                                source_zarr.name,

                            "array":
                                array_path,

                            "channel":
                                channel,

                            "status":
                                "SKIPPED_NO_MODEL",

                            "planes":
                                0,
                        }
                    )


                    continue


                plane_count = 0


                for selector, plane_desc in (
                    generate_channel_planes(
                        shape,
                        axes,
                        channel
                    )
                ):

                    raw = np.asarray(
                        source_array[
                            selector
                        ],
                        dtype=np.float32
                    )


                    raw = np.squeeze(
                        raw
                    )


                    if raw.shape != (
                        y,
                        x
                    ):

                        raise RuntimeError(
                            (
                                "Plane shape mismatch:\n"
                                f"{raw.shape}"
                                " != "
                                f"{(y, x)}"
                            )
                        )


                    # ============================================
                    # Flat-field correction
                    #
                    # darkfield = OFF
                    #
                    # corrected = raw / flatfield
                    #
                    # 不 blur
                    # 不 background subtraction
                    # 不改变坐标

                    corrected = (
                        raw
                        /
                        flatfield
                    )


                    corrected = (
                        convert_to_original_dtype(
                            corrected,
                            source_array.dtype
                        )
                    )


                    destination_array[
                        selector
                    ] = corrected


                    plane_count += 1


                print(
                    (
                        f"{array_path} "
                        f"| c={channel}: "
                        "FLAT-FIELD APPLIED "
                        f"({plane_count} planes)"
                    )
                )


                apply_rows.append(
                    {
                        "zarr":
                            source_zarr.name,

                        "array":
                            array_path,

                        "channel":
                            channel,

                        "status":
                            "CORRECTED",

                        "planes":
                            plane_count,

                        "model":
                            (
                                f"YX{y}x{x}"
                                f"__ch{channel}"
                            ),
                    }
                )

    # 20. 保存报告
    model_df = pd.DataFrame(
        model_rows
    )


    model_df.to_csv(
        (
            output_dir
            /
            "flat_field_models.tsv"
        ),
        sep="\t",
        index=False
    )


    fit_df = pd.DataFrame(
        fit_sample_rows
    )


    fit_df.to_csv(
        (
            output_dir
            /
            "flat_field_fit_samples.tsv"
        ),
        sep="\t",
        index=False
    )


    apply_df = pd.DataFrame(
        apply_rows
    )


    apply_df.to_csv(
        (
            output_dir
            /
            "flat_field_apply_report.tsv"
        ),
        sep="\t",
        index=False
    )

    # 21. Summary
    summary_file = (
        output_dir
        /
        "FLAT_FIELD_SUMMARY.txt"
    )


    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(
            "Flat-field correction summary\n"
        )

        handle.write(
            "=============================\n\n"
        )


        handle.write(
            (
                f"Input: "
                f"{input_dir}\n"
            )
        )


        handle.write(
            (
                f"Output: "
                f"{output_dir}\n\n"
            )
        )


        handle.write(
            (
                "DAPI channel: "
                f"c={DAPI_CHANNEL}\n"
            )
        )

        handle.write(
            "DAPI correction: DISABLED\n\n"
        )


        handle.write(
            (
                "Probe channels: "
                "all channels except DAPI\n"
            )
        )

        handle.write(
            "Probe correction: ENABLED\n\n"
        )


        handle.write(
            (
                "smoothness_flatfield: "
                f"{BASIC_SMOOTHNESS_FLATFIELD}\n"
            )
        )


        handle.write(
            (
                "get_darkfield: "
                f"{BASIC_GET_DARKFIELD}\n"
            )
        )


        handle.write(
            (
                "Shared models generated: "
                f"{len(models)}\n"
            )
        )


    print(
        "\n========================================"
    )

    print(
        "FINISHED"
    )

    print(
        (
            "\nBaSiC "
            "smoothness_flatfield = "
            f"{BASIC_SMOOTHNESS_FLATFIELD}"
        )
    )


    print(
        (
            "\nOutput:\n"
            f"{output_dir}"
        )
    )


if __name__ == "__main__":

    main()
