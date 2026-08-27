#!/home/mshuai/.conda/envs/picture/bin/python

from pathlib import Path
import csv
import math
import re

import numpy as np
import pandas as pd
import zarr

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_laplace
from scipy.spatial import cKDTree

from skimage.feature import peak_local_max


# ============================================================
# 1. 路径
# ============================================================

RAW_DIR = Path(
    "/home/mshuai/transfer/OME/test"
)

CORRECTED_DIR = Path(
    "/home/mshuai/preprocessing/zarr-flat"
)

QC_DIR = Path(
    "/home/mshuai/preprocessing/QC_spots"
)

QC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OVERLAY_DIR = QC_DIR / "overlays"

OVERLAY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 只分析 Channel 2
#
# Python 从 0 开始：
#
# Ch0
# Ch1
# Ch2  ← 探针
# ============================================================

CHANNEL = 2


# ============================================================
# 3. Spot detection 参数
#
# 如果你的 spot 大约 2~4 pixel，
# sigma=1.2 通常是合理起点。
#
# 后面可以根据真实探针大小调整。
# ============================================================

LOG_SIGMA = 1.2

MIN_DISTANCE = 2

# LoG response:
#
# threshold =
# median + MAD_THRESHOLD × robust_sigma
#
MAD_THRESHOLD = 5.0


# ============================================================
# 4. Raw / corrected spot 匹配半径
#
# Flat-field 不应该改变坐标。
#
# 允许 detector 因强度变化产生
# 1~2 pixel peak shift。
# ============================================================

MATCH_RADIUS = 2.0


# ============================================================
# 5. SNR 参数
#
# 中心 spot：
# 半径 <= 2 pixel
#
# background：
# 4~7 pixel annulus
# ============================================================

SPOT_RADIUS = 2

BACKGROUND_INNER_RADIUS = 4

BACKGROUND_OUTER_RADIUS = 7


# ============================================================
# 6. 工程 QC 阈值
#
# 注意：
# 这些不是文献统一 cutoff，
# 是为了自动报警的保守值。
# ============================================================

MIN_MATCH_RECALL = 0.95

MAX_COUNT_LOSS = 0.10

MAX_MEDIAN_DISPLACEMENT = 1.0

MIN_MEDIAN_SNR_RATIO = 0.90


# ============================================================
# 7. 工具函数
# ============================================================

def safe_name(text):

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )


def join_path(parent, child):

    parent = str(parent).strip("/")
    child = str(child).strip("/")

    if parent:

        return f"{parent}/{child}"

    return child


# ============================================================
# 8. 读取 axes
# ============================================================

def parse_axes(multiscale, shape):

    axes_raw = multiscale.get(
        "axes",
        None,
    )

    axes = []

    if axes_raw:

        for axis in axes_raw:

            if isinstance(axis, str):

                axes.append(
                    axis.lower()
                )

            elif isinstance(axis, dict):

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


    # fallback
    if len(shape) == 5:

        return [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]


    raise RuntimeError(
        f"无法判断 axes: shape={shape}"
    )


# ============================================================
# 9. 找所有 full-resolution image units
#
# 例如：
#
# 0/0
# 1/0
# 2/0
#
# 只取 multiscale 第一个 dataset
# ============================================================

def discover_primary_arrays(root):

    results = []

    def recurse(group, group_path=""):

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


                # --------------------------------------------
                # 只取 full-resolution dataset
                # --------------------------------------------

                dataset = datasets[0]

                relative_path = str(
                    dataset.get(
                        "path",
                        ""
                    )
                )

                full_path = join_path(
                    group_path,
                    relative_path,
                )


                try:

                    array = root[
                        full_path
                    ]

                except Exception:

                    continue


                axes = parse_axes(
                    ms,
                    array.shape,
                )


                if (
                    "c" not in axes
                    or
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
                                array.shape
                            ),

                        "axes":
                            axes,
                    }
                )


        for name, subgroup in group.groups():

            recurse(
                subgroup,
                join_path(
                    group_path,
                    name,
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


# ============================================================
# 10. 生成 Ch2 的所有 XY plane
# ============================================================

def generate_ch2_planes(
    shape,
    axes,
):

    y_axis = axes.index(
        "y"
    )

    x_axis = axes.index(
        "x"
    )

    c_axis = axes.index(
        "c"
    )


    if CHANNEL >= shape[c_axis]:

        return


    varying_axes = []

    for i in range(
        len(shape)
    ):

        if i in (
            y_axis,
            x_axis,
            c_axis,
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

        selector[
            c_axis
        ] = CHANNEL


        description = []


        if sizes:

            coords = np.unravel_index(
                flat_index,
                sizes,
            )


            for axis_index, value in zip(
                varying_axes,
                coords,
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


# ============================================================
# 11. Spot detection
# ============================================================

def detect_spots(image):

    image = np.asarray(
        image,
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # LoG
    #
    # bright spot -> positive response
    # --------------------------------------------------------

    response = (
        -
        gaussian_laplace(
            image,
            sigma=
                LOG_SIGMA,
        )
    )


    # --------------------------------------------------------
    # Robust background estimate
    # --------------------------------------------------------

    median = float(
        np.median(
            response
        )
    )


    mad = float(
        np.median(
            np.abs(
                response
                -
                median
            )
        )
    )


    robust_sigma = (
        1.4826
        *
        mad
    )


    threshold = (
        median
        +
        MAD_THRESHOLD
        *
        robust_sigma
    )


    # --------------------------------------------------------
    # peak detection
    #
    # 返回：
    # y,x
    # --------------------------------------------------------

    coordinates = peak_local_max(
        response,

        min_distance=
            MIN_DISTANCE,

        threshold_abs=
            threshold,

        exclude_border=
            BACKGROUND_OUTER_RADIUS,
    )


    return (
        coordinates,
        response,
        threshold,
    )


# ============================================================
# 12. Raw / corrected spot matching
#
# mutual nearest-neighbor
#
# 避免两个 raw spot 匹配同一个 corrected spot
# ============================================================

def match_spots(
    raw_spots,
    corrected_spots,
):

    if (
        len(raw_spots) == 0
        or
        len(corrected_spots) == 0
    ):

        return []


    raw_tree = cKDTree(
        raw_spots
    )

    corr_tree = cKDTree(
        corrected_spots
    )


    dist_raw_to_corr, idx_corr = (
        corr_tree.query(
            raw_spots,
            k=1,
        )
    )


    dist_corr_to_raw, idx_raw = (
        raw_tree.query(
            corrected_spots,
            k=1,
        )
    )


    matches = []


    for raw_index in range(
        len(
            raw_spots
        )
    ):

        corrected_index = int(
            idx_corr[
                raw_index
            ]
        )


        distance = float(
            dist_raw_to_corr[
                raw_index
            ]
        )


        if distance > MATCH_RADIUS:

            continue


        # mutual nearest
        if (
            int(
                idx_raw[
                    corrected_index
                ]
            )
            !=
            raw_index
        ):

            continue


        matches.append(
            (
                raw_index,
                corrected_index,
                distance,
            )
        )


    return matches


# ============================================================
# 13. 单个 spot SNR
# ============================================================

def spot_snr(
    image,
    y,
    x,
):

    image = np.asarray(
        image,
        dtype=np.float32,
    )


    height, width = (
        image.shape
    )


    r = (
        BACKGROUND_OUTER_RADIUS
    )


    if (
        y - r < 0
        or
        y + r >= height
        or
        x - r < 0
        or
        x + r >= width
    ):

        return np.nan


    patch = image[
        y-r:y+r+1,
        x-r:x+r+1,
    ]


    yy, xx = np.ogrid[
        -r:r+1,
        -r:r+1
    ]


    distance = np.sqrt(
        yy**2
        +
        xx**2
    )


    # --------------------------------------------------------
    # Spot intensity
    # --------------------------------------------------------

    spot_mask = (
        distance
        <=
        SPOT_RADIUS
    )


    # 用 spot 内最大像素
    signal = float(
        np.max(
            patch[
                spot_mask
            ]
        )
    )


    # --------------------------------------------------------
    # Local background annulus
    # --------------------------------------------------------

    background_mask = (
        (
            distance
            >=
            BACKGROUND_INNER_RADIUS
        )
        &
        (
            distance
            <=
            BACKGROUND_OUTER_RADIUS
        )
    )


    background_values = patch[
        background_mask
    ]


    background_median = float(
        np.median(
            background_values
        )
    )


    background_mad = float(
        np.median(
            np.abs(
                background_values
                -
                background_median
            )
        )
    )


    noise = (
        1.4826
        *
        background_mad
    )


    if noise < 1e-6:

        noise = float(
            np.std(
                background_values
            )
        )


    if noise < 1e-6:

        return np.nan


    snr = (
        signal
        -
        background_median
    ) / noise


    return float(
        snr
    )


# ============================================================
# 14. Overlay figure
# ============================================================

def save_overlay(
    raw,
    corrected,
    raw_spots,
    corrected_spots,
    matches,
    output_file,
    title,
):

    # --------------------------------------------------------
    # 同一个显示范围
    # --------------------------------------------------------

    raw_p1 = np.percentile(
        raw,
        1
    )

    corr_p1 = np.percentile(
        corrected,
        1
    )

    raw_p998 = np.percentile(
        raw,
        99.8
    )

    corr_p998 = np.percentile(
        corrected,
        99.8
    )


    vmin = min(
        raw_p1,
        corr_p1,
    )

    vmax = max(
        raw_p998,
        corr_p998,
    )


    # ========================================================
    # Figure 1: Raw
    # ========================================================

    fig = plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        raw,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )


    if len(raw_spots):

        plt.scatter(
            raw_spots[:, 1],
            raw_spots[:, 0],
            s=18,
            facecolors="none",
            edgecolors="red",
            linewidths=0.7,
        )


    plt.title(
        (
            title
            +
            f"\nRAW Ch2 spots = {len(raw_spots)}"
        )
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    raw_file = output_file.with_name(
        output_file.stem
        +
        "_RAW.png"
    )

    plt.savefig(
        raw_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # ========================================================
    # Figure 2: Corrected
    # ========================================================

    fig = plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        corrected,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )


    if len(corrected_spots):

        plt.scatter(
            corrected_spots[:, 1],
            corrected_spots[:, 0],
            s=18,
            facecolors="none",
            edgecolors="lime",
            linewidths=0.7,
        )


    plt.title(
        (
            title
            +
            f"\nCORRECTED Ch2 spots = "
            f"{len(corrected_spots)}"
        )
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    corrected_file = output_file.with_name(
        output_file.stem
        +
        "_CORRECTED.png"
    )

    plt.savefig(
        corrected_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # ========================================================
    # Figure 3: matched spot positions
    #
    # 红圈 = RAW
    # 绿色叉 = Corrected
    #
    # 如果位置不变，会基本重叠。
    # ========================================================

    fig = plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        raw,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )


    for (
        raw_index,
        corrected_index,
        distance,
    ) in matches:

        y1, x1 = (
            raw_spots[
                raw_index
            ]
        )

        y2, x2 = (
            corrected_spots[
                corrected_index
            ]
        )


        plt.plot(
            [
                x1,
                x2
            ],
            [
                y1,
                y2
            ],
            linewidth=0.6,
        )


    if matches:

        matched_raw = np.array(
            [
                raw_spots[
                    m[0]
                ]
                for m in matches
            ]
        )


        matched_corr = np.array(
            [
                corrected_spots[
                    m[1]
                ]
                for m in matches
            ]
        )


        plt.scatter(
            matched_raw[:, 1],
            matched_raw[:, 0],
            s=25,
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
            label="Raw",
        )


        plt.scatter(
            matched_corr[:, 1],
            matched_corr[:, 0],
            s=14,
            marker="x",
            label="Corrected",
        )


        plt.legend(
            loc="upper right"
        )


    plt.title(
        (
            title
            +
            f"\nMatched spots = {len(matches)}"
        )
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    match_file = output_file.with_name(
        output_file.stem
        +
        "_MATCH.png"
    )

    plt.savefig(
        match_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# 15. 主程序
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "Probe Spot QC — Channel 2"
    )

    print(
        "========================================"
    )


    raw_zarrs = sorted(
        RAW_DIR.glob(
            "*.zarr"
        )
    )


    all_rows = []

    all_spot_rows = []


    for raw_zarr in raw_zarrs:

        corrected_zarr = (
            CORRECTED_DIR
            /
            raw_zarr.name
        )


        if not corrected_zarr.exists():

            print(
                (
                    "\n跳过：没有 corrected Zarr\n"
                    f"{raw_zarr.name}"
                )
            )

            continue


        print(
            "\n"
            "----------------------------------------"
        )

        print(
            raw_zarr.name
        )

        print(
            "----------------------------------------"
        )


        raw_root = zarr.open_group(
            str(
                raw_zarr
            ),
            mode="r",
        )


        corrected_root = zarr.open_group(
            str(
                corrected_zarr
            ),
            mode="r",
        )


        arrays = discover_primary_arrays(
            raw_root
        )


        for info in arrays:

            array_path = info[
                "path"
            ]


            if array_path not in corrected_root:

                continue


            raw_array = raw_root[
                array_path
            ]

            corrected_array = corrected_root[
                array_path
            ]


            if (
                raw_array.shape
                !=
                corrected_array.shape
            ):

                print(
                    (
                        "Shape mismatch: "
                        f"{array_path}"
                    )
                )

                continue


            for selector, plane_desc in (
                generate_ch2_planes(
                    info[
                        "shape"
                    ],
                    info[
                        "axes"
                    ],
                )
            ):

                raw = np.asarray(
                    raw_array[
                        selector
                    ],
                    dtype=np.float32,
                )


                corrected = np.asarray(
                    corrected_array[
                        selector
                    ],
                    dtype=np.float32,
                )


                raw = np.squeeze(
                    raw
                )

                corrected = np.squeeze(
                    corrected
                )


                # =================================================
                # Detect
                # =================================================

                (
                    raw_spots,
                    raw_response,
                    raw_threshold,
                ) = detect_spots(
                    raw
                )


                (
                    corr_spots,
                    corr_response,
                    corr_threshold,
                ) = detect_spots(
                    corrected
                )


                # =================================================
                # Match
                # =================================================

                matches = match_spots(
                    raw_spots,
                    corr_spots,
                )


                raw_count = len(
                    raw_spots
                )

                corr_count = len(
                    corr_spots
                )


                if raw_count > 0:

                    count_ratio = (
                        corr_count
                        /
                        raw_count
                    )

                    count_change = (
                        (
                            corr_count
                            -
                            raw_count
                        )
                        /
                        raw_count
                    )

                    match_recall = (
                        len(
                            matches
                        )
                        /
                        raw_count
                    )

                else:

                    count_ratio = np.nan
                    count_change = np.nan
                    match_recall = np.nan


                if corr_count > 0:

                    match_precision = (
                        len(
                            matches
                        )
                        /
                        corr_count
                    )

                else:

                    match_precision = np.nan


                # =================================================
                # Position displacement
                # =================================================

                distances = [
                    m[2]
                    for m in matches
                ]


                if distances:

                    median_displacement = float(
                        np.median(
                            distances
                        )
                    )

                    p95_displacement = float(
                        np.percentile(
                            distances,
                            95
                        )
                    )

                else:

                    median_displacement = np.nan
                    p95_displacement = np.nan


                # =================================================
                # SNR of matched spots
                # =================================================

                raw_snrs = []

                corr_snrs = []


                for (
                    raw_index,
                    corr_index,
                    distance,
                ) in matches:

                    raw_y, raw_x = (
                        raw_spots[
                            raw_index
                        ]
                    )

                    corr_y, corr_x = (
                        corr_spots[
                            corr_index
                        ]
                    )


                    snr_raw = spot_snr(
                        raw,
                        int(
                            raw_y
                        ),
                        int(
                            raw_x
                        ),
                    )


                    snr_corr = spot_snr(
                        corrected,
                        int(
                            corr_y
                        ),
                        int(
                            corr_x
                        ),
                    )


                    if (
                        np.isfinite(
                            snr_raw
                        )
                        and
                        np.isfinite(
                            snr_corr
                        )
                    ):

                        raw_snrs.append(
                            snr_raw
                        )

                        corr_snrs.append(
                            snr_corr
                        )


                        all_spot_rows.append(
                            {
                                "zarr":
                                    raw_zarr.name,

                                "array":
                                    array_path,

                                "plane":
                                    plane_desc,

                                "raw_y":
                                    int(
                                        raw_y
                                    ),

                                "raw_x":
                                    int(
                                        raw_x
                                    ),

                                "corrected_y":
                                    int(
                                        corr_y
                                    ),

                                "corrected_x":
                                    int(
                                        corr_x
                                    ),

                                "displacement_px":
                                    float(
                                        distance
                                    ),

                                "snr_raw":
                                    float(
                                        snr_raw
                                    ),

                                "snr_corrected":
                                    float(
                                        snr_corr
                                    ),

                                "snr_ratio":
                                    float(
                                        snr_corr
                                        /
                                        max(
                                            snr_raw,
                                            1e-6,
                                        )
                                    ),
                            }
                        )


                if raw_snrs:

                    median_snr_raw = float(
                        np.median(
                            raw_snrs
                        )
                    )

                    median_snr_corr = float(
                        np.median(
                            corr_snrs
                        )
                    )


                    median_snr_ratio = float(
                        np.median(
                            np.array(
                                corr_snrs
                            )
                            /
                            np.maximum(
                                np.array(
                                    raw_snrs
                                ),
                                1e-6,
                            )
                        )
                    )

                else:

                    median_snr_raw = np.nan
                    median_snr_corr = np.nan
                    median_snr_ratio = np.nan


                # =================================================
                # Decisions
                # =================================================

                position_pass = (
                    np.isfinite(
                        match_recall
                    )
                    and
                    match_recall
                    >=
                    MIN_MATCH_RECALL
                    and
                    np.isfinite(
                        median_displacement
                    )
                    and
                    median_displacement
                    <=
                    MAX_MEDIAN_DISPLACEMENT
                )


                count_pass = (
                    np.isfinite(
                        count_change
                    )
                    and
                    count_change
                    >=
                    -
                    MAX_COUNT_LOSS
                )


                snr_pass = (
                    np.isfinite(
                        median_snr_ratio
                    )
                    and
                    median_snr_ratio
                    >=
                    MIN_MEDIAN_SNR_RATIO
                )


                final_pass = (
                    position_pass
                    and
                    count_pass
                    and
                    snr_pass
                )


                decision = (
                    "YES"
                    if final_pass
                    else "NO"
                )


                # =================================================
                # Save row
                # =================================================

                all_rows.append(
                    {
                        "zarr":
                            raw_zarr.name,

                        "array":
                            array_path,

                        "plane":
                            plane_desc,

                        "channel":
                            CHANNEL,

                        "raw_spot_count":
                            raw_count,

                        "corrected_spot_count":
                            corr_count,

                        "count_ratio":
                            count_ratio,

                        "count_change_percent":
                            (
                                count_change
                                *
                                100
                                if np.isfinite(
                                    count_change
                                )
                                else np.nan
                            ),

                        "matched_spots":
                            len(
                                matches
                            ),

                        "match_recall":
                            match_recall,

                        "match_precision":
                            match_precision,

                        "median_displacement_px":
                            median_displacement,

                        "p95_displacement_px":
                            p95_displacement,

                        "median_snr_raw":
                            median_snr_raw,

                        "median_snr_corrected":
                            median_snr_corr,

                        "median_snr_ratio":
                            median_snr_ratio,

                        "position_pass":
                            (
                                "YES"
                                if position_pass
                                else "NO"
                            ),

                        "count_pass":
                            (
                                "YES"
                                if count_pass
                                else "NO"
                            ),

                        "snr_pass":
                            (
                                "YES"
                                if snr_pass
                                else "NO"
                            ),

                        "decision":
                            decision,
                    }
                )


                # =================================================
                # Print
                # =================================================

                print(
                    (
                        f"\n{array_path} | "
                        f"{plane_desc}"
                    )
                )

                print(
                    (
                        f"  Spots: "
                        f"{raw_count} -> "
                        f"{corr_count}"
                    )
                )

                print(
                    (
                        f"  Matched recall: "
                        f"{match_recall:.3f}"
                        if np.isfinite(
                            match_recall
                        )
                        else
                        "  Matched recall: NA"
                    )
                )

                print(
                    (
                        f"  Median displacement: "
                        f"{median_displacement:.3f} px"
                        if np.isfinite(
                            median_displacement
                        )
                        else
                        "  Median displacement: NA"
                    )
                )

                print(
                    (
                        f"  Median SNR: "
                        f"{median_snr_raw:.2f}"
                        f" -> "
                        f"{median_snr_corr:.2f}"
                        if (
                            np.isfinite(
                                median_snr_raw
                            )
                            and
                            np.isfinite(
                                median_snr_corr
                            )
                        )
                        else
                        "  Median SNR: NA"
                    )
                )

                print(
                    (
                        f"  Decision: "
                        f"{decision}"
                    )
                )


                # =================================================
                # Overlay
                # =================================================

                base_name = (
                    safe_name(
                        raw_zarr.stem
                    )
                    +
                    "__"
                    +
                    safe_name(
                        array_path
                    )
                    +
                    "__"
                    +
                    safe_name(
                        plane_desc
                    )
                )


                save_overlay(
                    raw,
                    corrected,
                    raw_spots,
                    corr_spots,
                    matches,
                    OVERLAY_DIR
                    /
                    (
                        base_name
                        +
                        ".png"
                    ),
                    (
                        raw_zarr.name
                        +
                        "\n"
                        +
                        array_path
                        +
                        " | "
                        +
                        plane_desc
                        +
                        " | Ch2"
                    ),
                )


    # ========================================================
    # 16. 保存 summary TSV
    # ========================================================

    summary_df = pd.DataFrame(
        all_rows
    )


    summary_file = (
        QC_DIR
        /
        "probe_spot_summary.tsv"
    )


    summary_df.to_csv(
        summary_file,
        sep="\t",
        index=False,
    )


    # ========================================================
    # 17. 每个 matched spot 的详细数据
    # ========================================================

    spot_df = pd.DataFrame(
        all_spot_rows
    )


    spot_file = (
        QC_DIR
        /
        "matched_spots.tsv"
    )


    spot_df.to_csv(
        spot_file,
        sep="\t",
        index=False,
    )


    # ========================================================
    # 18. 全局 summary
    # ========================================================

    if len(summary_df) > 0:

        total_raw = int(
            summary_df[
                "raw_spot_count"
            ].sum()
        )


        total_corrected = int(
            summary_df[
                "corrected_spot_count"
            ].sum()
        )


        total_matched = int(
            summary_df[
                "matched_spots"
            ].sum()
        )


        global_recall = (
            total_matched
            /
            total_raw
            if total_raw > 0
            else np.nan
        )


        global_count_ratio = (
            total_corrected
            /
            total_raw
            if total_raw > 0
            else np.nan
        )


        if len(spot_df) > 0:

            median_global_displacement = float(
                spot_df[
                    "displacement_px"
                ].median()
            )


            median_global_snr_ratio = float(
                spot_df[
                    "snr_ratio"
                ].median()
            )

        else:

            median_global_displacement = np.nan

            median_global_snr_ratio = np.nan


        global_position_pass = (
            np.isfinite(
                global_recall
            )
            and
            global_recall
            >=
            MIN_MATCH_RECALL
            and
            np.isfinite(
                median_global_displacement
            )
            and
            median_global_displacement
            <=
            MAX_MEDIAN_DISPLACEMENT
        )


        global_count_pass = (
            np.isfinite(
                global_count_ratio
            )
            and
            global_count_ratio
            >=
            (
                1
                -
                MAX_COUNT_LOSS
            )
        )


        global_snr_pass = (
            np.isfinite(
                median_global_snr_ratio
            )
            and
            median_global_snr_ratio
            >=
            MIN_MEDIAN_SNR_RATIO
        )


        overall = (
            "YES"
            if (
                global_position_pass
                and
                global_count_pass
                and
                global_snr_pass
            )
            else
            "NO"
        )


    else:

        total_raw = 0

        total_corrected = 0

        total_matched = 0

        global_recall = np.nan

        global_count_ratio = np.nan

        median_global_displacement = np.nan

        median_global_snr_ratio = np.nan

        overall = "NO"


    # ========================================================
    # 19. 保存最终结果
    # ========================================================

    decision_file = (
        QC_DIR
        /
        "PROBE_SPOT_DECISION.txt"
    )


    with open(
        decision_file,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Probe spot preservation QC — Channel 2\n"
        )

        handle.write(
            "======================================\n\n"
        )

        handle.write(
            f"FINAL DECISION: {overall}\n\n"
        )


        handle.write(
            "1. Spot position preservation\n"
        )

        handle.write(
            (
                f"Matched recall: "
                f"{global_recall:.4f}\n"
                if np.isfinite(
                    global_recall
                )
                else
                "Matched recall: NA\n"
            )
        )

        handle.write(
            (
                f"Median displacement: "
                f"{median_global_displacement:.4f} px\n\n"
                if np.isfinite(
                    median_global_displacement
                )
                else
                "Median displacement: NA\n\n"
            )
        )


        handle.write(
            "2. Spot count\n"
        )

        handle.write(
            (
                f"Raw spots: "
                f"{total_raw}\n"
            )
        )

        handle.write(
            (
                f"Corrected spots: "
                f"{total_corrected}\n"
            )
        )

        handle.write(
            (
                f"Corrected / Raw: "
                f"{global_count_ratio:.4f}\n\n"
                if np.isfinite(
                    global_count_ratio
                )
                else
                "Corrected / Raw: NA\n\n"
            )
        )


        handle.write(
            "3. Spot SNR\n"
        )

        handle.write(
            (
                f"Median corrected/raw SNR ratio: "
                f"{median_global_snr_ratio:.4f}\n\n"
                if np.isfinite(
                    median_global_snr_ratio
                )
                else
                "Median corrected/raw SNR ratio: NA\n\n"
            )
        )


        handle.write(
            "Engineering thresholds\n"
        )

        handle.write(
            (
                f"- match recall >= "
                f"{MIN_MATCH_RECALL}\n"
            )
        )

        handle.write(
            (
                f"- count loss <= "
                f"{MAX_COUNT_LOSS * 100:.1f}%\n"
            )
        )

        handle.write(
            (
                f"- median displacement <= "
                f"{MAX_MEDIAN_DISPLACEMENT} px\n"
            )
        )

        handle.write(
            (
                f"- median SNR ratio >= "
                f"{MIN_MEDIAN_SNR_RATIO}\n"
            )
        )


    # ========================================================
    # 20. Count summary figure
    # ========================================================

    if len(summary_df) > 0:

        fig = plt.figure(
            figsize=(8, 6)
        )


        values = [
            total_raw,
            total_corrected,
        ]


        plt.bar(
            [
                "Raw",
                "Corrected"
            ],
            values,
        )


        plt.ylabel(
            "Detected Ch2 spots"
        )

        plt.title(
            "Probe spot count"
        )

        plt.tight_layout()


        plt.savefig(
            QC_DIR
            /
            "spot_count_raw_vs_corrected.png",
            dpi=180,
        )


        plt.close(
            fig
        )


    # ========================================================
    # 21. SNR scatter
    # ========================================================

    if len(spot_df) > 0:

        fig = plt.figure(
            figsize=(7, 7)
        )


        x = spot_df[
            "snr_raw"
        ].to_numpy()


        y = spot_df[
            "snr_corrected"
        ].to_numpy()


        plt.scatter(
            x,
            y,
            s=8,
            alpha=0.4,
        )


        maximum = np.nanpercentile(
            np.concatenate(
                [
                    x,
                    y
                ]
            ),
            99,
        )


        plt.plot(
            [
                0,
                maximum
            ],
            [
                0,
                maximum
            ],
            linestyle="--",
        )


        plt.xlim(
            0,
            maximum
        )

        plt.ylim(
            0,
            maximum
        )


        plt.xlabel(
            "Raw spot SNR"
        )

        plt.ylabel(
            "Corrected spot SNR"
        )

        plt.title(
            "Probe spot SNR preservation"
        )


        plt.tight_layout()


        plt.savefig(
            QC_DIR
            /
            "spot_snr_raw_vs_corrected.png",
            dpi=180,
        )


        plt.close(
            fig
        )


    # ========================================================
    # 22. displacement histogram
    # ========================================================

    if len(spot_df) > 0:

        fig = plt.figure(
            figsize=(8, 6)
        )


        plt.hist(
            spot_df[
                "displacement_px"
            ],
            bins=30,
        )


        plt.xlabel(
            "Raw → corrected spot displacement (pixels)"
        )

        plt.ylabel(
            "Number of matched spots"
        )

        plt.title(
            "Probe spot position preservation"
        )


        plt.tight_layout()


        plt.savefig(
            QC_DIR
            /
            "spot_displacement.png",
            dpi=180,
        )


        plt.close(
            fig
        )


    print(
        "\n"
        "========================================"
    )

    print(
        "FINISHED"
    )

    print(
        "========================================"
    )

    print(
        (
            "\nFINAL DECISION: "
            f"{overall}"
        )
    )

    print(
        (
            "\nRaw spots: "
            f"{total_raw}"
        )
    )

    print(
        (
            "Corrected spots: "
            f"{total_corrected}"
        )
    )


    if np.isfinite(
        global_recall
    ):

        print(
            (
                "Matched recall: "
                f"{global_recall:.4f}"
            )
        )


    if np.isfinite(
        median_global_displacement
    ):

        print(
            (
                "Median displacement: "
                f"{median_global_displacement:.4f} px"
            )
        )


    if np.isfinite(
        median_global_snr_ratio
    ):

        print(
            (
                "Median SNR ratio: "
                f"{median_global_snr_ratio:.4f}"
            )
        )


    print(
        (
            "\nQC output:\n"
            f"{QC_DIR}"
        )
    )


if __name__ == "__main__":

    main()
