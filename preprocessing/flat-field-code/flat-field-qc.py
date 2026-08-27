#!/home/mshuai/.conda/envs/picture/bin/python

from pathlib import Path
import html
import math
import re
import sys

import numpy as np
import pandas as pd
import zarr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity


# ============================================================
# 1. 路径
# ============================================================

RAW_ROOT = (
    Path.home()
    / "transfer"
    / "OME"
    / "test"
)

CORRECTED_ROOT = (
    Path.home()
    / "preprocessing"
    / "zarr-flat"
)

QC_ROOT = (
    Path.home()
    / "preprocessing"
    / "QC"
)

MODEL_ROOT = (
    CORRECTED_ROOT
    / "_flatfield_models"
)


# ============================================================
# 2. QC 参数
# ============================================================

# 每个 Zarr、每个 channel 最多抽多少张 XY plane 做定量 QC
MAX_METRIC_PLANES_PER_CHANNEL = 10

# 每个 Zarr、每个 channel 最多输出多少张比较图片
MAX_PNG_PLANES_PER_CHANNEL = 3

# QC 时，如果图片特别大，下采样到最长边不超过这个值
# 只影响 QC 计算和画图，不修改真实数据
MAX_QC_SIDE = 1024


# ============================================================
# 3. 自动 YES / NO 阈值
#
# 这些不是领域统一的“官方硬阈值”，
# 是为了给你的 pipeline 一个保守的自动 gate。
# 后面有真实数据后可以再调整。
# ============================================================

# 原图和校正图的高频结构相关性
MIN_HIGH_PASS_CORR = 0.95

# 结构相似性
MIN_SSIM = 0.90

# 校正后的低频空间不均匀性，
# 最多允许比原图增加 10%
MAX_LOWFREQ_RATIO = 1.10

# 饱和像素最多允许增加 0.5%
MAX_SATURATION_INCREASE = 0.005

# flatfield 应该是一个平滑场。
# 高频残差 RMS / median(flatfield)
MAX_MODEL_HIGH_FREQ_RMS = 0.03
#！！！！！！！！！！！！！！！！！！！！！！！！！！0.03

# flatfield p95 / p05 不应该极端到离谱
MAX_MODEL_DYNAMIC_RANGE = 4.0
#！！！！！！！！！！！！！！！！！！！！！！！4.0


# ============================================================
# 4. 输出目录
# ============================================================

COMPARISON_DIR = QC_ROOT / "comparisons"
MODEL_QC_DIR = QC_ROOT / "models"
SUMMARY_DIR = QC_ROOT / "summary"
TABLE_DIR = QC_ROOT / "tables"


# ============================================================
# 工具
# ============================================================

def ensure_directories():

    for path in (
        QC_ROOT,
        COMPARISON_DIR,
        MODEL_QC_DIR,
        SUMMARY_DIR,
        TABLE_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def safe_name(text):

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )


def join_zarr_path(base, child):

    if not base:
        return str(child).strip("/")

    return (
        f"{base}/{child}"
        .strip("/")
    )


# ============================================================
# OME-Zarr metadata
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
                            "",
                        )
                    ).lower()
                )

    if len(axes) == len(shape):
        return axes

    # --------------------------------------------------------
    # fallback
    # --------------------------------------------------------

    ndim = len(shape)

    if ndim == 2:
        return ["y", "x"]

    if ndim == 3:

        if shape[0] <= 8:
            return ["c", "y", "x"]

        return ["z", "y", "x"]

    if ndim == 4:

        if shape[0] <= 8:
            return [
                "c",
                "z",
                "y",
                "x",
            ]

        return [
            "t",
            "z",
            "y",
            "x",
        ]

    if ndim == 5:
        return [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]

    axes = [
        f"d{i}"
        for i in range(ndim)
    ]

    axes[-2] = "y"
    axes[-1] = "x"

    return axes


def get_channel_labels(group, root, channel_count):

    candidates = [
        group,
        root,
    ]

    for candidate in candidates:

        try:
            omero = candidate.attrs.get(
                "omero"
            )
        except Exception:
            omero = None

        if not omero:
            continue

        channels = omero.get(
            "channels",
            []
        )

        labels = []

        for i in range(channel_count):

            if i < len(channels):

                label = channels[i].get(
                    "label"
                )

                if label:
                    labels.append(
                        str(label)
                    )
                    continue

            labels.append(
                f"channel_{i}"
            )

        return labels

    return [
        f"channel_{i}"
        for i in range(channel_count)
    ]


def get_voxel_size(multiscale, dataset, axes):

    result = {}

    transformations = dataset.get(
        "coordinateTransformations",
        []
    )

    scale = None

    for transform in transformations:

        if (
            transform.get("type")
            == "scale"
        ):
            scale = transform.get(
                "scale"
            )
            break

    if (
        scale is None
        or len(scale) != len(axes)
    ):
        return result

    axes_raw = multiscale.get(
        "axes",
        []
    )

    for i, axis in enumerate(axes):

        if axis not in (
            "x",
            "y",
            "z",
        ):
            continue

        unit = ""

        if (
            i < len(axes_raw)
            and isinstance(
                axes_raw[i],
                dict,
            )
        ):
            unit = axes_raw[i].get(
                "unit",
                ""
            )

        result[axis] = (
            scale[i],
            unit,
        )

    return result


# ============================================================
# 找最高分辨率 level
# ============================================================

def discover_primary_images(root):
    """
    只读取 OME-Zarr multiscale 中的第一个 dataset，
    即通常的最高分辨率 level 0。

    不会把：
        0
        1
        2
        3
    全部重复拿来 QC。
    """

    results = []

    def recurse(group, base_path=""):

        try:
            multiscales = group.attrs.get(
                "multiscales"
            )
        except Exception:
            multiscales = None

        if multiscales:

            for ms_index, multiscale in enumerate(
                multiscales
            ):

                datasets = multiscale.get(
                    "datasets",
                    []
                )

                if not datasets:
                    continue

                # --------------------------------------------
                # OME-Zarr 中第一个通常是 full resolution
                # --------------------------------------------

                dataset = datasets[0]

                relative_path = str(
                    dataset["path"]
                )

                full_path = join_zarr_path(
                    base_path,
                    relative_path,
                )

                try:
                    array = root[
                        full_path
                    ]
                except Exception:
                    continue

                if not isinstance(
                    array,
                    zarr.Array,
                ):
                    continue

                if array.ndim < 2:
                    continue

                axes = parse_axes(
                    multiscale,
                    array.shape,
                )

                if (
                    "x" not in axes
                    or "y" not in axes
                ):
                    continue

                if "c" in axes:

                    c_axis = axes.index(
                        "c"
                    )

                    channel_count = (
                        array.shape[c_axis]
                    )

                else:

                    channel_count = 1

                labels = get_channel_labels(
                    group,
                    root,
                    channel_count,
                )

                voxel_size = get_voxel_size(
                    multiscale,
                    dataset,
                    axes,
                )

                results.append(
                    {
                        "path": full_path,
                        "shape": tuple(
                            array.shape
                        ),
                        "dtype": str(
                            array.dtype
                        ),
                        "axes": axes,
                        "channels":
                            channel_count,
                        "channel_labels":
                            labels,
                        "voxel_size":
                            voxel_size,
                    }
                )

        for name, subgroup in group.groups():

            recurse(
                subgroup,
                join_zarr_path(
                    base_path,
                    name,
                )
            )

    recurse(root)

    # --------------------------------------------------------
    # 防止重复
    # --------------------------------------------------------

    unique = {}

    for item in results:
        unique[item["path"]] = item

    results = list(
        unique.values()
    )

    # --------------------------------------------------------
    # fallback：
    # 如果不是规范 OME-Zarr，找最大的图像 array
    # --------------------------------------------------------

    if not results:

        candidates = []

        def walk(group, base=""):

            for name, array in group.arrays():

                path = join_zarr_path(
                    base,
                    name,
                )

                if (
                    array.ndim >= 2
                    and array.dtype.kind in "uif"
                    and array.shape[-1] >= 32
                    and array.shape[-2] >= 32
                ):

                    size = int(
                        np.prod(
                            array.shape
                        )
                    )

                    candidates.append(
                        (
                            size,
                            path,
                            array,
                        )
                    )

            for name, subgroup in group.groups():

                walk(
                    subgroup,
                    join_zarr_path(
                        base,
                        name,
                    )
                )

        walk(root)

        if candidates:

            candidates.sort(
                reverse=True,
                key=lambda x: x[0],
            )

            _, path, array = (
                candidates[0]
            )

            dummy_multiscale = {}

            axes = parse_axes(
                dummy_multiscale,
                array.shape,
            )

            if "c" in axes:

                c_axis = axes.index(
                    "c"
                )

                channel_count = (
                    array.shape[c_axis]
                )

            else:
                channel_count = 1

            results.append(
                {
                    "path": path,
                    "shape":
                        tuple(array.shape),
                    "dtype":
                        str(array.dtype),
                    "axes":
                        axes,
                    "channels":
                        channel_count,
                    "channel_labels":
                        [
                            f"channel_{i}"
                            for i in range(
                                channel_count
                            )
                        ],
                    "voxel_size":
                        {},
                }
            )

    return results


# ============================================================
# XY plane 选择
# ============================================================

def get_plane_selectors(
    shape,
    axes,
    channel,
    max_planes,
):
    """
    返回：
        selector
        plane_description
    """

    y_axis = axes.index("y")
    x_axis = axes.index("x")

    c_axis = (
        axes.index("c")
        if "c" in axes
        else None
    )

    other_axes = []

    for axis_index in range(
        len(shape)
    ):

        if axis_index in (
            y_axis,
            x_axis,
        ):
            continue

        if (
            c_axis is not None
            and axis_index == c_axis
        ):
            continue

        other_axes.append(
            axis_index
        )

    sizes = [
        shape[i]
        for i in other_axes
    ]

    if sizes:
        total = int(
            np.prod(sizes)
        )
    else:
        total = 1

    count = min(
        total,
        max_planes,
    )

    flat_indices = np.unique(
        np.linspace(
            0,
            total - 1,
            count,
            dtype=int,
        )
    )

    output = []

    for flat_index in flat_indices:

        selector = [
            0
        ] * len(shape)

        selector[y_axis] = (
            slice(None)
        )

        selector[x_axis] = (
            slice(None)
        )

        if c_axis is not None:
            selector[c_axis] = (
                channel
            )

        description = []

        if sizes:

            coords = np.unravel_index(
                flat_index,
                sizes,
            )

            for axis_index, value in zip(
                other_axes,
                coords,
            ):

                selector[
                    axis_index
                ] = int(value)

                description.append(
                    f"{axes[axis_index]}="
                    f"{int(value)}"
                )

        if not description:
            description = [
                "2D"
            ]

        output.append(
            (
                tuple(selector),
                ",".join(
                    description
                ),
            )
        )

    return output


# ============================================================
# 图像处理，仅用于 QC
# ============================================================

def load_plane(array, selector):

    image = np.asarray(
        array[selector],
        dtype=np.float32,
    )

    image = np.squeeze(
        image
    )

    if image.ndim != 2:

        raise ValueError(
            f"不是二维 XY plane: "
            f"{image.shape}"
        )

    return image


def downsample_qc(image):

    h, w = image.shape

    step = max(
        1,
        int(
            math.ceil(
                max(h, w)
                / MAX_QC_SIDE
            )
        )
    )

    return (
        image[::step, ::step],
        step,
    )


def robust_normalize(image):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    finite = np.isfinite(
        image
    )

    if not np.any(finite):
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    values = image[finite]

    p1, p99 = np.percentile(
        values,
        [1, 99],
    )

    if p99 <= p1:

        p1 = float(
            np.min(values)
        )

        p99 = float(
            np.max(values)
        )

    if p99 <= p1:

        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    output = (
        image - p1
    ) / (
        p99 - p1
    )

    output = np.clip(
        output,
        0,
        1,
    )

    output[
        ~np.isfinite(output)
    ] = 0

    return output


def pearson_corr(a, b):

    a = np.asarray(
        a,
        dtype=np.float64,
    ).ravel()

    b = np.asarray(
        b,
        dtype=np.float64,
    ).ravel()

    if (
        np.std(a) == 0
        or np.std(b) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def low_frequency_cv(image):

    # 根据图像大小自动选择 sigma

    sigma = max(
        8,
        min(image.shape) / 32,
    )

    smooth = gaussian_filter(
        image,
        sigma=sigma,
    )

    mean = float(
        np.mean(smooth)
    )

    if mean <= 1e-8:
        return np.nan, smooth

    cv = float(
        np.std(smooth)
        / mean
    )

    return cv, smooth


def edge_center_bias(image):

    h, w = image.shape

    border_y = max(
        1,
        int(h * 0.15)
    )

    border_x = max(
        1,
        int(w * 0.15)
    )

    mask_border = np.zeros(
        (h, w),
        dtype=bool,
    )

    mask_border[
        :border_y,
        :
    ] = True

    mask_border[
        -border_y:,
        :
    ] = True

    mask_border[
        :,
        :border_x
    ] = True

    mask_border[
        :,
        -border_x:
    ] = True

    y1 = int(
        h * 0.25
    )

    y2 = int(
        h * 0.75
    )

    x1 = int(
        w * 0.25
    )

    x2 = int(
        w * 0.75
    )

    center = image[
        y1:y2,
        x1:x2,
    ]

    border_mean = float(
        np.mean(
            image[mask_border]
        )
    )

    center_mean = float(
        np.mean(center)
    )

    global_mean = float(
        np.mean(image)
    )

    if global_mean <= 1e-8:
        return np.nan

    return abs(
        border_mean
        - center_mean
    ) / global_mean


def saturation_fraction(
    image,
    dtype,
):

    dtype = np.dtype(
        dtype
    )

    if dtype.kind not in "ui":
        return np.nan

    max_value = np.iinfo(
        dtype
    ).max

    return float(
        np.mean(
            image >= max_value
        )
    )


# ============================================================
# 单张 XY 的指标
# ============================================================

def calculate_metrics(
    raw,
    corrected,
    dtype,
):
    """
    评价两个核心问题：

    1. illumination 是否变得更均匀
    2. 原始高频生物结构有没有被错误删除
    """

    raw_small, step = (
        downsample_qc(raw)
    )

    corrected_small, _ = (
        downsample_qc(
            corrected
        )
    )

    raw_norm = robust_normalize(
        raw_small
    )

    corrected_norm = (
        robust_normalize(
            corrected_small
        )
    )

    # --------------------------------------------------------
    # low-frequency shading
    # --------------------------------------------------------

    low_raw, smooth_raw = (
        low_frequency_cv(
            raw_norm
        )
    )

    low_corrected, smooth_corrected = (
        low_frequency_cv(
            corrected_norm
        )
    )

    # --------------------------------------------------------
    # high-frequency structure preservation
    # --------------------------------------------------------

    sigma_hp = 1.5

    raw_hp = (
        raw_norm
        - gaussian_filter(
            raw_norm,
            sigma=sigma_hp,
        )
    )

    corrected_hp = (
        corrected_norm
        - gaussian_filter(
            corrected_norm,
            sigma=sigma_hp,
        )
    )

    hp_corr = pearson_corr(
        raw_hp,
        corrected_hp,
    )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    try:

        ssim_value = float(
            structural_similarity(
                raw_norm,
                corrected_norm,
                data_range=1.0,
            )
        )

    except Exception:

        ssim_value = np.nan

    # --------------------------------------------------------
    # intensity information
    # --------------------------------------------------------

    raw_values = raw[
        np.isfinite(raw)
    ]

    corr_values = corrected[
        np.isfinite(corrected)
    ]

    raw_p01, raw_p50, raw_p99 = (
        np.percentile(
            raw_values,
            [1, 50, 99],
        )
    )

    corr_p01, corr_p50, corr_p99 = (
        np.percentile(
            corr_values,
            [1, 50, 99],
        )
    )

    median_scale = (
        float(corr_p50)
        / max(
            float(raw_p50),
            1e-8,
        )
    )

    sat_raw = (
        saturation_fraction(
            raw,
            dtype,
        )
    )

    sat_corrected = (
        saturation_fraction(
            corrected,
            dtype,
        )
    )

    edge_raw = (
        edge_center_bias(
            smooth_raw
        )
    )

    edge_corrected = (
        edge_center_bias(
            smooth_corrected
        )
    )

    return {
        "qc_downsample_step":
            step,

        "raw_mean":
            float(
                np.mean(raw_values)
            ),

        "corrected_mean":
            float(
                np.mean(corr_values)
            ),

        "raw_p01":
            float(raw_p01),

        "raw_p50":
            float(raw_p50),

        "raw_p99":
            float(raw_p99),

        "corrected_p01":
            float(corr_p01),

        "corrected_p50":
            float(corr_p50),

        "corrected_p99":
            float(corr_p99),

        "median_intensity_scale":
            median_scale,

        "lowfreq_cv_raw":
            low_raw,

        "lowfreq_cv_corrected":
            low_corrected,

        "edge_center_bias_raw":
            edge_raw,

        "edge_center_bias_corrected":
            edge_corrected,

        "high_pass_correlation":
            hp_corr,

        "ssim":
            ssim_value,

        "saturation_raw":
            sat_raw,

        "saturation_corrected":
            sat_corrected,
    }


# ============================================================
# 比较图
# ============================================================

def save_comparison_figure(
    raw,
    corrected,
    output_path,
    title,
    metrics,
):

    raw_small, _ = (
        downsample_qc(raw)
    )

    corrected_small, _ = (
        downsample_qc(
            corrected
        )
    )

    raw_norm = robust_normalize(
        raw_small
    )

    corrected_norm = (
        robust_normalize(
            corrected_small
        )
    )

    difference = (
        corrected_norm
        - raw_norm
    )

    sigma = max(
        8,
        min(
            raw_norm.shape
        ) / 32,
    )

    smooth_raw = gaussian_filter(
        raw_norm,
        sigma=sigma,
    )

    smooth_corr = gaussian_filter(
        corrected_norm,
        sigma=sigma,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    ax = axes[0, 0]

    ax.imshow(
        raw_norm,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        "Original"
    )

    ax.axis("off")


    ax = axes[0, 1]

    ax.imshow(
        corrected_norm,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        "Flat-field corrected"
    )

    ax.axis("off")


    ax = axes[0, 2]

    vmax = max(
        abs(
            float(
                np.min(
                    difference
                )
            )
        ),
        abs(
            float(
                np.max(
                    difference
                )
            )
        ),
        1e-6,
    )

    image = ax.imshow(
        difference,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_title(
        "Corrected - Original"
    )

    ax.axis("off")

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
    )


    ax = axes[1, 0]

    im = ax.imshow(
        smooth_raw,
        cmap="viridis",
    )

    ax.set_title(
        "Low-frequency field: original"
    )

    ax.axis("off")

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
    )


    ax = axes[1, 1]

    im = ax.imshow(
        smooth_corr,
        cmap="viridis",
    )

    ax.set_title(
        "Low-frequency field: corrected"
    )

    ax.axis("off")

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
    )


    ax = axes[1, 2]

    text = (
        f"High-pass correlation\n"
        f"{metrics['high_pass_correlation']:.4f}\n\n"
        f"SSIM\n"
        f"{metrics['ssim']:.4f}\n\n"
        f"Low-frequency CV\n"
        f"{metrics['lowfreq_cv_raw']:.4f}"
        f" → "
        f"{metrics['lowfreq_cv_corrected']:.4f}\n\n"
        f"Edge-center bias\n"
        f"{metrics['edge_center_bias_raw']:.4f}"
        f" → "
        f"{metrics['edge_center_bias_corrected']:.4f}\n\n"
        f"Median intensity scale\n"
        f"{metrics['median_intensity_scale']:.3f}"
    )

    ax.text(
        0.05,
        0.95,
        text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    ax.axis("off")

    fig.suptitle(
        title,
        fontsize=13,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Flatfield model QC
# ============================================================

def qc_models():

    rows = []

    if not MODEL_ROOT.exists():

        print(
            "\nWARNING: "
            "没有找到 _flatfield_models"
        )

        return rows

    flatfield_files = sorted(
        MODEL_ROOT.glob(
            "*_flatfield.npy"
        )
    )

    for flat_path in flatfield_files:

        model_name = (
            flat_path.name
            .replace(
                "_flatfield.npy",
                "",
            )
        )

        dark_path = (
            MODEL_ROOT
            / (
                model_name
                + "_darkfield.npy"
            )
        )

        flat = np.asarray(
            np.load(
                flat_path,
                mmap_mode="r",
            ),
            dtype=np.float32,
        )

        flat_small, _ = (
            downsample_qc(
                flat
            )
        )

        finite = flat_small[
            np.isfinite(flat_small)
        ]

        if finite.size == 0:

            rows.append(
                {
                    "model":
                        model_name,
                    "decision":
                        "NO",
                    "reason":
                        "flatfield contains no finite values",
                }
            )

            continue

        median = float(
            np.median(
                finite
            )
        )

        if median <= 0:

            rows.append(
                {
                    "model":
                        model_name,
                    "decision":
                        "NO",
                    "reason":
                        "flatfield median <= 0",
                }
            )

            continue

        normalized = (
            flat_small
            / median
        )

        p05, p50, p95 = (
            np.percentile(
                normalized[
                    np.isfinite(
                        normalized
                    )
                ],
                [5, 50, 95],
            )
        )

        dynamic_range = (
            float(p95)
            / max(
                float(p05),
                1e-8,
            )
        )

        sigma = max(
            3,
            min(
                normalized.shape
            ) / 40,
        )

        smooth = gaussian_filter(
            normalized,
            sigma=sigma,
        )

        high_frequency = (
            normalized
            - smooth
        )

        hf_rms = float(
            np.sqrt(
                np.mean(
                    high_frequency
                    ** 2
                )
            )
        )

        model_pass = (
            hf_rms
            <= MAX_MODEL_HIGH_FREQ_RMS
            and
            dynamic_range
            <= MAX_MODEL_DYNAMIC_RANGE
            and
            p05 > 0
        )

        decision = (
            "YES"
            if model_pass
            else "NO"
        )

        reasons = []

        if (
            hf_rms
            > MAX_MODEL_HIGH_FREQ_RMS
        ):

            reasons.append(
                "flatfield contains excessive high-frequency structure"
            )

        if (
            dynamic_range
            > MAX_MODEL_DYNAMIC_RANGE
        ):

            reasons.append(
                "flatfield dynamic range is extreme"
            )

        if p05 <= 0:

            reasons.append(
                "flatfield has non-positive values"
            )

        if not reasons:

            reasons.append(
                "flatfield is smooth and within accepted range"
            )

        dark = None

        if dark_path.exists():

            dark = np.asarray(
                np.load(
                    dark_path,
                    mmap_mode="r",
                ),
                dtype=np.float32,
            )

            dark, _ = (
                downsample_qc(
                    dark
                )
            )

        # ----------------------------------------------------
        # Model figure
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11, 9),
        )

        im = axes[0, 0].imshow(
            normalized,
            cmap="viridis",
        )

        axes[0, 0].set_title(
            "Flatfield / median"
        )

        axes[0, 0].axis(
            "off"
        )

        fig.colorbar(
            im,
            ax=axes[0, 0],
            fraction=0.046,
        )

        if dark is not None:

            im = axes[0, 1].imshow(
                dark,
                cmap="viridis",
            )

            axes[0, 1].set_title(
                "Darkfield"
            )

            axes[0, 1].axis(
                "off"
            )

            fig.colorbar(
                im,
                ax=axes[0, 1],
                fraction=0.046,
            )

        else:

            axes[0, 1].text(
                0.5,
                0.5,
                "Darkfield not found",
                ha="center",
                va="center",
            )

            axes[0, 1].axis(
                "off"
            )

        center_y = (
            normalized.shape[0]
            // 2
        )

        center_x = (
            normalized.shape[1]
            // 2
        )

        axes[1, 0].plot(
            normalized[
                center_y,
                :
            ],
            label="horizontal",
        )

        axes[1, 0].plot(
            normalized[
                :,
                center_x
            ],
            label="vertical",
        )

        axes[1, 0].axhline(
            1,
            linestyle="--",
        )

        axes[1, 0].set_title(
            "Central profiles"
        )

        axes[1, 0].legend()


        metrics_text = (
            f"Decision: {decision}\n\n"
            f"p05 = {p05:.4f}\n"
            f"p50 = {p50:.4f}\n"
            f"p95 = {p95:.4f}\n\n"
            f"p95 / p05 = "
            f"{dynamic_range:.3f}\n\n"
            f"high-frequency RMS = "
            f"{hf_rms:.5f}\n\n"
            + "\n".join(
                reasons
            )
        )

        axes[1, 1].text(
            0.05,
            0.95,
            metrics_text,
            ha="left",
            va="top",
            family="monospace",
        )

        axes[1, 1].axis(
            "off"
        )

        fig.suptitle(
            model_name
        )

        plt.tight_layout()

        output_path = (
            MODEL_QC_DIR
            / (
                safe_name(
                    model_name
                )
                + "_QC.png"
            )
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        rows.append(
            {
                "model":
                    model_name,

                "flatfield_shape":
                    str(flat.shape),

                "flatfield_p05":
                    float(p05),

                "flatfield_p50":
                    float(p50),

                "flatfield_p95":
                    float(p95),

                "p95_over_p05":
                    dynamic_range,

                "high_frequency_rms":
                    hf_rms,

                "decision":
                    decision,

                "reason":
                    "; ".join(
                        reasons
                    ),

                "qc_image":
                    str(
                        output_path.relative_to(
                            QC_ROOT
                        )
                    ),
            }
        )

    return rows


# ============================================================
# 比较所有 Zarr
# ============================================================

def compare_zarrs():

    rows = []
    integrity_failures = []

    corrected_stores = sorted(
        CORRECTED_ROOT.glob(
            "*.zarr"
        )
    )

    if not corrected_stores:

        raise RuntimeError(
            f"没有在下面找到 corrected Zarr:\n"
            f"{CORRECTED_ROOT}"
        )

    for corrected_path in corrected_stores:

        raw_path = (
            RAW_ROOT
            / corrected_path.name
        )

        print(
            "\n"
            "========================================"
        )

        print(
            f"QC: {corrected_path.name}"
        )

        print(
            "========================================"
        )

        if not raw_path.exists():

            integrity_failures.append(
                f"Raw Zarr missing: "
                f"{raw_path}"
            )

            print(
                "ERROR: 找不到对应原图"
            )

            continue

        raw_root = zarr.open_group(
            str(raw_path),
            mode="r",
        )

        corrected_root = (
            zarr.open_group(
                str(corrected_path),
                mode="r",
            )
        )

        raw_images = (
            discover_primary_images(
                raw_root
            )
        )

        corrected_images = (
            discover_primary_images(
                corrected_root
            )
        )

        corrected_map = {
            item["path"]: item
            for item in corrected_images
        }

        for info in raw_images:

            array_path = info[
                "path"
            ]

            if (
                array_path
                not in corrected_map
            ):

                integrity_failures.append(
                    f"{corrected_path.name}: "
                    f"array missing in corrected: "
                    f"{array_path}"
                )

                continue

            corrected_info = (
                corrected_map[
                    array_path
                ]
            )

            raw_array = raw_root[
                array_path
            ]

            corrected_array = (
                corrected_root[
                    array_path
                ]
            )

            # ------------------------------------------------
            # 数据完整性
            # ------------------------------------------------

            if (
                tuple(
                    raw_array.shape
                )
                !=
                tuple(
                    corrected_array.shape
                )
            ):

                integrity_failures.append(
                    f"{corrected_path.name} "
                    f"{array_path}: "
                    "shape changed"
                )

                continue

            if (
                str(raw_array.dtype)
                !=
                str(
                    corrected_array.dtype
                )
            ):

                integrity_failures.append(
                    f"{corrected_path.name} "
                    f"{array_path}: "
                    "dtype changed"
                )

            axes = info[
                "axes"
            ]

            shape = info[
                "shape"
            ]

            labels = info[
                "channel_labels"
            ]

            for channel in range(
                info["channels"]
            ):

                channel_label = (
                    labels[channel]
                    if channel < len(
                        labels
                    )
                    else f"channel_{channel}"
                )

                selectors = (
                    get_plane_selectors(
                        shape,
                        axes,
                        channel,
                        MAX_METRIC_PLANES_PER_CHANNEL,
                    )
                )

                # 只画其中若干张
                png_indices = set(
                    np.unique(
                        np.linspace(
                            0,
                            len(selectors) - 1,
                            min(
                                len(selectors),
                                MAX_PNG_PLANES_PER_CHANNEL,
                            ),
                            dtype=int,
                        )
                    ).tolist()
                )

                for plane_number, (
                    selector,
                    plane_description,
                ) in enumerate(
                    selectors
                ):

                    try:

                        raw = load_plane(
                            raw_array,
                            selector,
                        )

                        corrected = (
                            load_plane(
                                corrected_array,
                                selector,
                            )
                        )

                    except Exception as error:

                        integrity_failures.append(
                            f"{corrected_path.name} "
                            f"{array_path} "
                            f"ch={channel}: "
                            f"{error}"
                        )

                        continue

                    metrics = (
                        calculate_metrics(
                            raw,
                            corrected,
                            raw_array.dtype,
                        )
                    )

                    row = {
                        "zarr":
                            corrected_path.name,

                        "array":
                            array_path,

                        "shape":
                            str(shape),

                        "dtype":
                            str(
                                raw_array.dtype
                            ),

                        "axes":
                            ",".join(
                                axes
                            ),

                        "voxel_size":
                            str(
                                info[
                                    "voxel_size"
                                ]
                            ),

                        "channel_index":
                            channel,

                        "channel_label":
                            channel_label,

                        "plane":
                            plane_description,
                    }

                    row.update(
                        metrics
                    )

                    # ------------------------------------------------
                    # 单 plane 自动判断
                    # ------------------------------------------------

                    low_ratio = (
                        metrics[
                            "lowfreq_cv_corrected"
                        ]
                        /
                        max(
                            metrics[
                                "lowfreq_cv_raw"
                            ],
                            1e-8,
                        )
                    )

                    sat_raw = (
                        metrics[
                            "saturation_raw"
                        ]
                    )

                    sat_corr = (
                        metrics[
                            "saturation_corrected"
                        ]
                    )

                    if (
                        np.isfinite(
                            sat_raw
                        )
                        and
                        np.isfinite(
                            sat_corr
                        )
                    ):

                        sat_increase = (
                            sat_corr
                            - sat_raw
                        )

                    else:

                        sat_increase = 0

                    plane_pass = (
                        np.isfinite(
                            metrics[
                                "high_pass_correlation"
                            ]
                        )
                        and
                        metrics[
                            "high_pass_correlation"
                        ]
                        >= MIN_HIGH_PASS_CORR

                        and

                        np.isfinite(
                            metrics["ssim"]
                        )
                        and
                        metrics["ssim"]
                        >= MIN_SSIM

                        and

                        low_ratio
                        <= MAX_LOWFREQ_RATIO

                        and

                        sat_increase
                        <= MAX_SATURATION_INCREASE
                    )

                    row[
                        "lowfreq_ratio_after_over_before"
                    ] = low_ratio

                    row[
                        "saturation_increase"
                    ] = sat_increase

                    row[
                        "plane_decision"
                    ] = (
                        "YES"
                        if plane_pass
                        else "NO"
                    )

                    # ------------------------------------------------
                    # PNG
                    # ------------------------------------------------

                    if (
                        plane_number
                        in png_indices
                    ):

                        image_filename = (
                            safe_name(
                                corrected_path.stem
                            )
                            + "__"
                            + safe_name(
                                array_path
                            )
                            + f"__ch{channel}_"
                            + safe_name(
                                channel_label
                            )
                            + "__"
                            + safe_name(
                                plane_description
                            )
                            + ".png"
                        )

                        image_path = (
                            COMPARISON_DIR
                            / image_filename
                        )

                        title = (
                            f"{corrected_path.name}\n"
                            f"array={array_path} | "
                            f"channel={channel} "
                            f"({channel_label}) | "
                            f"{plane_description}"
                        )

                        save_comparison_figure(
                            raw,
                            corrected,
                            image_path,
                            title,
                            metrics,
                        )

                        row[
                            "comparison_image"
                        ] = str(
                            image_path.relative_to(
                                QC_ROOT
                            )
                        )

                    else:

                        row[
                            "comparison_image"
                        ] = ""

                    rows.append(
                        row
                    )

    return rows, integrity_failures


# ============================================================
# Channel / dataset summary
# ============================================================

def summarize_datasets(
    plane_df
):

    rows = []

    if plane_df.empty:
        return pd.DataFrame()

    grouping = [
        "zarr",
        "array",
        "channel_index",
        "channel_label",
    ]

    for keys, group in plane_df.groupby(
        grouping,
        dropna=False,
    ):

        (
            zarr_name,
            array_path,
            channel_index,
            channel_label,
        ) = keys

        hp_corr = float(
            np.nanmedian(
                group[
                    "high_pass_correlation"
                ]
            )
        )

        ssim_value = float(
            np.nanmedian(
                group[
                    "ssim"
                ]
            )
        )

        low_raw = float(
            np.nanmedian(
                group[
                    "lowfreq_cv_raw"
                ]
            )
        )

        low_corr = float(
            np.nanmedian(
                group[
                    "lowfreq_cv_corrected"
                ]
            )
        )

        low_ratio = (
            low_corr
            /
            max(
                low_raw,
                1e-8,
            )
        )

        edge_raw = float(
            np.nanmedian(
                group[
                    "edge_center_bias_raw"
                ]
            )
        )

        edge_corr = float(
            np.nanmedian(
                group[
                    "edge_center_bias_corrected"
                ]
            )
        )

        saturation_increase = float(
            np.nanmax(
                group[
                    "saturation_increase"
                ]
            )
        )

        structure_pass = (
            hp_corr
            >= MIN_HIGH_PASS_CORR
            and
            ssim_value
            >= MIN_SSIM
        )

        illumination_pass = (
            low_ratio
            <= MAX_LOWFREQ_RATIO
        )

        saturation_pass = (
            saturation_increase
            <= MAX_SATURATION_INCREASE
        )

        overall = (
            structure_pass
            and
            illumination_pass
            and
            saturation_pass
        )

        reasons = []

        if not structure_pass:

            reasons.append(
                "biological structure changed too much"
            )

        if not illumination_pass:

            reasons.append(
                "low-frequency nonuniformity became worse"
            )

        if not saturation_pass:

            reasons.append(
                "too many new saturated pixels"
            )

        if overall:

            reasons.append(
                "structure preserved and illumination not worsened"
            )

        rows.append(
            {
                "zarr":
                    zarr_name,

                "array":
                    array_path,

                "channel_index":
                    channel_index,

                "channel_label":
                    channel_label,

                "n_planes":
                    len(group),

                "median_high_pass_correlation":
                    hp_corr,

                "median_ssim":
                    ssim_value,

                "median_lowfreq_cv_raw":
                    low_raw,

                "median_lowfreq_cv_corrected":
                    low_corr,

                "lowfreq_after_over_before":
                    low_ratio,

                "median_edge_center_bias_raw":
                    edge_raw,

                "median_edge_center_bias_corrected":
                    edge_corr,

                "max_saturation_increase":
                    saturation_increase,

                "structure_pass":
                    "YES"
                    if structure_pass
                    else "NO",

                "illumination_pass":
                    "YES"
                    if illumination_pass
                    else "NO",

                "saturation_pass":
                    "YES"
                    if saturation_pass
                    else "NO",

                "decision":
                    "YES"
                    if overall
                    else "NO",

                "reason":
                    "; ".join(
                        reasons
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Summary plots
# ============================================================

def save_summary_plots(
    summary_df
):

    if summary_df.empty:
        return

    labels = (
        summary_df["zarr"]
        .astype(str)
        + " | "
        + summary_df[
            "channel_label"
        ].astype(str)
    )

    x = np.arange(
        len(summary_df)
    )

    # --------------------------------------------------------
    # Low-frequency CV before vs after
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            max(
                10,
                len(summary_df)
                * 0.6,
            ),
            6,
        )
    )

    width = 0.36

    ax.bar(
        x - width / 2,
        summary_df[
            "median_lowfreq_cv_raw"
        ],
        width,
        label="Original",
    )

    ax.bar(
        x + width / 2,
        summary_df[
            "median_lowfreq_cv_corrected"
        ],
        width,
        label="Corrected",
    )

    ax.set_ylabel(
        "Low-frequency CV"
    )

    ax.set_title(
        "Illumination non-uniformity before / after"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=60,
        ha="right",
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        SUMMARY_DIR
        / "lowfreq_before_after.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


    # --------------------------------------------------------
    # Structure preservation
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        summary_df[
            "median_high_pass_correlation"
        ],
        summary_df[
            "median_ssim"
        ],
    )

    ax.axvline(
        MIN_HIGH_PASS_CORR,
        linestyle="--",
    )

    ax.axhline(
        MIN_SSIM,
        linestyle="--",
    )

    for _, row in (
        summary_df.iterrows()
    ):

        ax.annotate(
            (
                str(
                    row[
                        "zarr"
                    ]
                )
                + " "
                + str(
                    row[
                        "channel_label"
                    ]
                )
            ),
            (
                row[
                    "median_high_pass_correlation"
                ],
                row[
                    "median_ssim"
                ],
            ),
            fontsize=7,
        )

    ax.set_xlabel(
        "High-pass correlation"
    )

    ax.set_ylabel(
        "SSIM"
    )

    ax.set_title(
        "Biological structure preservation"
    )

    plt.tight_layout()

    fig.savefig(
        SUMMARY_DIR
        / "structure_preservation.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# HTML report
# ============================================================

def generate_html_report(
    overall_decision,
    reasons,
    summary_df,
    model_df,
    integrity_failures,
):

    status_text = (
        "YES — 可以进入 stitching"
        if overall_decision == "YES"
        else "NO — 不建议进入 stitching"
    )

    summary_html = (
        summary_df.to_html(
            index=False,
            float_format=lambda x:
                f"{x:.5g}",
        )
        if not summary_df.empty
        else "<p>No dataset metrics.</p>"
    )

    model_html = (
        model_df.to_html(
            index=False,
            float_format=lambda x:
                f"{x:.5g}",
        )
        if not model_df.empty
        else "<p>No model metrics.</p>"
    )

    comparison_images = sorted(
        COMPARISON_DIR.glob(
            "*.png"
        )
    )

    model_images = sorted(
        MODEL_QC_DIR.glob(
            "*.png"
        )
    )

    comparison_html = ""

    for image_path in comparison_images:

        relative = (
            image_path.relative_to(
                QC_ROOT
            )
        )

        comparison_html += (
            "<div class='image-card'>"
            f"<p>{html.escape(image_path.name)}</p>"
            f"<img src='{relative.as_posix()}'>"
            "</div>"
        )

    model_image_html = ""

    for image_path in model_images:

        relative = (
            image_path.relative_to(
                QC_ROOT
            )
        )

        model_image_html += (
            "<div class='image-card'>"
            f"<p>{html.escape(image_path.name)}</p>"
            f"<img src='{relative.as_posix()}'>"
            "</div>"
        )

    integrity_html = ""

    if integrity_failures:

        integrity_html = (
            "<ul>"
            + "".join(
                f"<li>{html.escape(x)}</li>"
                for x in integrity_failures
            )
            + "</ul>"
        )

    else:

        integrity_html = (
            "<p>YES — "
            "raw / corrected data structure matched.</p>"
        )

    reasons_html = (
        "<ul>"
        + "".join(
            f"<li>{html.escape(x)}</li>"
            for x in reasons
        )
        + "</ul>"
    )

    report = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Flat-field QC Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 35px;
    line-height: 1.5;
}}

.status {{
    font-size: 28px;
    font-weight: bold;
    padding: 20px;
    border: 3px solid #333;
    margin-bottom: 25px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
}}

th, td {{
    border: 1px solid #aaa;
    padding: 6px;
}}

th {{
    background: #eee;
}}

.image-card {{
    margin-bottom: 30px;
}}

.image-card img {{
    max-width: 1200px;
    width: 100%;
    border: 1px solid #999;
}}

</style>

</head>

<body>

<h1>Flat-field Correction QC</h1>

<div class="status">
{status_text}
</div>

<h2>Decision reasons</h2>

{reasons_html}

<h2>Automatic thresholds</h2>

<ul>
<li>High-pass correlation ≥ {MIN_HIGH_PASS_CORR}</li>
<li>SSIM ≥ {MIN_SSIM}</li>
<li>Low-frequency CV after / before ≤ {MAX_LOWFREQ_RATIO}</li>
<li>New saturated pixels ≤ {MAX_SATURATION_INCREASE}</li>
<li>Flatfield high-frequency RMS ≤ {MAX_MODEL_HIGH_FREQ_RMS}</li>
<li>Flatfield p95/p05 ≤ {MAX_MODEL_DYNAMIC_RANGE}</li>
</ul>

<h2>Data integrity</h2>

{integrity_html}

<h2>Dataset / channel summary</h2>

{summary_html}

<h2>Flatfield model QC</h2>

{model_html}

<h2>Summary figures</h2>

<div class="image-card">
<img src="summary/lowfreq_before_after.png">
</div>

<div class="image-card">
<img src="summary/structure_preservation.png">
</div>

<h2>Flatfield models</h2>

{model_image_html}

<h2>Original vs Corrected</h2>

{comparison_html}

</body>
</html>
"""

    output = (
        QC_ROOT
        / "flat_field_QC_report.html"
    )

    output.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    print(
        "========================================"
    )

    print(
        "Flat-field Correction QC"
    )

    print(
        "========================================"
    )

    print(
        f"\nRaw:\n{RAW_ROOT}"
    )

    print(
        f"\nCorrected:\n"
        f"{CORRECTED_ROOT}"
    )

    print(
        f"\nQC output:\n"
        f"{QC_ROOT}"
    )

    if not RAW_ROOT.exists():

        print(
            "\nERROR: "
            "原始 Zarr 目录不存在。"
        )

        sys.exit(1)

    if not CORRECTED_ROOT.exists():

        print(
            "\nERROR: "
            "flat-field 输出目录不存在。"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # 1. corrected vs raw
    # --------------------------------------------------------

    plane_rows, integrity_failures = (
        compare_zarrs()
    )

    plane_df = pd.DataFrame(
        plane_rows
    )

    plane_table = (
        TABLE_DIR
        / "per_plane_metrics.tsv"
    )

    plane_df.to_csv(
        plane_table,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # 2. dataset/channel summary
    # --------------------------------------------------------

    summary_df = (
        summarize_datasets(
            plane_df
        )
    )

    summary_table = (
        TABLE_DIR
        / "dataset_channel_summary.tsv"
    )

    summary_df.to_csv(
        summary_table,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Flatfield model QC
    # --------------------------------------------------------

    model_rows = (
        qc_models()
    )

    model_df = pd.DataFrame(
        model_rows
    )

    model_table = (
        TABLE_DIR
        / "flatfield_model_QC.tsv"
    )

    model_df.to_csv(
        model_table,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Summary figures
    # --------------------------------------------------------

    save_summary_plots(
        summary_df
    )

    # --------------------------------------------------------
    # 5. 最终 YES / NO
    # --------------------------------------------------------

    reasons = []

    integrity_pass = (
        len(
            integrity_failures
        ) == 0
    )

    dataset_pass = (
        not summary_df.empty
        and
        bool(
            (
                summary_df[
                    "decision"
                ]
                == "YES"
            ).all()
        )
    )

    model_pass = (
        not model_df.empty
        and
        bool(
            (
                model_df[
                    "decision"
                ]
                == "YES"
            ).all()
        )
    )

    if integrity_pass:

        reasons.append(
            "YES: raw and corrected Zarr geometry/data structure match"
        )

    else:

        reasons.append(
            "NO: raw and corrected Zarr structure mismatch"
        )


    if dataset_pass:

        reasons.append(
            "YES: biological image structure is preserved"
        )

    else:

        reasons.append(
            "NO: one or more datasets/channels failed image QC"
        )


    if model_pass:

        reasons.append(
            "YES: flatfield models are spatially smooth"
        )

    else:

        reasons.append(
            "NO: one or more flatfield models failed model QC"
        )


    overall_decision = (
        "YES"
        if (
            integrity_pass
            and
            dataset_pass
            and
            model_pass
        )
        else "NO"
    )

    # --------------------------------------------------------
    # 6. 写最终 decision 文件
    # --------------------------------------------------------

    decision_path = (
        QC_ROOT
        / "OVERALL_DECISION.txt"
    )

    with open(
        decision_path,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            overall_decision
            + "\n\n"
        )

        if (
            overall_decision
            == "YES"
        ):

            handle.write(
                "Flat-field QC passed.\n"
                "可以进入 stitching。\n\n"
            )

        else:

            handle.write(
                "Flat-field QC failed.\n"
                "不建议直接进入 stitching。\n\n"
            )

        handle.write(
            "Reasons:\n"
        )

        for reason in reasons:

            handle.write(
                f"- {reason}\n"
            )

        if integrity_failures:

            handle.write(
                "\nIntegrity failures:\n"
            )

            for failure in (
                integrity_failures
            ):

                handle.write(
                    f"- {failure}\n"
                )

    # --------------------------------------------------------
    # 7. HTML report
    # --------------------------------------------------------

    generate_html_report(
        overall_decision,
        reasons,
        summary_df,
        model_df,
        integrity_failures,
    )

    # --------------------------------------------------------
    # console
    # --------------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "QC FINISHED"
    )

    print(
        "========================================"
    )

    print(
        "\nFINAL DECISION:"
    )

    print(
        f"\n>>> {overall_decision} <<<"
    )

    if (
        overall_decision
        == "YES"
    ):

        print(
            "\n可以进入 stitching。"
        )

    else:

        print(
            "\n不建议进入 stitching，"
            "请查看 QC 报告。"
        )

    print(
        f"\nQC directory:\n"
        f"{QC_ROOT}"
    )

    print(
        f"\nHTML report:\n"
        f"{QC_ROOT / 'flat_field_QC_report.html'}"
    )

    print(
        f"\nDecision:\n"
        f"{decision_path}"
    )


if __name__ == "__main__":
    main()
