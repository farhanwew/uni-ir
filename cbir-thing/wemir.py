"""
WEMIR - Weighted Edge Matching Information Retrieval
====================================================
Implementation of the CBIR method from:
    Tamilkodi & Nesakumari (2021)
    "A novel framework for retrieval of image using weighted edge matching algorithm"
    Multimedia Tools and Applications

Pipeline (faithfully following the paper):
    Preprocessing (Section 2):
        1. Median filter for noise removal
        2. K-means clustering (k=3) → group pixels
        3. Select one cluster → histogram equalization (confined mean)
        4. 1-level DWT → LL subband

    Feature Extraction - WEMIR (Section 3, Steps 1-10):
        1. Take LL image from preprocessing
        2. SVD for size reduction (I = U S V^T, use S_r @ Vt_r)
        3. Make matrix square (pad with zeros if needed)
        4-5. Row/column subtraction (subtract minima)
        6-8. Hungarian algorithm (minimum line covering + adjustment)
        9. Select assignments (single zeros in rows/columns)
        10. Store the central pixel value of each assigned minimum edge

    Retrieval (Step 12):
        Euclidean / Manhattan distance between feature vectors
"""

import numpy as np
import cv2
import pywt
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import pickle
import time


# =============================================================================
# Preprocessing (Paper Section 2)
# =============================================================================

STANDARD_SIZE = (256, 256)


def median_filter(image, ksize=3):
    """Remove noise using median filter (paper Section 2)."""
    return cv2.medianBlur(image, ksize)


def kmeans_cluster(image, k=3, max_iter=100):
    """
    K-means clustering on RGB pixel values (paper Section 2.1).

    Groups pixel values into k clusters using Euclidean distance.
    Uses deterministic initialization based on luminance quantization.
    """
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        0.2,
    )
    # Deterministic init: assign initial labels based on pixel luminance
    luminance = 0.299 * pixels[:, 2] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 0]
    init_labels = (
        np.digitize(
            luminance,
            bins=np.linspace(luminance.min(), luminance.max() + 1e-6, k + 1)[1:-1],
        )
        .astype(np.int32)
        .reshape(-1, 1)
    )
    _, labels, centers = cv2.kmeans(
        pixels, k, init_labels, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS
    )
    return labels.flatten(), centers


def select_largest_cluster(image, labels, k=3):
    """
    Select the cluster with the most pixels.

    Paper: "spot any one group to fed into confined mean computation"
    We select the largest cluster as a deterministic choice.
    """
    counts = np.bincount(labels, minlength=k)
    largest = np.argmax(counts)
    mask = (labels == largest).reshape(image.shape[:2])
    result = np.zeros_like(image)
    result[mask] = image[mask]
    return result, mask


def histogram_equalization(image):
    """
    Histogram equalization (paper Section 2.2: "confined mean computation").

    Paper formula: E[i,j] = floor(N * sum(H[m], m=0..I[i,j]))
    Maps pixel intensities via CDF for uniform distribution.
    """
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)


def dwt_ll(image):
    """
    1-level Discrete Wavelet Transform → LL subband (paper Section 2.3).

    Uses Haar wavelet. Returns LL (approximation) and detail subbands.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    coeffs = pywt.dwt2(gray.astype(np.float64), "haar")
    ll = coeffs[0]
    lh, hl, hh = coeffs[1]
    return ll, lh, hl, hh


# =============================================================================
# WEMIR Feature Extraction (Paper Section 3, Steps 1-10)
# =============================================================================


def svd_reduce(matrix, rank=10, target_size=(25, 25)):
    """
    SVD for size reduction (paper Step 2).

    "Reduce the image size by using single value decomposition (SVD)"

    Factorizes I = U S V^T, reconstructs with rank-r approximation
    (preserving spatial structure), then resizes to a fixed target
    size for consistent feature extraction.

    Args:
        matrix: 2D array (the LL subband)
        rank: number of singular values to keep (default 10)
        target_size: output dimensions, divisible by 5 (default 25x25)

    Returns:
        Reduced matrix of shape target_size
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    if rank is None:
        rank = 10
    rank = min(rank, len(S))
    # Rank-r approximation preserves spatial structure from U
    reduced = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    # Resize to fixed target for consistent feature count
    reduced = cv2.resize(
        reduced, (target_size[1], target_size[0]),
        interpolation=cv2.INTER_AREA
    )
    return reduced




def hungarian_block(block):
    """
    Apply Hungarian algorithm to a 5x5 block (paper Steps 4-9).

    Paper describes:
        Step 4: Row subtraction (subtract row minima)
        Step 5: Column subtraction (subtract col minima)
        Step 6: Draw minimum lines to cover all zeros
        Step 7: If lines == size -> done; else adjust
        Step 8: Repeat until optimal
        Step 9: Select assignments
        Step 10: Store intensity values

    Uses scipy's linear_sum_assignment for correctness.

    Returns:
        Array of assigned intensity values (minimum edge values)
    """
    cost = block.copy()
    if cost.min() < 0:
        cost = cost - cost.min()

    row_ind, col_ind = linear_sum_assignment(cost)
    return block[row_ind, col_ind]


def extract_shape_features(ll, svd_rank=10, block_size=5):
    """
    Extract shape features via SVD + Hungarian (paper Steps 1-10).

    1. SVD reduction on LL subband (resize to fixed size)
    2. Split into 5x5 blocks
    3. Hungarian algorithm on each block
    4. Store assigned intensity values

    Returns:
        1D feature vector (5 values per block)
    """
    # Step 2: SVD reduction (includes resize to 25x25)
    reduced = svd_reduce(ll, rank=svd_rank)

    h, w = reduced.shape
    features = []

    # Steps 4-10: Hungarian on each 5x5 block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = reduced[i : i + block_size, j : j + block_size]
            assigned = hungarian_block(block)
            features.extend(assigned)

    return np.array(features, dtype=np.float64)


# =============================================================================
# Color Feature Extraction
# (Paper abstract: "fusion approach to extract color, texture and shape")
# =============================================================================


def extract_color_features(image):
    """
    Extract color features (paper: "higher order of confined mean"
    for color feature extraction).

    Computes HSV color histogram from the full preprocessed image.
    Uses 16 hue, 4 saturation, 4 value bins = 256 features.
    """
    if len(image.shape) < 3:
        return np.zeros(256)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [16, 4, 4], [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


# =============================================================================
# Texture Feature Extraction
# (Paper: "multi optimization techniques" for texture)
# =============================================================================


def extract_texture_features(lh, hl, hh):
    """
    Extract texture features from DWT detail subbands.

    Paper: "multi optimization techniques are used" for texture extraction.
    The DWT detail subbands (LH, HL, HH) capture directional texture
    information. We compute statistical features from each subband.
    """
    features = []
    for subband in [lh, hl, hh]:
        abs_sub = np.abs(subband)
        total = abs_sub.sum() + 1e-10
        features.extend([
            np.mean(abs_sub),                    # mean energy
            np.std(abs_sub),                     # energy spread
            np.sqrt(np.mean(subband ** 2)),       # RMS energy
            -np.sum((abs_sub / total)             # entropy
                    * np.log2(abs_sub / total + 1e-10)),
        ])
    return np.array(features)


# =============================================================================
# Full Pipeline
# =============================================================================


def preprocess(image):
    """
    Full preprocessing pipeline (paper Section 2).

    1. Resize to standard size
    2. Median filter for noise removal
    3. K-means clustering (k=3) on RGB pixels
    4. Select largest cluster
    5. Histogram equalization (confined mean)
    6. 1-level DWT → LL + detail subbands

    Returns:
        Tuple of (color_image, ll, lh, hl, hh) where:
            color_image: full histogram-equalized image (for color features)
            ll, lh, hl, hh: DWT subbands (from cluster image, for shape/texture)
    """
    image = cv2.resize(image, STANDARD_SIZE, interpolation=cv2.INTER_AREA)
    filtered = median_filter(image)

    # Color features use the full enhanced image
    color_image = histogram_equalization(filtered)

    # Shape/texture features use the cluster-focused image
    labels, centers = kmeans_cluster(filtered, k=3)
    cluster_img, mask = select_largest_cluster(filtered, labels)
    enhanced = histogram_equalization(cluster_img)
    ll, lh, hl, hh = dwt_ll(enhanced)

    return color_image, ll, lh, hl, hh


def extract_features(image, svd_rank=10):
    """
    Full WEMIR feature extraction — fusion of color, texture, and shape.

    Paper abstract: "It is a fusion approach to extract the color, texture
    and shape features from images."

    Pipeline:
        1. Preprocessing (median filter, K-means, histeq, DWT)
        2. Color features: HSV histogram from full image
        3. Texture features: DWT detail subband statistics
        4. Shape features: SVD + Hungarian on LL subband
        5. Fusion: concatenate + L2 normalize

    Args:
        image: BGR image (numpy array)
        svd_rank: SVD rank for shape features (default 10)

    Returns:
        1D L2-normalized feature vector
    """
    color_image, ll, lh, hl, hh = preprocess(image)

    # Color features
    color_feat = extract_color_features(color_image)

    # Texture features
    texture_feat = extract_texture_features(lh, hl, hh)

    # Shape features (WEMIR core)
    shape_feat = extract_shape_features(ll, svd_rank)

    # Fusion
    combined = np.concatenate([color_feat, texture_feat, shape_feat])

    # L2 normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    return combined


# =============================================================================
# Distance / Similarity (Paper Step 12)
# =============================================================================


def compute_distance(feat_a, feat_b, metric="euclidean"):
    """
    Compute distance between two feature vectors (paper Step 12).

    Handles different-length vectors by zero-padding the shorter one.
    """
    max_len = max(len(feat_a), len(feat_b))
    a = np.zeros(max_len)
    b = np.zeros(max_len)
    a[: len(feat_a)] = feat_a
    b[: len(feat_b)] = feat_b

    if metric == "euclidean":
        return np.sqrt(np.sum((a - b) ** 2))
    elif metric == "manhattan":
        return np.sum(np.abs(a - b))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# Index
# =============================================================================


class WEMIRIndex:
    """
    WEMIR feature index for content-based image retrieval.

    Builds a database of feature vectors from a directory of images,
    and supports querying with a new image to find the most similar ones.
    """

    def __init__(self, svd_rank=10):
        self.svd_rank = svd_rank
        self.features = {}   # path (str) -> feature vector
        self.labels = {}     # path (str) -> category label

    def build(self, image_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
        """
        Build the feature index from all images in a directory.

        Expects images organized in category subfolders:
            image_dir/category1/img1.jpg, img2.jpg, ...
        """
        image_dir = Path(image_dir)
        image_paths = sorted(
            [p for p in image_dir.rglob("*") if p.suffix.lower() in extensions]
        )

        total = len(image_paths)
        print(f"Building WEMIR index for {total} images...")
        start = time.time()

        for idx, img_path in enumerate(image_paths, 1):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [{idx}/{total}] skipped (unreadable): {img_path.name}")
                continue

            try:
                feat = extract_features(image, self.svd_rank)
                self.features[str(img_path)] = feat
                self.labels[str(img_path)] = img_path.parent.name

                if idx % 50 == 0 or idx == total:
                    elapsed = time.time() - start
                    print(f"  [{idx}/{total}] {elapsed:.1f}s elapsed")

            except Exception as e:
                print(f"  [{idx}/{total}] failed: {img_path.name} -- {e}")

        elapsed = time.time() - start
        print(f"Done! Indexed {len(self.features)} images in {elapsed:.1f}s")

    def query(self, query_image_or_path, top_k=10, metric="euclidean"):
        """Retrieve the top-k most similar images to a query."""
        if isinstance(query_image_or_path, (str, Path)):
            image = cv2.imread(str(query_image_or_path))
            if image is None:
                raise ValueError(f"Cannot read image: {query_image_or_path}")
        else:
            image = query_image_or_path

        query_feat = extract_features(image, self.svd_rank)

        distances = []
        for path, feat in self.features.items():
            dist = compute_distance(query_feat, feat, metric)
            label = self.labels.get(path, "unknown")
            distances.append((path, dist, label))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def evaluate(self, query_image_path, top_k=10, metric="euclidean"):
        """
        Query and compute precision/recall.

        The query image itself is excluded from results (self-match).
        Precision = relevant retrieved / total retrieved
        Recall = relevant retrieved / total relevant in database
        """
        query_path = Path(query_image_path)
        query_label = query_path.parent.name
        query_str = str(query_path)

        # Request extra to account for self-match removal
        results = self.query(query_image_path, top_k + 1, metric)
        results = [(p, d, l) for p, d, l in results if p != query_str]
        results = results[:top_k]

        total_relevant = sum(
            1 for p, lbl in zip(self.labels.keys(), self.labels.values())
            if lbl == query_label and p != query_str
        )

        relevant_retrieved = sum(1 for _, _, lbl in results if lbl == query_label)

        precision = relevant_retrieved / len(results) if results else 0.0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

        return {
            "results": results,
            "precision": precision,
            "recall": recall,
            "query_label": query_label,
            "total_relevant": total_relevant,
            "relevant_retrieved": relevant_retrieved,
        }

    def save(self, path):
        """Save the index to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump({
                "features": self.features,
                "labels": self.labels,
                "svd_rank": self.svd_rank,
            }, f)
        print(f"Index saved to {path}")

    @classmethod
    def load(cls, path):
        """Load an index from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(svd_rank=data.get("svd_rank"))
        idx.features = data["features"]
        idx.labels = data["labels"]
        print(f"Index loaded: {len(idx.features)} images")
        return idx
