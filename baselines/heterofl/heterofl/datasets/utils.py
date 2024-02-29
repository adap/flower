"""Contains utility functions required for datasests.

Adopted from authors implementation.
"""

import glob
import gzip
import hashlib
import os
import tarfile
import zipfile
from collections import Counter

import anytree
import numpy as np
from PIL import Image
from six.moves import urllib
from tqdm import tqdm

from heterofl.utils import makedir_exist_ok

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


def find_classes(drctry):
    """Find the classes in a directory."""
    classes = [d.name for d in os.scandir(drctry) if d.is_dir()]
    classes.sort()
    classes_to_labels = {classes[i]: i for i in range(len(classes))}
    return classes_to_labels


def pil_loader(path):
    """Load image from path using PIL."""
    with open(path, "rb") as file:
        img = Image.open(file)
        return img.convert("RGB")


# def accimage_loader(path):
#     """Load image from path using accimage_loader."""
#     import accimage

#     try:
#         return accimage.Image(path)
#     except IOError:
#         return pil_loader(path)


def default_loader(path):
    """Load image from path using default loader."""
    # if get_image_backend() == "accimage":
    #     return accimage_loader(path)

    return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    """Check whether file possesses any of the extensions listed."""
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_classes_counts(label):
    """Count number of classes."""
    label = np.array(label)
    if label.ndim > 1:
        label = label.sum(axis=tuple(range(1, label.ndim)))
    classes_counts = Counter(label)
    return classes_counts


def _make_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def _calculate_md5(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _check_md5(path, md5, **kwargs):
    return md5 == _calculate_md5(path, **kwargs)


def _check_integrity(path, md5=None):
    if not os.path.isfile(path):
        return False
    if md5 is None:
        return True
    return _check_md5(path, md5)


def download_url(url, root, filename, md5):
    """Download files from the url."""
    path = os.path.join(root, filename)
    makedir_exist_ok(root)
    if os.path.isfile(path) and _check_integrity(path, md5):
        print("Using downloaded and verified file: " + path)
    else:
        try:
            print("Downloading " + url + " to " + path)
            urllib.request.urlretrieve(
                url, path, reporthook=_make_bar_updater(tqdm(unit="B", unit_scale=True))
            )
        except OSError:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + path
                )
                urllib.request.urlretrieve(
                    url,
                    path,
                    reporthook=_make_bar_updater(tqdm(unit="B", unit_scale=True)),
                )
        if not _check_integrity(path, md5):
            raise RuntimeError("Not valid downloaded file")


def extract_file(src, dest=None, delete=False):
    """Extract the file."""
    print("Extracting {}".format(src))
    dest = os.path.dirname(src) if dest is None else dest
    filename = os.path.basename(src)
    if filename.endswith(".zip"):
        with zipfile.ZipFile(src, "r") as zip_f:
            zip_f.extractall(dest)
    elif filename.endswith(".tar"):
        with tarfile.open(src) as tar_f:
            tar_f.extractall(dest)
    elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(src, "r:gz") as tar_f:
            tar_f.extractall(dest)
    elif filename.endswith(".gz"):
        with open(src.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(src) as zip_f:
            out_f.write(zip_f.read())
    if delete:
        os.remove(src)


def make_data(root, extensions):
    """Get all the files in the root directory that follows the given extensions."""
    path = []
    files = glob.glob("{}/**/*".format(root), recursive=True)
    for file in files:
        if has_file_allowed_extension(file, extensions):
            path.append(os.path.normpath(file))
    return path


# pylint: disable=dangerous-default-value
def make_img(path, classes_to_labels, extensions=IMG_EXTENSIONS):
    """Make image."""
    img, label = [], []
    classes = []
    leaf_nodes = classes_to_labels.leaves
    for node in leaf_nodes:
        classes.append(node.name)
    for cls in sorted(classes):
        folder = os.path.join(path, cls)
        if not os.path.isdir(folder):
            continue
        for root, _, filenames in sorted(os.walk(folder)):
            for filename in sorted(filenames):
                if has_file_allowed_extension(filename, extensions):
                    cur_path = os.path.join(root, filename)
                    img.append(cur_path)
                    label.append(
                        anytree.find_by_attr(classes_to_labels, cls).flat_index
                    )
    return img, label


def make_tree(root, name, attribute=None):
    """Create a tree of name."""
    if len(name) == 0:
        return
    if attribute is None:
        attribute = {}
    this_name = name[0]
    next_name = name[1:]
    this_attribute = {k: attribute[k][0] for k in attribute}
    next_attribute = {k: attribute[k][1:] for k in attribute}
    this_node = anytree.find_by_attr(root, this_name)
    this_index = root.index + [len(root.children)]
    if this_node is None:
        this_node = anytree.Node(
            this_name, parent=root, index=this_index, **this_attribute
        )
    make_tree(this_node, next_name, next_attribute)
    return


def make_flat_index(root, given=None):
    """Make flat index for each leaf node in the tree."""
    if given:
        classes_size = 0
        for node in anytree.PreOrderIter(root):
            if len(node.children) == 0:
                node.flat_index = given.index(node.name)
                classes_size = (
                    given.index(node.name) + 1
                    if given.index(node.name) + 1 > classes_size
                    else classes_size
                )
    else:
        classes_size = 0
        for node in anytree.PreOrderIter(root):
            if len(node.children) == 0:
                node.flat_index = classes_size
                classes_size += 1
    return classes_size


class Compose:
    """Custom Compose class."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inp):
        """Apply transforms when called."""
        for transform in self.transforms:
            inp["img"] = transform(inp["img"])
        return inp

    def __repr__(self):
        """Represent Compose as string."""
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(transform)
        format_string += "\n)"
        return format_string
