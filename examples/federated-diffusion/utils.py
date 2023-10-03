from collections import namedtuple
from functools import partial

import numpy as np
from PIL import Image

from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import torchvision.models as models

Manifold = namedtuple("Manifold", ["features", "radii"])
PrecisionAndRecall = namedtuple("PrecisinoAndRecall", ["precision", "recall"])


class IPR:
    def __init__(self, batch_size=50, k=3, num_samples=10000, model=None):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        if model is None:
            print(
                "loading vgg16 for improved precision and recall...", end="", flush=True
            )
            self.vgg16 = models.vgg16(pretrained=True).cuda().eval()
            print("done")
        else:
            self.vgg16 = model

    def __call__(self, subject):
        return self.precision_and_recall(subject)

    def precision_and_recall(self, subject):
        """
        Compute precision and recall for given subject
        reference should be precomputed by IPR.compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        """
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        manifold_subject = self.compute_manifold(subject)
        precision = compute_metric(
            self.manifold_ref, manifold_subject.features, "computing precision..."
        )
        recall = compute_metric(
            manifold_subject, self.manifold_ref.features, "computing recall..."
        )
        return PrecisionAndRecall(precision, recall)

    def compute_manifold_ref(self, path):
        self.manifold_ref = self.compute_manifold(path)

    def compute_manifold(self, input):
        """
        Compute manifold of given input
        args:
            input: path or images, same as above
        returns:
            Manifold(features, radii)
        """
        # features
        if isinstance(input, str):
            if input.endswith(".npz"):  # input is precalculated file
                print("loading", input)
                f = np.load(input)
                feats = f["feature"]
                radii = f["radii"]
                f.close()
                return Manifold(feats, radii)
            else:  # input is dir
                feats = self.extract_features_from_files(input)
        elif isinstance(input, torch.Tensor):
            feats = self.extract_features(input)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self.extract_features(input)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)
                feats = self.extract_features(input)
            elif isinstance(input[0], np.ndarray):
                input = np.concatenate(input, axis=0)
                input = torch.Tensor(input)
                feats = self.extract_features(input)
            elif isinstance(input[0], str):  # input is list of fnames
                feats = self.extract_features_from_files(input)
            else:
                raise TypeError
        else:
            print(type(input))
            raise TypeError

        # radii
        distances = compute_pairwise_distances(feats)
        radii = distances2radii(distances, k=self.k)
        return Manifold(feats, radii)

    def extract_features(self, images):
        """
        Extract features of vgg16-fc2 for all images
        params:
            images: torch.Tensors of size N x C x H x W
        returns:
            A numpy array of dimension (num images, dims)
        """
        desc = "extracting features of %d images" % images.size(0)
        num_batches = int(np.ceil(images.size(0) / self.batch_size))
        _, _, height, width = images.shape
        if height != 224 or width != 224:
            print("IPR: resizing %s to (224, 224)" % str((height, width)))
            resize = partial(F.interpolate, size=(224, 224))
        else:

            def resize(x):
                return x

        features = []
        for bi in trange(num_batches, desc=desc, disable=True):
            start = bi * self.batch_size
            end = start + self.batch_size
            batch = images[start:end]
            batch = resize(batch)
            before_fc = self.vgg16.features(batch.cuda())
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def extract_features_from_files(self, path_or_fnames):
        """
        Extract features of vgg16-fc2 for all images in path
        params:
            path_or_fnames: dir containing images or list of fnames(str)
        returns:
            A numpy array of dimension (num images, dims)
        """

        dataloader = get_custom_loader(
            path_or_fnames, batch_size=self.batch_size, num_samples=self.num_samples
        )
        num_found_images = len(dataloader.dataset)
        desc = "extracting features of %d images" % num_found_images
        if num_found_images < self.num_samples:
            print(
                "WARNING: num_found_images(%d) < num_samples(%d)"
                % (num_found_images, self.num_samples)
            )

        features = []
        for batch in tqdm(dataloader, desc=desc):
            before_fc = self.vgg16.features(batch.cuda())
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def save_ref(self, fname):
        print("saving manifold to", fname, "...")
        np.savez_compressed(
            fname, feature=self.manifold_ref.features, radii=self.manifold_ref.radii
        )


def compute_pairwise_distances(X, Y=None):
    """
    args:
        X: np.array of shape N x dim
        Y: np.array of shape N x dim
    returns:
        N x N symmetric np.array
    """
    num_X = X.shape[0]
    if Y is None:
        num_Y = num_X
    else:
        num_Y = Y.shape[0]
    X = X.astype(np.float64)  # to prevent underflow
    X_norm_square = np.sum(X**2, axis=1, keepdims=True)
    if Y is None:
        Y_norm_square = X_norm_square
    else:
        Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)
    X_square = np.repeat(X_norm_square, num_Y, axis=1)
    Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
    if Y is None:
        Y = X
    XY = np.dot(X, Y.T)
    diff_square = X_square - 2 * XY + Y_square

    # check negative distance
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        print(
            "WARNING: %d negative diff_squares found and set to zero, min_diff_square="
            % idx.sum(),
            min_diff_square,
        )

    distances = np.sqrt(diff_square)
    return distances


def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


def compute_metric(manifold_ref, feats_subject, desc=""):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in trange(num_subjects, desc=desc, disable=True):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / num_subjects


def prepare_tensors(orig_images_dataloader, gen_images, num):
    orig_tensor = torch.empty(0, dtype=torch.float32, requires_grad=False)

    target = []
    for _, batch in enumerate(orig_images_dataloader):
        clean_images = batch[0]
        target.extend(batch[1].tolist())  # get the target label
        orig_tensor = torch.cat([orig_tensor, clean_images], dim=0)
        if orig_tensor.shape[0] == num:
            break

    orig_tensor = orig_tensor / 2 + 0.5  # convert from -1 to 1 - 0 to 1
    orig_tensor = orig_tensor * 255.0  # convert to 0-255
    orig_tensor = orig_tensor.to(torch.uint8)

    gen_image_array_list = [
        np.array(image) for image in gen_images
    ]  # Convert each PIL Image to a numpy array
    gen_image_array = np.array(gen_image_array_list)
    gen_tensor = torch.tensor(
        gen_image_array
    )  # Convert the list of numpy arrays to a PyTorch tensor
    gen_tensor = gen_tensor.permute(0, 3, 2, 1)

    return orig_tensor, gen_tensor, target


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid
