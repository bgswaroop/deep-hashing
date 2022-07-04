import json
from io import BytesIO

import cv2
import numpy as np
import skimage as sk
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.filters import gaussian
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from wand.api import library as wandlibrary
from wand.image import Image as WandImage


class AddGaussianNoisePyTorch(object):
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, tensor):
        return torch.clip(tensor + torch.randn(tensor.size()) * self.std + self.mean, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def compare_methods_with_noisy_test():
    with open(r'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/'
              r'48-bit-scratch_30-60epochs/plots/gaussian.json') as f:
        baseline_ConvNet = json.load(f)
    exp1_name = 'push3_pull3_avg3_inhibition1_scale0_bias1'
    with open(rf'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/{exp1_name}/plots/gaussian.json') as f:
        exp1_push_pull = json.load(f)
    exp2_name = 'push3_pull5_avg0_inhibition1_scale0_bias1'
    with open(rf'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/{exp2_name}/plots/gaussian.json') as f:
        push_pull_exp2 = json.load(f)
    exp3_name = 'push3_pull5_avg3_inhibition1_scale0_bias1'
    with open(rf'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/{exp3_name}/plots/gaussian.json') as f:
        push_pull_exp3 = json.load(f)
    exp4_name = 'push5_pull5_avg3_inhibition1_scale0_bias1'
    with open(rf'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/{exp4_name}/plots/gaussian.json') as f:
        push_pull_exp4 = json.load(f)
    exp5_name = 'push5_pull5_avg5_inhibition1_scale0_bias1'
    with open(rf'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/{exp5_name}/plots/gaussian.json') as f:
        push_pull_exp5 = json.load(f)
    exp6_name = 'push5_pull5_avg0_inhibition1_scale0_bias1'
    with open(rf'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/{exp6_name}/plots/gaussian.json') as f:
        push_pull_exp6 = json.load(f)
    exp7_name = 'push3_pull3_avg0_inhibition1_scale0_bias1'
    with open(rf'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/{exp7_name}/plots/gaussian.json') as f:
        push_pull_exp7 = json.load(f)

    assert baseline_ConvNet['std_dev'] == exp1_push_pull['std_dev'], 'Mismatch in x-axis std_dev'

    plt.figure()
    plt.plot(baseline_ConvNet['std_dev'], baseline_ConvNet['mAP_score'], label='baseline_conv5x5_bias1')
    plt.plot(baseline_ConvNet['std_dev'], exp1_push_pull['mAP_score'], label=f'{exp1_name}')
    plt.plot(baseline_ConvNet['std_dev'], push_pull_exp2['mAP_score'], label=f'{exp2_name}')
    plt.plot(baseline_ConvNet['std_dev'], push_pull_exp3['mAP_score'], label=f'{exp3_name}')
    plt.plot(baseline_ConvNet['std_dev'], push_pull_exp4['mAP_score'], label=f'{exp4_name}')
    plt.plot(baseline_ConvNet['std_dev'], push_pull_exp5['mAP_score'], label=f'{exp5_name}')
    plt.plot(baseline_ConvNet['std_dev'], push_pull_exp6['mAP_score'], label=f'{exp6_name}')
    plt.plot(baseline_ConvNet['std_dev'], push_pull_exp7['mAP_score'], label=f'{exp7_name}')
    plt.xlabel('std_dev')
    plt.ylabel('mAP_score')
    plt.title('Effect of Gaussian Noise during test\nwhen ConvNet is trained on clean images')

    ax = plt.gca()
    # ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.3, 0.78])

    # plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # plt.tight_layout(rect=[0,0,0.75,1])
    plt.savefig(r'_plot_figures/impact_of_gaussian_noise_inhibition1.png', bbox_inches="tight")
    plt.show()


def plot_push_pull_kernels(push_response, pull_response, avg_pull_response, x, x_out, x_out_scaled, k=0):
    fig, ax = plt.subplots(2, 3)
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]
    ax5 = ax[0][2]
    ax6 = ax[1][2]

    im1 = ax1.imshow(push_response[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    ax1.set_title('push_response')

    im2 = ax2.imshow(x_out[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    ax2.set_title('final_response')

    im3 = ax3.imshow(pull_response[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')
    ax3.set_title('pull_response')

    im4 = ax4.imshow(avg_pull_response[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax, orientation='vertical')
    ax4.set_title('avg_pull_response')

    im5 = ax5.imshow(x[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax, orientation='vertical')
    ax5.set_title('input')

    im6 = ax6.imshow(x_out_scaled[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im6, cax=cax, orientation='vertical')
    ax6.set_title('final_response_scaled')

    plt.tight_layout()
    plt.show()


def plot_cifar_with_gaussian_noise():
    fig, ax = plt.subplots(2, 10, figsize=(15, 4))
    dataset_dir = r'/data/p288722/datasets/cifar'
    num_tests = 20
    row_id = 0
    col_id = 0
    for std in [float(x) for x in torch.arange(0, 0.20, 0.20 / num_tests)][:num_tests]:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoisePyTorch(mean=0, std=std),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_split_test = CIFAR10(dataset_dir, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(data_split_test, batch_size=32)

        for data, label in test_loader:
            sample_data = data[4]
            ax[row_id][col_id].imshow(torch.transpose(sample_data, 0, 2))
            ax[row_id][col_id].set_title(f'{round(std, 3)}')
            ax[row_id][col_id].axis('off')
            break

        col_id += 1
        if col_id == 10:
            col_id = 0
            row_id += 1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)
    fig.suptitle("Gaussian noise with varying levels of std using PyTorch", fontsize="x-large")
    fig.tight_layout()
    fig.savefig(r'_plot_figures/manual_gaussian_noise.png')
    fig.show()


def plot_corruptions_from_CIFAR10C():
    # dataset_name, severity_levels = 'gaussian_noise', [0.04, 0.06, .08, .09, .10]
    # dataset_name, severity_levels = 'shot_noise', [500, 250, 100, 75, 50]
    # dataset_name, severity_levels = 'impulse_noise', [.01, .02, .03, .05, .07]
    dataset_name, severity_levels = 'speckle_noise', [.06, .1, .12, .16, .2]

    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)

    dataset = np.load(rf'/data/p288722/datasets/cifar/CIFAR-10-C/{dataset_name}.npy')
    col_id = 0
    image_index = list(range(4, 50000, 10000))
    for data, std in zip(dataset[image_index], severity_levels):
        ax[col_id].imshow(data)
        ax[col_id].set_title(f'{round(std, 3)}')
        ax[col_id].axis('off')
        col_id += 1
        if col_id == 5:
            col_id = 0

    fig.suptitle(f"{dataset_name} images reproduced from the CIFAR-10-C data set", fontsize="x-large")
    fig.tight_layout()
    fig.savefig(rf'_plot_figures/CIFAR-10-C_{dataset_name}.png')
    fig.show()


def plot_simulate_corruptions_from_CIFAR10C():
    # 10 levels of Severities
    # Noise, severity_levels = GaussianNoiseNumPy, [0.04, 0.06, .08, .10, 0.12, 0.14, 0.16, 0.18, 0.19, 0.2]
    # Noise, severity_levels = ShotNoiseNumPy, [500, 250, 100, 80, 60, 50, 40, 30, 20, 15]
    # Noise, severity_levels = ImpulseNoiseNumPy, [.01, .02, .03, .05, .07, .09, .11, .13, .15, .17]
    # Noise, severity_levels = DefocusBlurNumPy, [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1),
    #                                             (1.5, 0.5), (1.9, 0.1), (2.2, 0.1), (2.5, 0.1), (3, 0.1)]
    # Noise, severity_levels = GlassBlurNumPy, [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2),
    #                                           (0.5, 1, 1), (0.5, 1, 2), (0.6, 1, 2), (0.6, 1, 3), (0.7, 1, 2)]
    Noise, severity_levels = MotionBlurNumPy, [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]

    # CIFAR-10-C Severities
    # Noise, severity_levels = GaussianNoiseNumPy, [0.04, 0.06, .08, .09, .10]
    # Noise, severity_levels = ShotNoiseNumPy, [500, 250, 100, 75, 50]
    # Noise, severity_levels = ImpulseNoiseNumPy, [.01, .02, .03, .05, .07]
    # Noise, severity_levels = SpeckleNoiseNumPy, [.06, .1, .12, .16, .2]
    # Noise, severity_levels = DefocusBlurNumPy, [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)]
    # Noise, severity_levels = GlassBlurNumPy, [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)]

    # ImageNet-C Severities
    # Noise, severity_levels = GaussianNoiseNumPy, [.08, .12, 0.18, 0.26, 0.38]
    # Noise, severity_levels = ShotNoiseNumPy, [60, 25, 12, 5, 3]
    # Noise, severity_levels = ImpulseNoiseNumPy, [.03, .06, .09, 0.17, 0.27]
    # Noise, severity_levels = SpeckleNoiseNumPy, [.15, .2, 0.35, 0.45, 0.6]
    # Noise, severity_levels = DefocusBlurNumPy, [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
    # Noise, severity_levels = GlassBlurNumPy, [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)]

    fig, ax = plt.subplots(1, len(severity_levels), figsize=(10, 2))
    dataset_dir = r'/data/p288722/datasets/cifar'

    col_id = 0
    for severity_level in severity_levels:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            Noise(severity_level),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_split_test = CIFAR10(dataset_dir, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(data_split_test, batch_size=32)

        for data, label in test_loader:
            sample_data = data[4]
            ax[col_id].imshow(torch.transpose(sample_data, 0, 2))
            ax[col_id].set_title(f'{severity_level}')
            ax[col_id].axis('off')
            break

        col_id += 1
        if col_id == len(severity_levels):
            col_id = 0

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)
    fig.suptitle(f"{Noise.__name__} - for various levels of severity", fontsize="x-large")
    fig.tight_layout()
    fig.savefig(rf'_plot_figures/CIFAR-10-C_simulated_{Noise.__name__}.png')
    fig.show()


class GaussianNoiseNumPy(object):
    def __init__(self, std, mean=0):
        self.std = std  # [.08, .12, 0.18, 0.26, 0.38]
        self.mean = mean

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, x):
        return np.clip(x + np.random.normal(size=x.shape, scale=self.std), 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ShotNoiseNumPy(object):
    def __init__(self, peak):
        self.peak = float(peak)  # [60, 25, 12, 5, 3]

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, x):
        return np.clip(np.random.poisson(x * self.peak) / self.peak, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(peak={0})'.format(self.peak)


class ImpulseNoiseNumPy(object):
    # Salt and pepper noise
    def __init__(self, amount):
        self.amount = float(amount)  # [.03, .06, .09, 0.17, 0.27]

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, x):
        return np.clip(sk.util.random_noise(np.array(x), mode='s&p', amount=self.amount), 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(amount={0})'.format(self.amount)


class SpeckleNoiseNumPy(object):
    def __init__(self, std):
        self.std = float(std)  # [.15, .2, 0.35, 0.45, 0.6]

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, x):
        return np.clip(x + x * np.random.normal(size=x.shape, scale=self.std), 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(std={0})'.format(self.std)


class DefocusBlurNumPy(object):
    def __init__(self, params):
        # [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        self.radius = params[0]
        self.alias_blur = params[1]

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, x):
        x = np.array(x)
        kernel = self.disk(radius=self.radius, alias_blur=self.alias_blur)

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[d, :, :], -1, kernel))
        channels = np.array(channels)  # .transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

        return np.clip(channels, 0, 1)

    @staticmethod
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    def __repr__(self):
        return self.__class__.__name__ + '(radius={0}, alias_blur={1})'.format(self.radius, self.alias_blur)


class GlassBlurNumPy(object):
    def __init__(self, params):
        # [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][severity - 1]
        self.sigma = params[0]
        self.max_delta = params[1]
        self.iterations = params[2]

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, x):
        # sigma, max_delta, iterations
        x = gaussian(np.array(x).transpose((1, 2, 0)), sigma=self.sigma, multichannel=True)

        # locally shuffle pixels
        for i in range(self.iterations):
            for h in range(32 - self.max_delta, self.max_delta, -1):
                for w in range(32 - self.max_delta, self.max_delta, -1):
                    dx, dy = np.random.randint(-self.max_delta, self.max_delta, size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        x = np.clip(gaussian(x, sigma=self.sigma, multichannel=True), 0, 1)
        x = x.transpose((2, 0, 1))
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0}, max_delta={1}, iter={2})'.format(self.sigma, self.max_delta,
                                                                                       self.iterations)


class MotionBlurNumPy(object):
    def __init__(self, params):
        self.radius = params[0]
        self.sigma = params[1]

    # Input tensor values are in the range [0, 1.0]
    def __call__(self, x):
        x = np.array(x).transpose((1, 2, 0))

        output = BytesIO()
        x.save(output, format='PNG')
        x = self.MotionImage(blob=output.getvalue())

        x.motion_blur(radius=self.radius, sigma=self.sigma, angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if x.shape != (32, 32):
            return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

        x = x.transpose((2, 0, 1))
        return x

    # Extend wand.image.Image class to include method signature
    class MotionImage(WandImage):
        def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
            wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0}, max_delta={1}, iter={2})'.format(self.sigma, self.max_delta,
                                                                                       self.iterations)


if __name__ == '__main__':
    # compare_methods_with_noisy_test()
    # plot_cifar_with_gaussian_noise()
    # plot_corruptions_from_CIFAR10C()
    plot_simulate_corruptions_from_CIFAR10C()
