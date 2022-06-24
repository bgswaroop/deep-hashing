import json

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10


class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

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
    plt.savefig('impact_of_gaussian_noise_inhibition1.png', bbox_inches="tight")
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
    fig, ax = plt.subplots(2, 10, figsize=(15, 3.3))
    dataset_dir = r'/data/p288722/datasets/cifar'
    num_tests = 20
    row_id = 0
    col_id = 0
    for std in [float(x) for x in torch.arange(0, 0.20, 0.20 / num_tests)][:num_tests]:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(mean=0, std=std),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_split_test = CIFAR10(dataset_dir, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(data_split_test, batch_size=32)

        for data, label in test_loader:
            sample_data = data[4]
            plt.figure()
            ax[row_id][col_id].imshow(torch.transpose(sample_data, 0, 2))
            ax[row_id][col_id].set_title(f'{round(std, 3)}')
            ax[row_id][col_id].axis('off')
            break

        col_id += 1
        if col_id == 10:
            col_id = 0
            row_id += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_methods_with_noisy_test()
    # plot_cifar_with_gaussian_noise()
