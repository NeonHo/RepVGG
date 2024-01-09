from matplotlib import pyplot as plt
import torch
from repvgg import RepVGGBlock

from repvggplus import create_RepVGGplus_by_name


resume_pth = "ckpt_dir/RepVGG-A0/QAT1/ckpt_epoch_20.pth"
arch = "RepVGG-A0"

model = create_RepVGGplus_by_name(arch, deploy=False, use_checkpoint=True)

checkpoint = torch.load(resume_pth, map_location='cpu')

msg = model.load_state_dict(checkpoint, strict=False)

for module in model.modules():
    if isinstance(module, RepVGGBlock):
        print(module)
        channel_num = module.rbr_dense.conv.weight.shape[0]
        for channel_idx in range(channel_num):
            weight_per_channel = module.rbr_dense.conv.weight[channel_idx, :, :, :]
            numpy_data = weight_per_channel.detach().numpy().flatten()
            plt.figure(figsize=(8, 8))
            plt.hist(numpy_data, bins=1024)
            if hasattr(module.rbr_dense.conv.weight, "scale"):
                plt.title("{}-scale({})".format(arch, module.rbr_dense.conv.weight.scale[channel_idx]))
            else:
                plt.title("{}".format(arch))
            plt.savefig('ckpt_dir/RepVGG-A0/histogram{}.pdf'.format(channel_idx))

max_accuracy = 0.0