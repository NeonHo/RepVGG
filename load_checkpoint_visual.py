import copy
from matplotlib import pyplot as plt
import torch
from hmquant.qat_torch.apis import trace_model_for_qat
from hmquant.qat_torch.operations.xh2.rep import QReparameterizedBlock
from repvgg import RepVGGBlock
from repvggplus import create_RepVGGplus_by_name


resume_pth = "ckpt_dir/RepVGG-A0/QAT2/RepVGG-A0-train.pth"
quant_resume_pth = "ckpt_dir/RepVGG-A0/QAT/ckpt_epoch_0.pth"
arch = "RepVGG-A0"
bins_num = 256

model = create_RepVGGplus_by_name(arch, deploy=False, use_checkpoint=True)
quant_model = create_RepVGGplus_by_name(arch, deploy=False, use_checkpoint=True)
quant_cfg = dict(
    global_wise_cfg=dict(
        o_cfg=dict(calib_metric="minmax", dtype="int8", use_grad_scale=False), 
        freeze_bn=False,
        w_cfg=dict(calib_metric="minmax", dtype="int8", use_grad_scale=False)
    )
)
quant_model = trace_model_for_qat(copy.deepcopy(quant_model.train()), quant_cfg, domain="xh2")

checkpoint = torch.load(resume_pth, map_location='cpu')
quant_checkpoint = torch.load(quant_resume_pth, map_location='cpu')

msg = model.load_state_dict(checkpoint, strict=True)
quant_msg = quant_model.load_state_dict(quant_checkpoint["model"], strict=True)

block_idx = 0

quant_model_dict = dict(quant_model.named_modules())

for key, module in model.named_modules():
    if isinstance(module, RepVGGBlock):
        print(key)
        quant_module = quant_model_dict[key]
        channel_num = module.rbr_dense.conv.weight.shape[0]
        for channel_idx in range(channel_num):
            if channel_idx % 16 != 0:
                continue
            weight_per_channel = module.rbr_dense.conv.weight[channel_idx, :, :, :]
            quant_weight_per_channel = quant_module.rbr_dense.conv.weight[channel_idx, :, :, :]
            numpy_data = weight_per_channel.detach().numpy().flatten()
            quant_numpy_data = quant_weight_per_channel.detach().numpy().flatten()
            plt.figure(figsize=(8, 8))
            plt.hist(numpy_data, bins=bins_num, alpha=0.3, label="FP")
            plt.hist(quant_numpy_data, bins=bins_num, alpha=0.6, label=quant_cfg["global_wise_cfg"]["w_cfg"]["dtype"])
            plt.legend()
            plt.xlabel("weight")
            plt.ylabel("frequency")
            plt.title("{}-block({})-channel({})-scale({})".format(arch, block_idx, channel_idx, quant_module.w_quantizer.scale[channel_idx]))
            plt.savefig('ckpt_dir/RepVGG-A0/histogram({})-block({})-channel({}).pdf'.format(arch, block_idx, channel_idx))
        block_idx += 1

max_accuracy = 0.0