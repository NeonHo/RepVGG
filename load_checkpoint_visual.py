import copy
from matplotlib import pyplot as plt
import torch
from hmquant.qat_torch.apis import trace_model_for_qat
from hmquant.qat_torch.operations.xh2.rep import QReparameterizedBlock
from repvgg import RepVGGBlock
from repvggplus import create_RepVGGplus_by_name
from hmquant.observers.utils import pure_diff
from rich.table import Table
from rich.console import Console


resume_pth = "ckpt_dir/RepVGG-A0/QAT2/RepVGG-A0-train.pth"
# quant_resume_pth = "ckpt_dir/RepVGG-A0/QAT/ckpt_epoch_0.pth"
quant_resume_pth = "ckpt_dir/RepVGG-A0/QAT/ckpt_epoch_0.pth"
arch = "RepVGG-A0"
bins_num = 256

model = create_RepVGGplus_by_name(arch, deploy=False, use_checkpoint=True)
quant_model = create_RepVGGplus_by_name(arch, deploy=False, use_checkpoint=True)
quant_cfg = dict(
    global_wise_cfg=dict(
        o_cfg=dict(calib_metric="minmax", dtype="int8", use_grad_scale=False, symmetric=False), 
        freeze_bn=False,
        w_cfg=dict(calib_metric="minmax", dtype="int8", use_grad_scale=False)
    )
)
quant_model = trace_model_for_qat(copy.deepcopy(quant_model.eval()), quant_cfg, domain="xh2")

checkpoint = torch.load(resume_pth, map_location='cpu')
quant_checkpoint = torch.load(quant_resume_pth, map_location='cpu')

msg = model.load_state_dict(checkpoint, strict=True)
quant_msg = quant_model.load_state_dict(quant_checkpoint["model"], strict=True)

quant_model.eval()

block_idx = 0

quant_model_dict = dict(quant_model.named_modules())

error_record_ops_dict = {}
for key, module in model.named_modules():
    if isinstance(module, RepVGGBlock):
        print(key)
        quant_module = quant_model_dict[key]
        channel_num = module.rbr_dense.conv.weight.shape[0]
        kernel, bias = quant_module.construct_kernel(x=None)
        for channel_idx in range(channel_num):
            if channel_idx % 16 != 0:
                continue
            weight_per_channel = kernel[channel_idx, :, :, :]
            quant_weight_per_channel = quant_module.w_quantizer(kernel).data[channel_idx, :, :, :]
            numpy_data = weight_per_channel.detach().numpy().flatten()
            quant_numpy_data = quant_weight_per_channel.detach().numpy().flatten()
            mean_ab_tensor, mean_re_tensor, related_mse, cos = pure_diff(raw_tensor=weight_per_channel, quanted_tensor=quant_weight_per_channel)
            error_record_ops_dict[key + "-{}".format(channel_idx)] = {}
            error_record_ops_dict[key + "-{}".format(channel_idx)]["w/a"] = "activation"
            error_record_ops_dict[key + "-{}".format(channel_idx)]["mean L1 Error"] = str(mean_ab_tensor.item())
            error_record_ops_dict[key + "-{}".format(channel_idx)]["mean related L1 Error"] = str(mean_re_tensor.item())
            error_record_ops_dict[key + "-{}".format(channel_idx)]["related mse"] = str(related_mse.item())
            error_record_ops_dict[key + "-{}".format(channel_idx)]["cosine dist"] = str(cos.item())
            plt.figure(figsize=(8, 8))
            plt.hist(numpy_data, bins=bins_num, alpha=0.3, label="FP")
            plt.hist(quant_numpy_data, bins=bins_num, alpha=0.6, label=quant_cfg["global_wise_cfg"]["w_cfg"]["dtype"])
            plt.legend()
            plt.xlabel("weight")
            plt.ylabel("frequency")
            plt.title("{}-block({})-channel({})-scale({})".format(arch, block_idx, channel_idx, quant_module.w_quantizer.scale[channel_idx]))
            plt.savefig('ckpt_dir/RepVGG-A0/histogram({})-block({})-channel({}).pdf'.format(arch, block_idx, channel_idx))
        block_idx += 1
table = Table(title="Sequencer Analyse Report -- activation")
table.add_column("Module")
table_data = []
for node_key, node_metric_dict in error_record_ops_dict.items():
    node_key_list = [node_key]
    for metric_value in node_metric_dict.values():
        node_key_list.append(metric_value)
    table_data.append(node_key_list)
for metric_key in error_record_ops_dict[next(iter(error_record_ops_dict))].keys():
    table.add_column(metric_key)
for row in table_data:
    table.add_row(*row)
    
console = Console(record=True)
console.print(table)

max_accuracy = 0.0