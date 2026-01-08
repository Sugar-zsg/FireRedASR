import torch

input_bin = "/workspace/bella-infra/user/zhangshuge002/master/realtime/train/test_train/FireRedASR/exp/custom_dataset_0108_test/stage2/best-model.pt/pytorch_model.bin"      # 你的 bin 文件路径
output_pt = "adapter_converted_with_lora.pt"   # 转换后的输出路径

print(f"Loading {input_bin}...")
state_dict = torch.load(input_bin, map_location='cpu')

print("\nFirst 5 keys in bin file:")
keys = list(state_dict.keys())[:]
for k in keys:
    print(f"  {k}")

checkpoint = {
    "model": state_dict
}

print(f"\nSaving to {output_pt}...")
torch.save(checkpoint, output_pt)
print("Done! Now use this .pt file with package_model.py")
