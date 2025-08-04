import torch as th

if th.cuda.is_available():
    gpu_count = th.cuda.device_count()
    print(f"偵測到 {gpu_count} 個可用的 GPU。")
    for i in range(gpu_count):
        print(f"  GPU {i}: {th.cuda.get_device_name(i)}")
else:
    print("未偵測到任何可用的 GPU。")