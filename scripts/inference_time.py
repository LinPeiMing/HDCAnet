import torch
from basicsr.archs.HDCAnet_arch import BlindSSR

iterations = 300   # 重复计算的轮次

model = BlindSSR()
device = torch.device("cuda:0")
model.to(device)

data = torch.randn(1, 6, 64, 64).to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# GPU预热
for _ in range(50):
    _ = model(data)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(data)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

