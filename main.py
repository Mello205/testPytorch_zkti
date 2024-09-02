import torch

# import torchvision
# import torchvision.transforms as transforms

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    device_id = 1  # 选择ID为1的GPU
    print(torch.cuda.is_available())
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.current_device())


