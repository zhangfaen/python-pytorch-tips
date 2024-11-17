ImportError: /usr/local/lib/python3.10/dist-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12

-----

**ChatGPT 回答的 （根本原因：在.zshrc中 定义了CUDA_HOME=/usr/local/cuda-12.1，导致系统使用的是cuda-12.1的库，而pytorch是cuda-12.4的库，导致找不到符号； 把.zshrc中CUDA相关的环境变量去除就好了）**


这个错误表明 PyTorch 在尝试加载 `libcusparse.so.12` 时，无法找到符号 `__nvJitLinkAddData_12_1`，可能是因为 **PyTorch 安装的 CUDA 版本与系统的 NVIDIA 驱动或 CUDA 库版本不匹配**。

以下是解决此问题的步骤：

---

### **1. 检查 NVIDIA 驱动版本**
确保系统的 NVIDIA 驱动程序版本支持 PyTorch 使用的 CUDA 版本：

```bash
nvidia-smi
```

在输出中，检查驱动程序版本。例如，如果您使用的是 CUDA 12.x，则 NVIDIA 驱动需要满足以下最低要求：
- CUDA 12.0 需要驱动版本 ≥ 525.60.13
- CUDA 12.1 需要驱动版本 ≥ 525.85.12

如果驱动版本较低，请升级驱动程序。

---

### **2. 检查 PyTorch 的 CUDA 版本**
确认您安装的 PyTorch 版本使用的 CUDA 版本：

```python
import torch
print(torch.version.cuda)  # 显示 PyTorch 使用的 CUDA 版本
```

假如输出为 `12.1`，则您需要确保驱动版本与 CUDA 12.1 兼容。

---

### **3. 检查系统中的 CUDA 库**
如果系统安装了多个 CUDA 工具包，可能导致 PyTorch 链接到错误的库版本：

- 检查 `libcusparse.so.12` 所在的位置：
  ```bash
  find /usr/local -name libcusparse.so.12
  ```
- 如果找到多个库路径，请确保路径与 PyTorch 使用的版本一致。

---

### **4. 确保安装正确的 PyTorch 版本**
如果问题是因为安装了不正确的 PyTorch CUDA 版本，可以重新安装与驱动和硬件匹配的 PyTorch：

#### **a. 卸载 PyTorch**
```bash
pip uninstall torch torchvision torchaudio
```

#### **b. 安装合适的版本**
根据您的系统环境和驱动版本，选择适配的 PyTorch CUDA 版本。以下是一些示例：

- 安装支持 CUDA 11.8 的 PyTorch：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- 安装支持 CUDA 12.1 的 PyTorch：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

如果不确定，请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的安装命令。

---

### **5. 临时解决方法**
如果问题依然存在，可能是 `LD_LIBRARY_PATH` 中存在冲突的库路径。可以尝试以下方法：

#### **a. 设置 CUDA 环境变量**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### **b. 强制指定动态库路径**
运行程序时，显式指定 PyTorch 使用的动态库路径：
```bash
LD_PRELOAD=/usr/local/cuda/lib64/libcusparse.so.12 python your_script.py
```

---

### **6. 确保软件环境一致性**
如果问题仍未解决，可能需要重新配置您的环境：
1. 在干净的环境中安装 PyTorch（如使用虚拟环境 `venv` 或 `conda`）。
2. 检查是否存在软件版本冲突。

---

### **总结**
- **核心问题**：PyTorch 使用的 CUDA 版本与系统环境不匹配。
- **解决方法**：
  - 确保驱动与 CUDA 版本兼容。
  - 安装与驱动匹配的 PyTorch CUDA 版本。
  - 设置正确的动态库路径避免库冲突。

如果仍然有疑问，请提供更多环境信息（如 GPU 型号、NVIDIA 驱动版本、PyTorch 安装命令等），以便进一步帮助您！