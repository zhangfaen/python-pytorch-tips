
    Unet-model-number-of-parameters-and-total-flops.md
    
    ```python
    from ptflops import get_model_complexity_info
    from fvcore.nn import FlopCountAnalysis 

    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    flops = FlopCountAnalysis(net, torch.randn(1, 3, 572, 572))
    print("total flops: ", flops.total())
    print("flops.by_module_and_operator():", flops.by_module_and_operator())

    macs, params = get_model_complexity_info(net, (3, 572, 572), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params)) 
    ```

    # EMED-UNet crosses the accuracy of U-Net with significantly reduced parameters from 31.043M to 6.72M, FLOPs from 386 G to 114 G, and model size from 386 MB to 80 MB. 
    # output: 
    # Computational complexity:       199.22 GMac (Giga(10 ** 9) multiply or add calculation, 有的时候也用 FLOPs : floating point operations)
    # Number of parameters:           17.26 M 

    <!-- 最后的输出大小为Cout*Mh*Mw

    参数量：
    Cout*Cin*Kh*Kw -->

    # 计算量：
    # 卷积乘：
    # (Kw*Kh)*(Mh*Mw)*(Cin*Cout)

    # 卷积加：
    # Kw*Kh*Cin-1*Mh*Mw*Cout

    # Ps：n个数相加，相加次数是n-1。

    # Bias加：
    # Mh*Mw*Cout

    # 总：
    # No Bias：（2*Kw*Kh*Cin-1）*Mh*Mw*Cout

    # bias: 2*Kw*Kh*Cin*Mh*Mw*Cout
