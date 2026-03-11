#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x):
    """sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def silu(x):
    """SiLU激活函数 (Swish): x * sigmoid(x)"""
    return x * sigmoid(x)

def generate_silu_lut(bit_width=16, lut_size=256, input_range=8.0):
    """
    生成SiLU激活函数的查找表
    
    参数：
    - bit_width: 数据位宽（默认16位）
    - lut_size: 查找表大小（默认256项）
    - input_range: 输入范围（-input_range到+input_range）
    """
    
    print("="*60)
    print("SiLU激活函数查找表生成器")
    print("="*60)
    print(f"配置参数：")
    print(f"  - 数据位宽: {bit_width} bits")
    print(f"  - LUT大小: {lut_size} entries")
    print(f"  - 输入范围: [{-input_range}, {input_range}]")
    print()
    
    # 创建输入值数组
    x_values = np.linspace(-input_range, input_range, lut_size)
    
    # 计算SiLU值
    silu_values = silu(x_values)
    
    # 量化到固定点表示
    # 使用Q8.8格式（8位整数，8位小数）
    scale_factor = 2**(bit_width//2)
    silu_quantized = np.round(silu_values * scale_factor).astype(int)
    
    # 限制在位宽范围内
    max_val = 2**(bit_width-1) - 1
    min_val = -2**(bit_width-1)
    silu_quantized = np.clip(silu_quantized, min_val, max_val)
    
    return x_values, silu_values, silu_quantized, scale_factor

def save_lut_to_file(lut_data, filename, format='hex'):
    """
    保存查找表到文件
    
    参数：
    - lut_data: 查找表数据
    - filename: 输出文件名
    - format: 输出格式 ('hex', 'bin', 'coe')
    """
    
    print(f"保存LUT到文件: {filename}")
    
    if format == 'hex':
        with open(filename, 'w') as f:
            f.write("// SiLU激活函数查找表 - Hexadecimal格式\n")
            f.write("// 每行一个16位十六进制值\n")
            f.write(f"// 总共 {len(lut_data)} 个条目\n\n")
            
            for i, value in enumerate(lut_data):
                # 转换为无符号表示
                if value < 0:
                    value = (1 << 16) + value
                f.write(f"{value:04X}  // 索引 {i:3d}\n")
                
    elif format == 'coe':
        # Xilinx COE格式（用于Block RAM初始化）
        with open(filename, 'w') as f:
            f.write("; SiLU激活函数查找表 - COE格式\n")
            f.write("; 用于Xilinx Block RAM初始化\n")
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            
            for i, value in enumerate(lut_data):
                if value < 0:
                    value = (1 << 16) + value
                    
                if i < len(lut_data) - 1:
                    f.write(f"{value:04X},\n")
                else:
                    f.write(f"{value:04X};\n")
                    
    elif format == 'bin':
        # 二进制格式
        with open(filename, 'w') as f:
            f.write("// SiLU激活函数查找表 - Binary格式\n")
            for i, value in enumerate(lut_data):
                if value < 0:
                    value = (1 << 16) + value
                f.write(f"{value:016b}  // 索引 {i:3d}\n")
    
    print(f"  文件保存成功！")

def plot_silu_function(x_values, silu_values, silu_quantized, scale_factor):
    """绘制SiLU函数图形"""
    
    plt.figure(figsize=(12, 5))
    
    # 子图1：原始SiLU函数
    plt.subplot(1, 2, 1)
    plt.plot(x_values, silu_values, 'b-', linewidth=2, label='SiLU (float)')
    plt.plot(x_values, silu_quantized/scale_factor, 'r--', 
             linewidth=1, label='SiLU (quantized)')
    plt.grid(True, alpha=0.3)
    plt.xlabel('输入值 x')
    plt.ylabel('SiLU(x) = x * sigmoid(x)')
    plt.title('SiLU激活函数')
    plt.legend()
    
    # 子图2：量化误差
    plt.subplot(1, 2, 2)
    error = silu_values - silu_quantized/scale_factor
    plt.plot(x_values, error * 100, 'g-', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('输入值 x')
    plt.ylabel('误差 (%)')
    plt.title('量化误差分析')
    
    plt.tight_layout()
    plt.savefig('silu_function_plot.png', dpi=150)
    plt.show()
    
    print("\n图形已保存为: silu_function_plot.png")

def generate_testbench_data(num_tests=100):
    """生成测试激励数据"""
    
    print("\n生成测试数据...")
    
    # 生成随机测试输入
    test_inputs = np.random.uniform(-8, 8, num_tests)
    
    # 计算期望输出
    expected_outputs = silu(test_inputs)
    
    # 保存测试数据
    with open('test_vectors.txt', 'w') as f:
        f.write("// SiLU测试向量\n")
        f.write("// 格式: 输入值, 期望输出\n\n")
        
        for inp, out in zip(test_inputs, expected_outputs):
            f.write(f"{inp:8.4f}, {out:8.4f}\n")
    
    print(f"  生成了 {num_tests} 个测试向量")
    print(f"  保存到: test_vectors.txt")

def main():
    """主函数"""
    
    # 设置输出目录
    output_dir = "./lut_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    os.chdir(output_dir)
    
    # 生成LUT
    x_vals, silu_float, silu_quant, scale = generate_silu_lut(
        bit_width=16,
        lut_size=256,
        input_range=8.0
    )
    
    # 保存到不同格式的文件
    print("\n保存LUT文件...")
    save_lut_to_file(silu_quant, 'silu_lut.hex', 'hex')
    save_lut_to_file(silu_quant, 'silu_lut.coe', 'coe')
    save_lut_to_file(silu_quant, 'silu_lut.bin', 'bin')
    
    # 生成测试数据
    generate_testbench_data(100)
    
    # 绘制函数图
    print("\n生成可视化图形...")
    plot_silu_function(x_vals, silu_float, silu_quant, scale)
    
    # 打印统计信息
    print("\n统计信息：")
    print(f"  最大量化误差: {np.max(np.abs(silu_float - silu_quant/scale)):.6f}")
    print(f"  平均量化误差: {np.mean(np.abs(silu_float - silu_quant/scale)):.6f}")
    print(f"  查找表大小: {len(silu_quant) * 2} bytes")
    
    print("\n完成！所有文件已生成。")

if __name__ == "__main__":
    main()
