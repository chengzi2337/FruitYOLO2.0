#!/usr/bin/env python3
"""
将 YOLOv8 的 .pt 模型文件转换为 .ckpt 格式
支持保留完整训练状态或仅保存模型权重
"""

import torch
import argparse
from pathlib import Path


def convert_pt_to_ckpt(pt_file, output_file=None, weights_only=False):
    """
    转换 .pt 文件为 .ckpt 格式
    
    Args:
        pt_file: 输入的 .pt 文件路径
        output_file: 输出的 .ckpt 文件路径（可选）
        weights_only: 是否仅保存模型权重（不包含优化器状态等）
    """
    pt_path = Path(pt_file)
    
    if not pt_path.exists():
        raise FileNotFoundError(f"找不到文件: {pt_file}")
    
    # 生成输出文件名
    if output_file is None:
        output_file = pt_path.with_suffix('.ckpt')
    else:
        output_file = Path(output_file)
    
    print(f"正在加载模型: {pt_path}")
    # 由于使用 ultralytics 自定义模型，需要设置 weights_only=False 以允许反序列化
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # 显示原始checkpoint内容
    print(f"\n原始checkpoint包含的键:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"  - {key}")
    
    # 准备要保存的数据
    if weights_only:
        # 仅保存模型权重
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                save_dict = {'state_dict': checkpoint['model'].state_dict()}
            elif 'state_dict' in checkpoint:
                save_dict = checkpoint
            else:
                # 假设整个checkpoint就是state_dict
                save_dict = {'state_dict': checkpoint}
        else:
            # checkpoint本身可能就是模型对象
            save_dict = {'state_dict': checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint}
        
        print(f"\n保存模式: 仅权重")
    else:
        # 保存完整checkpoint（包含优化器、epoch等信息）
        save_dict = checkpoint
        print(f"\n保存模式: 完整checkpoint")
    
    # 保存为.ckpt格式
    print(f"正在保存到: {output_file}")
    torch.save(save_dict, output_file)
    
    # 验证保存的文件
    print(f"\n验证转换结果...")
    loaded = torch.load(output_file, map_location='cpu', weights_only=False)
    print(f"✓ 成功保存，文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    if isinstance(loaded, dict):
        print(f"✓ 保存的checkpoint包含的键:")
        for key in loaded.keys():
            print(f"    - {key}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='将YOLOv8的.pt文件转换为.ckpt格式')
    parser.add_argument('input', type=str, help='输入的.pt文件路径')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='输出的.ckpt文件路径（默认：与输入文件同名但扩展名为.ckpt）')
    parser.add_argument('-w', '--weights-only', action='store_true',
                        help='仅保存模型权重，不包含优化器状态等训练信息')
    
    args = parser.parse_args()
    
    try:
        output_path = convert_pt_to_ckpt(
            args.input, 
            args.output, 
            args.weights_only
        )
        print(f"\n✅ 转换成功! 输出文件: {output_path}")
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
