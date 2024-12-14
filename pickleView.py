import pickle
import numpy as np
import argparse
from pprint import pprint

def load_and_display_pickle(file_path):
    """
    加载并显示pickle文件的内容
    
    Args:
        file_path (str): pickle文件的路径
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print("\n=== Pickle文件内容概览 ===")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print("\n字典键:")
            for key in data.keys():
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"- {key}: numpy数组, 形状={value.shape}, 类型={value.dtype}")
                else:
                    print(f"- {key}: {type(value)}")
            
            print("\n详细内容:")
            pprint(data)
            
        elif isinstance(data, np.ndarray):
            print(f"\n数组形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print("\n数组内容预览:")
            print(data)
            
        else:
            print("\n内容:")
            pprint(data)
            
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查看pickle文件内容的工具')
    parser.add_argument('file_path', type=str, help='pickle文件的路径')
    args = parser.parse_args()
    
    load_and_display_pickle(args.file_path)