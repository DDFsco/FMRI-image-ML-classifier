"""
Verify that labels are correctly assigned
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainmapnet.dataset import fMRIDataset

def main():
    data_root = "dataset_combined_vertical"
    subjects = ['pmindo235']
    
    ds = fMRIDataset(data_root, subjects, 'test', 224)
    
    print("=" * 60)
    print("验证标签分配")
    print("=" * 60)
    print(f"\nSubject: {subjects[0]}")
    print(f"总图像数: {len(ds)}")
    print("\n所有12张图的标签分配:")
    print("-" * 60)
    
    for i in range(12):
        path = ds.image_paths[i]
        label = ds.labels[i]
        # Extract cond number from path
        cond_num = int(path.split('cond')[1].split('_')[0])
        class_name = ds.get_class_name(label)
        expected_label = 1 if cond_num <= 6 else 0
        expected_class = "house" if cond_num <= 6 else "face"
        
        status = "✓" if label == expected_label else "✗"
        print(f"{status} cond{cond_num:2d}: label={label} ({class_name:5s}) [期望: {expected_label} ({expected_class})]")
    
    print("-" * 60)
    
    # Summary
    house_count = sum(1 for l in ds.labels if l == 1)
    face_count = sum(1 for l in ds.labels if l == 0)
    
    print(f"\n总结:")
    print(f"  House (label=1): {house_count} 张 (应该是6张)")
    print(f"  Face  (label=0): {face_count} 张 (应该是6张)")
    
    if house_count == 6 and face_count == 6:
        print("\n✓ 标签分配正确！")
    else:
        print("\n✗ 标签分配有误！")

if __name__ == "__main__":
    main()

