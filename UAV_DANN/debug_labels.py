import pickle
import numpy as np

# 加载预处理数据
data_path = './data/results/single_condition/processed_data_condition_0.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print("=" * 60)
print("数据基本信息")
print("=" * 60)
for key in data.keys():
    if isinstance(data[key], np.ndarray):
        print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
    else:
        print(f"{key}: {data[key]}")

print("\n" + "=" * 60)
print("验证集标签详细分析")
print("=" * 60)

y_val = data['y_source_val']
print(f"验证集大小: {len(y_val)}")
print(f"唯一标签: {np.unique(y_val)}")
print(f"标签分布: {np.bincount(y_val)}")

# 详细分布
labels_map = {
    0: "No_Fault",
    1: "Motor", 
    2: "Accelerometer",
    3: "Gyroscope",
    4: "Magnetometer",
    5: "Barometer",
    6: "GPS"
}

print("\n按类别统计:")
for label_id, count in enumerate(np.bincount(y_val)):
    percentage = count / len(y_val) * 100
    print(f"  标签{label_id} ({labels_map.get(label_id, '未知')}): {count} 样本 ({percentage:.1f}%)")

print("\n" + "=" * 60)
print("训练集标签分析")
print("=" * 60)

y_train = data['y_source_train']
print(f"训练集大小: {len(y_train)}")
print(f"唯一标签: {np.unique(y_train)}")
print(f"标签分布: {np.bincount(y_train)}")

print("\n按类别统计:")
for label_id, count in enumerate(np.bincount(y_train)):
    percentage = count / len(y_train) * 100
    print(f"  标签{label_id} ({labels_map.get(label_id, '未知')}): {count} 样本 ({percentage:.1f}%)")

# 检查是否有文件名信息
if 'source_files' in data:
    print("\n" + "=" * 60)
    print("文件名示例")
    print("=" * 60)
    print(f"前5个文件: {data['source_files'][:5]}")
