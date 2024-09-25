import os

sum=0
# 定义函数来计数文件和子目录
def count_files_and_subfolders(path):
    file_count = len(os.listdir(path))
    subfolder_count = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    sum=file_count
    print(f"Subdirectory {path} Files: {file_count} Subfolders: {subfolder_count}")

    # 递归检查子目录
    for subdir in sorted([os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]):
        sum+=count_files_and_subfolders(subdir)
    return sum

# 调用函数，从根目录开始
print(count_files_and_subfolders('data_zoo/imagenet1krgbd_imgemb/train'))