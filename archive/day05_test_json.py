import os

# 写死绝对路径，看看 Python 到底能看到什么
target_path = r"C:\AI\1010\data\processed\chunks.json"
print("正在检查路径：", target_path)
print("文件是否存在：", os.path.exists(target_path))

if os.path.exists(target_path):
    with open(target_path, "r", encoding="utf-8") as f:
        print("成功读取！文件前 100 个字符：", f.read(100))
else:
    print("❌ 还是找不到！请去文件夹里把“文件扩展名”打开，看看真实名字到底是什么。")