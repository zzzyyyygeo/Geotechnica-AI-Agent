# 模拟一组基坑监测点的沉降数据（单位：mm）
settlements = [12.5, 8.3, 15.6, 22.1, 9.8, 18.4]

# 用for循环遍历，找出超过预警值（15mm）的点
warning_threshold = 15.0
print("【超限预警点】")
for index, value in enumerate(settlements):
    if value > warning_threshold:
        print(f"监测点 {index + 1}: 沉降 {value} mm，已超限！")

# 条件判断：工程安全分级
def check_safety(settlement):
    """根据沉降值判断安全等级"""
    if settlement < 10:
        return "安全（绿色）"
    elif settlement < 20:
        return "注意（黄色）"
    else:
        return "危险（红色）"

# 测试函数（注意这里要用 print）
print(check_safety(8.5))
print(check_safety(16.0))
print(check_safety(25.0))