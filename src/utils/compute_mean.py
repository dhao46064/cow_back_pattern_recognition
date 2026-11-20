import csv
import os


def calculate_and_append_averages(csv_path, skip_header=True, encoding='utf-8'):
    """
    读取CSV文件计算每列平均值，并将平均值追加到文件最后一行
    （纯Python内置库实现，无需安装任何依赖）

    Args:
        csv_path (str): CSV文件的路径（绝对/相对路径）
        skip_header (bool): 是否跳过表头行，默认True
        encoding (str): 文件编码，默认utf-8，乱码自动重试gbk
    """
    # 第一步：读取文件并计算每列平均值
    col_data = {}
    headers = []
    max_col_count = 0  # 记录文件最大列数，保证追加行对齐

    try:
        # 读取文件计算平均值
        with open(csv_path, 'r', encoding=encoding, newline='') as f:
            csv_reader = csv.reader(f)

            # 处理表头
            if skip_header:
                headers = next(csv_reader)
                max_col_count = len(headers)
                print(f"检测到表头：{headers}")
            else:
                print("未跳过表头，将第一行视为数据行")

            # 遍历数据行统计
            row_count = 0
            for row in csv_reader:
                row_count += 1
                max_col_count = max(max_col_count, len(row))  # 更新最大列数
                for col_idx, value in enumerate(row):
                    try:
                        num = float(value.strip())
                        if col_idx not in col_data:
                            col_data[col_idx] = {'sum': 0.0, 'count': 0}
                        col_data[col_idx]['sum'] += num
                        col_data[col_idx]['count'] += 1
                    except ValueError:
                        continue

            if row_count == 0:
                print("警告：文件中无数据行，无需追加平均值")
                return

            # 计算每列平均值（按最大列数补全，保证对齐）
            average_row = []
            print("\n===== 各列平均值 =====")
            for col_idx in range(max_col_count):
                if col_idx in col_data and col_data[col_idx]['count'] > 0:
                    avg = col_data[col_idx]['sum'] / col_data[col_idx]['count']
                    avg_str = f"{avg:.6f}"  # 保留6位小数
                    # 显示列名（有表头用表头，无表头用列索引）
                    col_name = headers[col_idx] if headers and col_idx < len(headers) else f"列{col_idx}"
                    print(f"{col_name}: {avg_str}")
                else:
                    avg_str = ""  # 无数据的列填空值，保证格式对齐
                average_row.append(avg_str)

        # 第二步：将平均值行追加到CSV文件末尾
        with open(csv_path, 'a', encoding=encoding, newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(average_row)  # 写入平均值行
        print(f"\n✅ 平均值已成功追加到文件：{csv_path} 的最后一行")

    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_path}，请检查路径是否正确")
    except UnicodeDecodeError:
        print(f"错误：编码{encoding}读取失败，尝试用gbk编码重新读取...")
        calculate_and_append_averages(csv_path, skip_header, encoding='gbk')
    except PermissionError:
        print(f"错误：无写入权限！请关闭正在打开的 {csv_path} 文件后重试")
    except Exception as e:
        print(f"发生错误：{str(e)}")


# ==================== 调用示例 ====================
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    # Windows示例：csv_path = "C:\\Users\\你的用户名\\Desktop\\data.csv"
    # Mac/Linux示例：csv_path = "/Users/你的用户名/Desktop/data.csv"
    csv_path = (r"D:\code_project\code_cp\obb_verification_results_3.3_V-1\global_summary.csv")  # 改成你的文件路径

    # 调用函数（有表头则skip_header=True，无表头则False）
    calculate_and_append_averages(csv_path, skip_header=True)