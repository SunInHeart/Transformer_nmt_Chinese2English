import opencc

def convert_traditional_to_simplified(input_file, output_file):
    # 创建一个OpenCC实例，用于繁体中文到简体中文的转换
    cc = opencc.OpenCC('t2s.json')

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        traditional_text = f.read()

    # 将繁体中文转换为简体中文
    simplified_text = cc.convert(traditional_text)

    # 将转换后的文本写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(simplified_text)

    print(f"转换完成！简体中文已保存到 {output_file}")

# 示例：将 input.txt 中的繁体中文转换为 output.txt 中的简体中文
convert_traditional_to_simplified('train.txt', 'train_sim.txt')
