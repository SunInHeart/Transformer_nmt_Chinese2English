import gradio as gr
import torch
import numpy as np
from nmt_model import make_model
from nmt_model import greedy_decode
from nmt_model import PrepareData

# 确保在所有地方都设置Matplotlib的后端为非交互式后端
# import matplotlib
# matplotlib.use('agg')

PAD = 0                             # padding占位符的索引
UNK = 1                             # 未登录词标识符的索引
LAYERS = 6                          # transformer中encoder、decoder层数
H_NUM = 8                           # 多头注意力个数
D_MODEL = 256                       # 输入、输出词向量维数
D_FF = 1024                         # feed forward全连接层维数
DROPOUT = 0.1                       # dropout比例
MAX_LENGTH = 60                     # 语句最大长度

TRAIN_FILE = 'nmt/cn-en/train.txt'  # 训练集
DEV_FILE = "nmt/cn-en/dev.txt"  # 验证集
SAVE_FILE = 'save/model.pt'         # 模型保存路径
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'

torch.manual_seed(0)

# 数据预处理
data = PrepareData(TRAIN_FILE, DEV_FILE)
src_vocab = len(data.cn_word_dict)
tgt_vocab = len(data.en_word_dict)
print("src_vocab %d" % src_vocab)
print("tgt_vocab %d" % tgt_vocab)

# 初始化模型
model = make_model(src_vocab, tgt_vocab, LAYERS, D_MODEL, D_FF, H_NUM, DROPOUT)

# 加载模型
model.load_state_dict(torch.load(SAVE_FILE))


# 预测
def predict(sentence):
    # 将单词映射为索引
    out_cn_ids = [data.cn_word_dict.get(word, UNK) for word in sentence]

    # 将当前以单词id表示的中文语句数据转为tensor，并放入DEVICE中
    src = torch.from_numpy(np.array(out_cn_ids)).long().to(DEVICE)
    # 增加一维
    src = src.unsqueeze(0)
    # 设置attention mask
    src_mask = (src != 0).unsqueeze(-2)
    # 用训练好的模型进行decode预测
    out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])

    # 初始化一个用于存放模型翻译结果语句单词的列表
    translation = []
    # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
    for j in range(1, out.size(1)):
        # 获取当前下标的输出字符
        sym = data.en_index_dict[out[0, j].item()]
        # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
        if sym != 'EOS':
            translation.append(sym)
        # 否则终止遍历
        else:
            break
    # 打印模型翻译输出的英文语句结果
    print("translation: %s" % " ".join(translation))
    return " ".join(translation)


iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="中文到英文翻译")
iface.launch(server_name='localhost', server_port=9870)
