import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from train import create_model
import torch.nn.functional as F
import codecs
import json
PAD = '[PAD]'
pad_id = 0

PAD = '[PAD]'
pad_id = 0
logger = None
name_map_aspect = {"电影名": "moviename","导演": "director","演员名":"actor", "类型":"movie_type" ,"国家":"country" ,"上映时间":"time" , "角色":"role","剧情":"plot","台词":"lines","奖项":"award" ,
            "票房":"income","评分":"rating","资源":"website", "音乐":"music", "其他":"aspect_others"}
name_map_action ={"请求事实":"request_fact", "请求推荐":"request_rec","请求感受":"request_feeling","告知事实":"inform_fact","告知推荐":"inform_rec","告知感受":"inform_feeling","其他":"others"}

SPECIAL_TOKENS = {
"start_context" : "[context]",
"end_context" : "[endofcontext]",
"start_action" : "[action]",
"end_action" : "[endofaction]",
"start_know": "[knowledge]",
"end_know": "[endofknowledge]",
"start_response":"[response]",
"end_response":"[endofresponse]",
"user":"[user]",
"system": "[system]",
}
root="./"
#root ="./"
def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--model_config', default=root+'config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default=root+'vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default=root+'new_data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default=root+'movie_data/valid_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练预料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default=root+'test.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--raw', action='store_true', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--model_path', default=root+ 'ul_model_best/model_epoch8/pytorch_model.bin', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default=root+'tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--repetition_penalty', default=1.5, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--max_len', type=int, default=50, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    return parser.parse_args()






def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    args = setup_train_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda'  if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    tokenizer.add_tokens([i for i in SPECIAL_TOKENS.values()])
    tokenizer.add_tokens([i for i in name_map_aspect.values()])
    tokenizer.add_tokens([i for i in name_map_action.values()])
 
    config = GPT2Config.from_pretrained(root+ "ul_model_best/model_epoch1/config.json")

    model = GPT2LMHeadModel.from_pretrained(args.model_path,  config=config)
    model.to(device)
    model.eval()
    response = []
    start_context = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_context"])
    end_context = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_context"])
    start_action = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_action"])
    end_action = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_action"])
    start_know = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_know"])
    end_know = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_know"])
    start_response = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_response"])
    end_response = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_response"])
    person = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["user"]) 
    with codecs.open(root +"query.txt", "r", "utf-8") as f1,codecs.open(root+"movie_response.txt", "w", "utf-8") as fout:
        lines = f1.readlines()
        for line in lines:
            line = json.loads(line)
            try:
#            print(line["_source"]["event"])
                text = line["_source"]["event"]["payload"]["body"]["directives"][0]["payload"]["text"]["text"]
                if len(text) >200:
                    continue
            except Exception as e:
                continue
    
            context =  [person] + [tokenizer.convert_tokens_to_ids(word) for word in text]

            dialogue_ids = [start_context] + context + [end_context] + [start_action]+ [end_action] + [start_know] + [end_know]  + [start_response]
            curr_input_tensor = torch.tensor(dialogue_ids).long().to(device)
            generated = []

            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = model(input_ids=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=1, top_p=0)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == 13324:  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token.item())
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            text_1 = tokenizer.convert_ids_to_tokens(generated)
            fout.write("".join(text)+"\n")
            fout.write("".join(text_1)+"\n\n")


if __name__ == '__main__':
    main()
