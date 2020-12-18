import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
import codecs
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from collections import defaultdict


PAD = '[PAD]'
pad_id = 0
logger = None

name_map_aspect = {"电影名": "moviename","导演": "director","演员名":"actor", "类型":"movie_type" ,"国家":"country" ,"上映时间":"time" , "角色":"role","剧情":"plot","台词":"lines","奖项":"award" ,
            "票房":"income","评分":"rating","资源":"website", "音乐":"music", "其他":"aspect_others"}
name_map_action ={"请求事实":"request_fact", "请求推荐":"request_rec","请求感受":"request_feeling","告知事实":"inform_fact","告知推荐":"inform_rec","告知感受":"inform_feeling","其他":"others"}
root="./"
#root = "./"
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
    parser.add_argument('--train_raw_path', default=root+'movie_data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default=root+'movie_data/train_tokenized_0715.txt', type=str,
                        required=False,
                        help='将原始训练预料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default=root+'training.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--raw', action='store_true', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=4000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=50, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_dir', default=root+'ul_model_best/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default=root+'tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    parser.add_argument('--prefix_length', type=int, default=20)
    parser.add_argument('--sequence_tune_rate', type=float, default=0.1)
    parser.add_argument('--continuation_length', type=int, default=10)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--top-p', type=float, default=0.0)
    parser.add_argument('--sequence-ngram-n', type=int, default=1)
    return parser.parse_args()


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

def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def create_model(args, vocab_size):
    """

    :param args:
    :param vocab_size:字典大小
    :return:
    """
    if args.pretrained_model:  # 如果指定了预训练的GPT2模型
        config = GPT2Config.from_pretrained(root+ "pretrain_model/config.json")
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model, config=config)
    else:  # 若没有指定预训练模型，则初始化模型
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")

def preprocess_raw_data(args, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    logger.info("tokenizing raw data,raw data path:{}, token output path:{}".format(args.train_raw_path,
                                                                                    args.train_tokenized_path))

    with codecs.open(args.train_raw_path, 'r', encoding='utf-8') as f, codecs.open(args.train_tokenized_path, 'w', encoding='utf-8') as f1:
        start_context = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_context"])
        end_context = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_context"])
        start_action = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_action"])
        end_action = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_action"])
        start_know = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_know"])
        end_know = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_know"])
        start_response = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["start_response"])
        end_response = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end_response"])
        for index, line in enumerate(f):
            movie_json = json.loads(line.strip())
            #            print(movie_json)
            context = []
            for i, segment in enumerate(movie_json):
                person = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["user"]) if i % 2 == 0 else tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["system"])
#                movie_name = segment["movie_name"]
                person = [person]
                print(person)
                utterance = segment["utter"].replace("《","").replace("》","")
                utterance = [tokenizer.convert_tokens_to_ids(word) for word in utterance]
                label = segment["label"].split("\u0001")
                token_label = []
                for labe in label:
                    if labe == "其他":
                        token_label = [tokenizer.convert_tokens_to_ids(name_map_action[labe])]
                    else:
                        #try:
                            labe =labe.split("-")
                            if len(labe) == 2:
                                labe_left = tokenizer.convert_tokens_to_ids(name_map_action[labe[0]])
                                labe_right = tokenizer.convert_tokens_to_ids(name_map_aspect[labe[1]])
                            else:
                                print(labe)
                                print(segment)
                                continue
                        #except Exception:
                        #    print(labe) 
                        #    print(Exception)
                        #    continue
                knowledge = []
                if "knowledge" in segment.keys():
                    knowledge = segment["knowledge"][0]
                    knowledge = [tokenizer.convert_tokens_to_ids(word) for word in knowledge]
                dialogue_ids = [start_context] + context + [end_context] + [start_action] + token_label + [end_action] + [start_know] + knowledge + [end_know]  + [start_response] + utterance + [end_response]

                if context == []:
                    context = context + person + utterance
                    continue
                context = context + person + utterance
                dialogue_ids = dialogue_ids[:n_ctx]
                #text = tokenizer.convert_ids_to_tokens(dialogue_ids)
                #print(text)
                for dialogue_id in dialogue_ids:
                    f1.write(str(dialogue_id) + ' ')
                f1.write("\n")

    logger.info("finish preprocessing raw data,the result is stored in {}".format(args.train_tokenized_path))


def calculate_loss_and_accuracy(outputs, labels, device):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)

def sample_sequence(model, prefix_batch, prefix_length, continuation_length, top_k, top_p):
    continuation_logits = []
    context = prefix_batch
    assert context.size(1) == prefix_length

    prev = context
    output = context
    past = None
    for i in range(continuation_length):
        logits, past = model.forward(input_ids=prev)
        logits = logits[:, -1, :]
        if top_k == 1 and top_p == 0:
            prev = logits.argmax(dim=1, keepdim=True)
        else:
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            prev = F.softmax(filtered_logits, dim=-1).multinomial(num_samples=1)

        continuation_logits.append(logits)
        output = torch.cat((output, prev), dim=1)

    continuation_logits = torch.stack(continuation_logits, 1)
    return output, continuation_logits

def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask



def ul_seq(model, batch, args):
    batch = batch[:, :args.prefix_length]
    completions, continuation_logits = sample_sequence(model, batch,
                                                       args.prefix_length, args.continuation_length, args.top_k, args.top_p)
    pred_toks = completions[:, args.prefix_length:].contiguous()

    mask = ngram_repeat_mask(pred_toks, args.sequence_ngram_n).type_as(continuation_logits)

    lprobs = F.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    logging_output = {
        'seq_loss': loss.item(),
        'seq_sample_size': ntokens,
        'seq_ntokens': ntokens,
        'seq_nsentences': batch.size(0),
        'seq_repeat_mask': mask.sum().item(),
    }

    # Sum each statistic, which will be normalized by the number of sentences in `aggregate_logging_outputs`.

    loss = loss / ntokens
    return loss, logging_output


def train(model, device, train_list,test_list, multi_gpu, args):
    train_dataset = MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_fn, drop_last = True)
    model.train()
    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    logger.info('starting training')
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录tensorboardX
    # 记录 out of memory的次数
    oom_time = 0
    # 开始训练
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, input_ids in enumerate(train_dataloader):
           # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            input_ids =input_ids.to(device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                outputs = model.forward(input_ids=input_ids)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)
                if torch.rand(1).item() < args.sequence_tune_rate:
                    print(input_ids)
                    if input_ids.size(1) < args.prefix_length:
                        continue
                    loss, batch_metrics = ul_seq(model, input_ids, args)


                if multi_gpu:
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    accuracy = accuracy / args.gradient_accumulation
                loss.backward()
                # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # 进行一定step的梯度累计之后，更新参数
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    # 更新参数
                    optimizer.step()
                    # 清空梯度信息
                    optimizer.zero_grad()
                    # 进行warm up
                    scheduler.step()
                    overall_step += 1
                    # 更新日志与tnesorboardX信息
                    if (overall_step + 1) % args.log_step == 0:
                        logger.info(
                            "batch {} of epoch {}, loss {}, accuracy {}".format(batch_idx + 1, epoch + 1, loss,
                                                                                accuracy))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
            if batch_idx % 5000 == 0: 
                logger.info('saving model for epoch {}'.format(epoch + 1))
                model_path = args.output_dir + 'model_epoch{}'.format(epoch + 1)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(model_path)
                logger.info('epoch {} temp finished'.format(epoch + 1))
 
        logger.info('saving model for epoch {}'.format(epoch + 1))
        model_path = args.output_dir + 'model_epoch{}'.format(epoch + 1)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
        logger.info('Evaluation start:')
        evaluate(model, device, test_list, multi_gpu, args)
    logger.info('training finished')

def evaluate(model, device, test_list, multi_gpu, args):
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    # 记录tensorboardX
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=collate_fn, drop_last = True )
    total_loss = 0
    total_batch = 0
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(test_dataloader):
            total_batch = batch_idx
            input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)

            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            total_loss = total_loss + loss  
            #logger.info("evaluate batch {} ,loss {} ,accuracy {}".format(batch_idx, loss, accuracy))
            # tb_writer.add_scalar('loss', loss.item(), overall_step)
        logger.info("finishing evaluating")
        logger.info("evaluate average loss {}".format(total_loss / total_batch))



def main():
    args = setup_train_args()
    # 日志同时输出到文件和console
    global logger
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed(args)

    # 设置使用哪些显卡进行训练
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    # important
    tokenizer.add_tokens([i for i in SPECIAL_TOKENS.values()])
    tokenizer.add_tokens([i for i in name_map_aspect.values()])
    tokenizer.add_tokens([i for i in name_map_action.values()])
    # tokenizer的字典大小
    vocab_size = len(tokenizer)

    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)

    # 创建modle的输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 加载dialogue GPT2模型
    #if args.raw:
    #preprocess_raw_data(args, tokenizer, 1024)
    model, n_ctx = create_model(args, vocab_size)
    model.to(device)
    # 对原始数据进行预处理,将原始语料转换成对应的token_id

    # 是否使用多块GPU进行并行运算
    multi_gpu = False
    if args.cuda and torch.cuda.device_count() > 1:
        logger.info("Let's use GPUs to train")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    # 记录模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 加载数据
    logger.info("loading traing data")
    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data = f.read()
    	
    data_list = data.split("\n")
    new_data_list = []
    for da in data_list:
        if len(da.strip().split(" ")) < 300:
            new_data_list.append(da)
    train_list, test_list = train_test_split(new_data_list, test_size=0.05, random_state=1)
    # 开始训练
    train(model, device, train_list,test_list,  multi_gpu, args)
    # 测试模型
    evaluate(model, device, test_list, multi_gpu, args)


if __name__ == '__main__':
    main()
