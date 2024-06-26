import os
import torch
import utils
import logging
import torch.nn as nn
from tqdm import trange
from preprocess import DataLoader
from NER import BertNER
from transformers.optimization import get_linear_schedule_with_warmup, AdamW


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_epoch(model, data_iterator, optimizer, scheduler, params):
    """训练模型，一个epoch"""
    model.train()

    # 定义平均损失
    loss_avg = utils.RunningAverage()
    
    one_epoch = trange(params.train_steps)
    for batch in one_epoch:
        # 获取下一个batch数据
        batch_data, batch_token_starts, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0) # get padding mask

        # 计算损失
        loss = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]

        # 清除梯度，BP计算梯度
        model.zero_grad()
        loss.backward()

        # 裁剪梯度
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # 更新参数
        optimizer.step()
        scheduler.step()

        # 更新平均损失
        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    
def evaluate(model, data_iterator, params):
    """评估模型"""
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    loss_avg = utils.RunningAverage()

    for _ in range(params.eval_steps):
        # 获取下一个batch
        batch_data, batch_token_starts, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)
        
        loss = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
        loss_avg.update(loss.item())
        
        batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]
        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = batch_tags.to('cpu').numpy()

        pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)])
        true_tags.extend([[idx2tag.get(idx) if idx != -1 else 'O' for idx in indices] for indices in batch_tags])

    metrics = {}
    f1 = utils.f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- eval metrics: " + metrics_str)
    return metrics   

def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir):
    """训练模型，每一个epoch评估一次"""
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # 计算每个epoch的训练步数和验证步数
        params.train_steps = params.train_size // params.batch_size
        params.val_steps = params.val_size // params.batch_size

        # 创建一个迭代器
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)

        # 训练单个epoch
        train_epoch(model, train_data_iterator, optimizer, scheduler, params)

        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, val_data_iterator, params, mark='Val')
        
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:    
            logging.info("renew F1")
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # 如果没有提升，则提前结束
        if (patience_counter >= params.patience_num) or epoch == params.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break



if __name__ == '__main__':
    model_dir = "model"
    json_path = "params.json"
    params = utils.Params(json_path)

    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    utils.set_logger('train.log')
    logging.info("device: {}".format(params.device))

    data_loader = DataLoader(params, token_pad_idx=0, tag_pad_idx=-1)
    
    logging.info("Loading the datasets...")

    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('dev')

    params.train_size = train_data['size']
    params.val_size = val_data['size']
    
    logging.info("Loading BERT model...")

    # 定义模型
    model = BertNER.from_pretrained("bert-base-chinese", num_labels=len(params.tag2idx))
    model.to(params.device)

    # 定义优化器
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay': params.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
    ]
    

    optimizer = AdamW(optimizer_grouped_parameters, lr=params.learning_rate, correct_bias=False)
    train_steps_per_epoch = params.train_size // params.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=params.epoch_num * train_steps_per_epoch)

    # 开始训练
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir)
