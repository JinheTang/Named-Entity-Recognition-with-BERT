from NER import BertNER
import utils as utils
import torch
from preprocess import DataLoader
import numpy as np

def inference(model, data_iterator, idx2tag):
    model.eval()
    # 存储所有批次的预测标签
    all_pred_tags = []
    # 迭代整个数据集
    for batch_data, batch_token_starts in data_iterator:
        batch_masks = batch_data.gt(0)
        # 获取模型输出
        batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]
        batch_output = batch_output.detach().cpu().numpy()
        # 获取当前批次的预测标签
        pred_tags = [[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)]
        # 将当前批次的预测标签添加到总列表中
        all_pred_tags.extend(pred_tags)
    
    # 返回整个数据集的预测标签列表
    return all_pred_tags

if __name__ == '__main__':
    params = utils.Params("params.json")
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertNER.from_pretrained("model")
    model.to(params.device)
    data_loader = DataLoader(params, token_pad_idx=0, tag_pad_idx=-1)
    test_data = data_loader.load_data('test')
    data_iterator = data_loader.data_iterator(test_data, shuffle=False)
    idx2tag = data_loader.idx2tag
    pred_tags = inference(model, data_iterator, idx2tag)
    with open("test_pred.txt", 'w') as file:
        for tags in pred_tags:
            file.write(" ".join(tags) + "\n")
