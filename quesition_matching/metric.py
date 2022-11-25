from tqdm import tqdm
from Levenshtein import distance
import numpy as np
special_tokens = ["<spt>","<ans>",]

label_names = ['input','output']

class Metric():
    def __init__(self,tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def match(self,pred_list:list,tgt_list:list):
        distance_mat = np.zeros((len(pred_list),len(tgt_list)))
        for i,pred in enumerate(pred_list):
            for j,tgt in enumerate(tgt_list):
                distance_mat[i,j] = distance(pred,tgt)

        match_tgt_list = np.argmin(distance_mat,axis=1)
        match_dis_list = np.min(distance_mat,axis=1)

        match_result = {}
        
        for pred,tgt_idx,tgt_dis in zip(pred_list,match_tgt_list,match_dis_list):
            tgt = tgt_list[tgt_idx]
            if tgt not in match_result.keys():
                match_result[tgt] = (pred,tgt_dis)
            elif tgt in match_result.keys() and tgt_dis < match_result[tgt][1]:
                match_result[tgt] = (pred,tgt_dis)

        match_pair_list = []
        for k,v in match_result.items():
            match_pair_list.append((k,v[0]))
        
        # not allow extra num predict
        penalty1 = len(tgt_list) / len(pred_list) if len(pred_list) > len(tgt_list) else len(pred_list) / len(tgt_list)
        # not allow multiple prediction to point same tgt
        penalty2 = len(match_pair_list) / len(pred_list) 

        return match_pair_list,penalty1,penalty2


    def seq_f1(self,y_pred,y_tgt):
        """
        :param y_pred: [n_samples]
        :param y_tgt: [n_samples]
        :return: 严格F1(exact match)和松弛F1(字符匹配率)
        """
        exact_match_cnt = 0
        char_match_cnt = 0
        char_pred_sum = char_tgt_sum = 0
        for pred,tgt in zip(y_pred,y_tgt):
            if pred == tgt:
                exact_match_cnt += 1
            char_pred_sum += len(pred)
            char_tgt_sum += len(tgt)
            for char in pred:
                if char in tgt:
                    char_match_cnt += 1
        em_acc = exact_match_cnt / (len(y_tgt)+0.001)
        char_acc = char_match_cnt / (char_pred_sum+0.001)
        char_recall = char_match_cnt / (char_tgt_sum+0.001)
        char_f1 = 0
        if char_acc + char_recall != 0:
            char_f1 = 2 * char_acc * char_recall / (char_acc + char_recall)

        return em_acc,char_f1

    def metric(self,eval_predict):
        predictions = eval_predict.predictions
        label_ids = eval_predict.label_ids

        em_cnt = 0
        pred_str_list = []
        tgt_str_list = []
        for pred,label in tqdm(zip(predictions,label_ids),total=len(predictions)):
            pred_str = self.tokenizer.decode(pred[pred>0],skip_special_tokens=True)
            label_str = self.tokenizer.decode(label[label>0],skip_special_tokens=True)
            if pred_str == label_str:
                em_cnt+=1
            
            pred_str_list.append(pred_str.split(' <ans> '))
            tgt_str_list.append(label_str.split(' <ans> '))
        
        metric_dict = self.str_metric(pred_str_list,tgt_str_list)
        metric_dict['acc'] = float(em_cnt/len(predictions))
        
        print(pred_str,label_str)
        print(metric_dict)
        return metric_dict

    def str_metric(self,pred_str_list,tgt_str_list):    
        em_acc_total = 0
        char_f1_total = 0
        for pred,tgt in zip(pred_str_list,tgt_str_list):
            if len(pred) == 0:
                if (len(tgt) == 1 and len(tgt[0]) == 0) or len(tgt) == 0:
                    em_acc_total += 1
                    char_f1_total += 1
                else:
                    em_acc_total += 0
                    char_f1_total += 0
                continue
            pair,p1,p2 = self.match(pred,tgt)
            em_acc,char_f1 = self.seq_f1([unit[1] for unit in pair],[unit[0] for unit in pair])
            em_acc_total += (em_acc * p1 * p2)
            char_f1_total += (char_f1 * p1 * p2)

        return {
            'em_acc_with_penalty':float(em_acc_total/len(pred_str_list)),
            'char_f1_with_penalty':float(char_f1_total/len(pred_str_list))
        }