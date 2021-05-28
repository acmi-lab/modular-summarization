
from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


import numpy as np

import sklearn


from utils.section_names import ami_section_names
import json


@Metric.register("all-classification-metric")
class AllClassificationMetric(Metric):
    def __init__(self, num_classes:int, dataset="abridge") -> None:
        self.num_classes=num_classes
        self.y_true=[]
        self.y_pred_class=[]
        self.y_pred_cont=[]
        self.dataset=dataset

    def __call__(self,
                 gold_labels: torch.Tensor,
                 predictions: torch.Tensor,                 
                 mask: Optional[torch.Tensor] = None,
                 thresholds: List[float]= None):
        """
        Parameters
        ----------
        predictions : shape is num_datapointsXnum_classes
        gold_labels : shape is num_datapointsXnum_classes
        mask: ``torch.Tensor``, optional (default = None).
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        predictions=predictions.detach().cpu().numpy()
        gold_labels=gold_labels.detach().cpu().numpy()
        
        assert predictions.ndim==2
        assert predictions.shape[1]==self.num_classes
        assert gold_labels.ndim==2
        assert gold_labels.shape[1]==self.num_classes        


        if mask is not None:
            assert mask.ndim==1
            mask = mask.cpu().numpy()        
            gold_labels = gold_labels[mask]
            predictions = predictions[mask]       
            
        
        self.y_true.append(gold_labels)
        self.y_pred_cont.append(predictions)

        if type(thresholds)==type(None):
            self.y_pred_class.append(np.round(predictions, decimals=0))
        else:
            self.y_pred_class.append((predictions >= thresholds).astype(int))
        

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated metrics
        """
        all_results_dict={"nothing":0.0}
        if reset:
            if self.dataset=="ami":
                label_names = ami_section_names
            elif self.dataset=="unary_prediction":
                label_names = ["important"]
            else:
                raise NotImplementedError

            # pdb.set_trace()
            all_results_dict = calc_metrics(np.concatenate(self.y_true, axis=0),
                                            np.concatenate(self.y_pred_class, axis=0),
                                            np.concatenate(self.y_pred_cont, axis=0),
                                            label_names)
            self.reset()
            
        return all_results_dict

    @overrides
    def reset(self):
        self.y_true=np.empty(shape=(0, self.num_classes))
        self.y_pred_class=np.empty(shape=(0, self.num_classes))
        self.y_pred_cont=np.empty(shape=(0, self.num_classes))




def precision_at_1(y_true, y_pred):
    assert y_pred.dtype==np.float32 or y_pred.dtype==np.float64 or y_pred.dtype==np.float16
    most_probable_pred=np.argmax(y_pred, axis=1)
    most_probable_pred=most_probable_pred.reshape(-1)
    
    hit_placewise=np.zeros(y_true.shape[1])
    
    hits=0
    total=0

    for idx, arr in zip(most_probable_pred, y_true):
        ispresent=arr[idx]
        if ispresent:
            hit_placewise[idx]+=1
        hits+=ispresent
        total+=1

    return hits/total, hit_placewise


def calc_metrics( y_true, y_pred_class, y_pred_cont, label_names):
    return_dict={}

#     print( {"y_true": y_true.shape, 
#             "y_pred_class": y_pred_class.shape,
#             "y_pred_cont": y_pred_cont.shape } )    

    num_points=y_true.shape[0]
    num_classes=y_true.shape[1]


    return_dict["aggregate_micro-precision"]=sklearn.metrics.precision_score(y_true, y_pred_class, average="micro")
    return_dict["aggregate_macro-precision"]=sklearn.metrics.precision_score(y_true, y_pred_class, average="macro")
    
    return_dict["aggregate_micro-recall"]=sklearn.metrics.recall_score(y_true, y_pred_class, average="micro")    
    return_dict["aggregate_macro-recall"]=sklearn.metrics.recall_score(y_true, y_pred_class, average="macro")
    
    return_dict["aggregate_micro-f1"]=sklearn.metrics.f1_score(y_true, y_pred_class, average="micro")
    return_dict["aggregate_macro-f1"]=sklearn.metrics.f1_score(y_true, y_pred_class, average="macro")
    
    return_dict["aggregate_accuracy"]=sklearn.metrics.accuracy_score(y_true.flatten(), y_pred_class.flatten())
    
    return_dict["aggregate_micro-auc"]=sklearn.metrics.roc_auc_score(y_true, y_pred_cont, average="micro")
    return_dict["aggregate_macro-auc"]=sklearn.metrics.roc_auc_score(y_true, y_pred_cont, average="macro")        

    classwise_results={}
    classwise_results["precision"]=sklearn.metrics.precision_score(y_true, y_pred_class, average=None)
    classwise_results["recall"]=sklearn.metrics.recall_score(y_true, y_pred_class, average=None)
    classwise_results["f1"]=sklearn.metrics.f1_score(y_true, y_pred_class, average=None)
    classwise_results["accuracy"]=np.array([sklearn.metrics.accuracy_score(y_true[:,_i], y_pred_class[:,_i]) 
                                                    for _i in range(num_classes)   ])
    classwise_results["auc"]=sklearn.metrics.roc_auc_score(y_true, y_pred_cont, average=None)
    
    if classwise_results["auc"].shape==():        # if theres only one class it returns a scalar
        classwise_results["auc"]=np.array([classwise_results["auc"]])
    
#     print("##################")
#     print(classwise_results["auc"].shape)
#     print("##################")

    # pdb.set_trace()
    
    precision_at_1_calculated = precision_at_1(y_true, y_pred_cont)    
    return_dict["aggregate_precision-at-1"]=precision_at_1_calculated[0]
    piecwise_contribution_to_p1=precision_at_1_calculated[1]
    
    classwise_dfdict={}
    for _i in range(num_classes):
        heading=label_names[_i]
        classwise_dfdict[heading]={}
        classwise_dfdict[heading]["prevalence_rate"]=sum(y_true[:,_i]/num_points)        
        classwise_dfdict[heading]["precision"]=classwise_results["precision"][_i]
        classwise_dfdict[heading]["recall"]=classwise_results["recall"][_i]
        classwise_dfdict[heading]["f1"]=classwise_results["f1"][_i]
        classwise_dfdict[heading]["accuracy"]=classwise_results["accuracy"][_i]
        classwise_dfdict[heading]["auc"]=classwise_results["auc"][_i]   
        classwise_dfdict[heading]["contribution_to_p1"]=piecwise_contribution_to_p1[_i]/sum(piecwise_contribution_to_p1)   
        
    dict_toprint={
        "classwise_results": classwise_dfdict,
        "aggregate_results": return_dict,
        }
    print(json.dumps(dict_toprint))
 

    for _i in range(num_classes):
        return_dict["classwise_"+str(_i)+"_prevalence_rate"]=sum(y_true[:,_i]/num_points)        
        return_dict["classwise_"+str(_i)+"_precision"]=classwise_results["precision"][_i]
        return_dict["classwise_"+str(_i)+"_recall"]=classwise_results["recall"][_i]
        return_dict["classwise_"+str(_i)+"_f1"]=classwise_results["f1"][_i]
        return_dict["classwise_"+str(_i)+"_accuracy"]=classwise_results["accuracy"][_i]
        return_dict["classwise_"+str(_i)+"_auc"]=classwise_results["auc"][_i]        
        
        
    
    return return_dict
