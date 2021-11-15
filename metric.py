from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import classification_report

def classificationM(reference_list, prediciton_list):
    # print(classification_report(reference_list, prediciton_list))
    f1_score(reference_list, prediciton_list, average='micro')

    micro_accuracy = accuracy_score(reference_list, prediciton_list)
    micro_precision = precision_score(reference_list, prediciton_list, average="micro")
    micro_recall = recall_score(reference_list, prediciton_list, average="micro")
    micro_f1 = f1_score(reference_list, prediciton_list, average="micro")

    macro_accuracy = accuracy_score(reference_list, prediciton_list)
    macro_precision = precision_score(reference_list, prediciton_list, average="macro")
    macro_recall = recall_score(reference_list, prediciton_list, average="macro")
    macro_f1 = f1_score(reference_list, prediciton_list, average="macro")

    weighted_accuracy = accuracy_score(reference_list, prediciton_list)
    weighted_precision = precision_score(reference_list, prediciton_list, average="weighted")
    weighted_recall = recall_score(reference_list, prediciton_list, average="weighted")
    weighted_f1 = f1_score(reference_list, prediciton_list, average="weighted")

    return (micro_accuracy, micro_precision, micro_recall, micro_f1), (macro_accuracy, macro_precision, macro_recall, macro_f1), (weighted_accuracy, weighted_precision, weighted_recall, weighted_f1)

# reference_list = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1]]
# prediciton_list = [[1,0,0],[1,0,0],[1,1,1],[1,0,0],[0,1,1]]

# print(classificationM(reference_list, prediciton_list))
# ((0.2, 0.5, 0.5714285714285714, 0.5333333333333333), (0.2, 0.5, 0.5555555555555555, 0.5238095238095238), (0.2, 0.5, 0.5714285714285714, 0.5306122448979592))