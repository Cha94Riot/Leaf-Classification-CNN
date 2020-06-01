import csv

def top5(class_probs, labels):
    #labels = open('classList.csv', 'r' )

    class_probs = class_probs[0]
    top_values_index = sorted(range(len(class_probs)), key=lambda i: class_probs[i], reverse=True)[:5]

    print(top_values_index)
    for i in top_values_index:
        print(labels[i], class_probs[i])
