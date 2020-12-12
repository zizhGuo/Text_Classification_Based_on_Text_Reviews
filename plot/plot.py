import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def autolabel(rects, ax):
    """
        Attach a text label above each bar in *rects*, displaying its height.
        @ref: https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_dot_curve(sizes, precision, recall, f1, acc, title):
    """ 
    This function draws a line images
        Params:
            @sizes: a list of samples number
            @precision: a list of the precision results
            @recall: a list of the recall results
            @f1: a list of the f1 results
            @acc: a list of the accuracy results
            @title: a string of the title
        Return:
            void  
    """
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 500)
    y = np.sin(x)

    fig, ax = plt.subplots()

    # Using set_dashes() to modify dashing of an existing line
    line1, = ax.plot(sizes, precision,'rs-',label='precision')
    # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    # Using plot(..., dashes=...) to set the dashing when creating a line
    line2, = ax.plot(sizes, recall, 'o-', label='recall')
    line3, = ax.plot(sizes, f1, 'bp-', label='f1')
    line4, = ax.plot(sizes, acc, 'y*-', label='acc')
    # line2, = ax.plot(sizes, recall, dashes=[6, 2], label='Using the dashes parameter')
    ax.set_title(title)
    ax.set_ylabel('Percentage')
    ax.legend()
    plt.show()

def plot_bar():
    """ 
    This function draws annotated bar chart.
        Params:
            void
        Return:
            void  
    """
    precision = [0.70, 0.71, 0.75]
    recall = [0.69, 0.71, 0.75]
    f1 = [0.70, 0.71, 0.75]
    acc = [0.69, 0.71, 0.75]
    x1 = np.arange(3)
    # x2 = ['Tf-Idf + MultinomialNB', 'LDA + MultinomialNB']
    x2 = ['No Removing stopwords + No Stemming', 'Removing stopwords + No Stemming', 'Removing stopwords + Stemming']
    
    width = 0.1
    fig, ax = plt.subplots()
    rects1 = ax.bar(x1 - width * 2, precision, width, label='precision')
    rects2 = ax.bar(x1 - width, recall, width, label='recall')
    rects3 = ax.bar(x1, f1, width, label='f1 score')
    rects4 = ax.bar(x1 + width, acc, width, label='accuracy')
    ax.set_xticks(x1)
    ax.set_xticklabels(x2)
    ax.set_ylabel('Percentage')
    ax.set_title('Comparison the performances between different texts processing stages')
    # ax.set_title('Comparison of different text models on the same dataset & classifiers')
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)
    plt.show()