import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def categories_histogram_YelpB(csv_file_path):
    """ 
    This function creates a statitics results of the dataset given the categories
        It draws the category histogram.
        Params:
            @csv_file_path: a string represents the CSV file path
        Return:
            void  
    """
    df = pd.read_csv('a.csv')
    print(df.info())
    print(df.head(10))
    cats = np.array([0, 0, 0, 0, 0, 0, 0])
    ind_s = 0
    ind_a = 1
    ind_f = 2
    ind_a_f = 3
    ind_a_s = 4
    ind_s_f = 5
    ind_a_s_f = 6

    for i in range(0, len(df)):
        cat = df.iloc[i, :]['categories']
        if 'Fast Food' in cat and 'Sushi Bars' not in cat and 'American (New)' not in cat:
            cats[ind_f] += 1
        if 'Fast Food' not in cat and 'Sushi Bars' in cat and 'American (New)' not in cat:
            cats[ind_s] += 1
        if 'Fast Food' not in cat and 'Sushi Bars' not in cat and 'American (New)' in cat:
            cats[ind_a] += 1
        if 'Fast Food' in cat and 'Sushi Bars' in cat and 'American (New)' not in cat:
            cats[ind_s_f] += 1
        if 'Fast Food' in cat and 'Sushi Bars' not in cat and 'American (New)' in cat:
            cats[ind_a_f] += 1
        if 'Fast Food' not in cat and 'Sushi Bars' in cat and 'American (New)' in cat:
            cats[ind_a_s] += 1
        if 'Fast Food' in cat and 'Sushi Bars' in cat and 'American (New)' in cat:
            cats[ind_a_s_f] += 1

    print(cats)

    labels = ['Sushi Bars', 'American (New)', 'Fast Food', 'A & F', 'A & S', 'S & F', 'A & S & F']

    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots()
    rects1 = ax.bar(x + width/2, cats, width,color=['red', 'yellow', 'green', 'blue', 'cyan', 'black'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number')
    ax.set_title('Restaurants Category Histogram')
    ax.legend()
    autolabel(rects1, ax)
    plt.show()