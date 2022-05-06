import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

import const
from folds import folds


###############################################################################

# Cores bonitas para serem usadas no grafico
# [REF] https://towardsdatascience.com/making-matplotlib-beautiful-by-default-d0d41e3534fd

CB91_Blue   = '#2CBDFE'
CB91_Green  = '#47DBCD'
CB91_Pink   = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber  = '#F5B14C'

colors = [CB91_Amber,
            CB91_Purple,
            CB91_Green,
            CB91_Pink,
            CB91_Blue,
            CB91_Violet]

###############################################################################

def get_folds_mean(row, mode):

    m = 0

    for fold in folds:
        m += row[ fold['name'] + '_' + mode ]

    return m / len(folds)

###############################################################################

# Gerando o grafico com a media de todos os folds

model_dir = os.path.join('220505', 'vgg16')

const.RESULTS_DIR = os.path.join( const.RESULTS_DIR, model_dir)

history = pd.read_csv( os.path.join(const.RESULTS_DIR, 'history.csv') )

for fold in folds:

    plt.plot(history[ fold['name'] + '_trn' ],
             color=colors[0],
             alpha=0.2,
             label='trn')

    plt.plot(history[ fold['name'] + '_val' ],
             color=colors[1],
             alpha=0.2,
             label='val')

history['trn_mean'] = history.apply( lambda row: get_folds_mean(row, 'trn'), axis=1)
history['val_mean'] = history.apply( lambda row: get_folds_mean(row, 'val'), axis=1)

plt.plot(history['trn_mean'],
             color=colors[0],
             alpha=1.0,
             label='trn')

plt.plot(history['val_mean'],
             color=colors[1],
             alpha=1.0,
             label='val')

###############################################################################

# Fazemos umas customizacoes do grafico para deixa-lo mais bonito
plt.title(model_dir)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Definindo a legenda manualmente
# [REF] https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
legend_lines = [Line2D( [0], [0], color=colors[0]),
                Line2D( [0], [0], color=colors[1])]
plt.legend(legend_lines, ['TRN','VAL'])

plt.ylim(0,3)

plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.savefig( os.path.join(const.RESULTS_DIR, 'history_plot.png') )

###############################################################################

# Gerando o resumo

best_losses = []
best_epochs = []
avg_times = []
total_times = []

for fold in folds:
    best_losses.append( round( history[ fold['name'] + '_val' ].min(), 3) )
    best_epochs.append( history[ fold['name'] + '_val' ].idxmin() )
    avg_times.append( round(history[ fold['name'] + '_time' ].mean(), 2) )
    total_times.append( round(history[ fold['name'] + '_time' ].sum()) )

total_times = [ f'{t//60}:{t%60}' for t in total_times ]

summary = pd.DataFrame()
summary['best_val_loss'] = best_losses
summary['best_epoch'] = best_epochs
summary['avg_time'] = avg_times
summary['total_time'] = total_times
summary.index.name = 'fold'

summary.to_csv( os.path.join(const.RESULTS_DIR, 'summary.csv') )