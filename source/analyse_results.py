import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

import argparse
from datetime import datetime

import const
from folds import folds

# Parseando argumentos
###############################################################################

parser = argparse.ArgumentParser()

parser.add_argument('model')
parser.add_argument('-d', '--date', default='today')
parser.add_argument('-f', '--folds', default='all')

args = parser.parse_args()

if args.folds != 'all':
    try:
        # foi passado apenas um inteiro como argumento
        folds = [ folds[ int(args.folds) ] ]

    except:
        args.folds = args.folds.replace('[', '')
        args.folds = args.folds.replace(']', '')

        if ':' in args.folds:
            # foi passado um slice como argumento
            lim = args.folds.split(':')
            lim[0] = None if lim[0] == '' else int(lim[0])
            lim[1] = None if lim[1] == '' else int(lim[1])
            folds = folds[ lim[0] : lim[1] ]

        elif ',' in args.folds:
            # foi passada uma lista como argumento
            args.folds = args.folds.split(',')
            folds = [ folds[ int(f) ] for f in args.folds ]

        else:
            raise Exception(f'Unknown folds choice formatting {args.folds}')

if args.date == 'today':
    date = datetime.now().strftime("%y%m%d")
else:
    date = args.date

model = args.model

###############################################################################

colors = const.colors

###############################################################################

def get_folds_mean(row, mode):

    m = 0

    for fold in folds:
        m += row[ fold['name'] + '_' + mode ]

    return m / len(folds)

###############################################################################

# Gerando o grafico com a media de todos os folds

model_dir = os.path.join(date, model)

const.RESULTS_DIR = os.path.join( const.RESULTS_DIR, model_dir)

if not os.path.isdir(const.RESULTS_DIR):
    raise Exception(f'Directory not found: {const.RESULTS_DIR}')

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
index = []

for fold in folds:
    best_losses.append( round( history[ fold['name'] + '_val' ].min(), 3) )
    best_epochs.append( history[ fold['name'] + '_val' ].idxmin() )
    avg_times.append( round(history[ fold['name'] + '_time' ].mean(), 2) )
    total_times.append( round(history[ fold['name'] + '_time' ].sum()) )
    index.append( fold['index'] )

total_times = [ f'{t//60}:{t%60}' for t in total_times ]

summary = pd.DataFrame()
summary['best_val_loss'] = best_losses
summary['best_epoch'] = best_epochs
summary['avg_time'] = avg_times
summary['total_time'] = total_times
summary.index = index
summary.index.name = 'fold'

summary.to_csv( os.path.join(const.RESULTS_DIR, 'summary.csv'), sep=';', decimal=',' )